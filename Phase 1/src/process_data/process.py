
#################################
"""
Offline processing orchestrator (Phase 1 steps 8-10).

Capture (capture/capture.py) is the GPU half: it drives CARLA and writes raw
frames to a local scratch directory. This is the CPU half, and the two mirror
each other -- Capture owns a run, Processor owns one split's processing pass.

ONE INVOCATION = ONE SPLIT. The split is a flag on the RUN in metadata.json, so
`--split train` processes exactly the runs flagged train and skips the rest.
Every run must therefore be split-pure; the validator refuses to start if a
run's scenes disagree with its flag, because "no scene spans two splits" is the
claim that makes the test number mean anything.

S3 is written ONCE, at the end, and only the finished KITTI tree goes up. Raw
frames stay on the box: capture and processing share an instance, so publishing
raw would mean ~200k PUTs for data the next stage reads off local disk anyway.

The loop is circular. Process a split, read the histogram, and if a class came
out thin: re-capture scenes that produce it as a new run, then re-run with
--topup-class to mine just the frames carrying that class and APPEND them to the
existing tree.

    python src/process_data/process.py --split train --raw-root /workspace/_scratch \\
        --output-root /workspace/kitti

    python src/process_data/process.py --split train --raw-root /workspace/_scratch \\
        --output-root /workspace/kitti --runs 6 \\
        --topup-class speed_sign_30 --topup-frames 200
"""


### Imports below

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.config import Config

import parser as kitti
from config_loader import CONFIGS, CONFIG_DIR
from viz import read_labels, render_samples


### Provenance helpers below

# Config files whose content hash goes into provenance. metadata.json is excluded:
# this pass WRITES it (the histogram), so hashing it would be self-referential.
_HASHED_CONFIGS = ("CARLA_config.json", "ego_config.json", "scene_description.json")

# The filter knobs live in parser as module constants, not in config, so they are
# recorded here BY VALUE. Otherwise the dataset's own definition of "visible" is
# invisible to anyone reading the data card.
_THRESHOLD_KEYS = (
    "MIN_UNOCCLUDED_FRAC", "MIN_VISIBLE_PIXELS", "MIN_SIGN_PIXELS",
    "MIN_BOX_HEIGHT_PX", "MIN_SIGN_HEIGHT_PX", "DEPTH_TOL_M", "MAX_RANGE_M",
    "VEHICLE_TAGS", "SIGN_TAG",
)

_UPLOAD_WORKERS = 16

# Instance-count target per class, from the Phase 1 doc. Deliberately not in config:
# it is a judgement about the dataset rather than an input to it, and it is the thing
# that decides whether another top-up round is worth GPU time.
TARGET_MIN, TARGET_MAX = 500, 1500

# Per-split floor. A class can clear TARGET_MIN overall and still be untestable,
# because the totals hide WHERE the instances are: a class with thousands in train
# and a handful in val/test yields an evaluation number built on nearly nothing.
# Below this, a per-class AP on that split is noise rather than a measurement.
TARGET_MIN_PER_SPLIT = 50


def _sha256_file(path: Path) -> str:
    """SHA256 of a file's raw bytes (the on-disk config, _notes and all)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit():
    """Current commit, or None outside a checkout. Duplicated from capture.py on
    purpose: importing capture here would drag in `carla`, which the CPU-only
    processing image deliberately does not install."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=CONFIG_DIR,
                             capture_output=True, text=True)
    except (FileNotFoundError, OSError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


### Orchestrator below

class Processor:
    """One offline pass: a split's captured runs -> a KITTI tree, published."""

    def __init__(self, output_root, split, raw_root=None, run_ids=None,
                 configs=CONFIGS, include_lidar=True, samples=8, limit=None,
                 upload=True, topup_class=None, topup_frames=None, topup_seed=0,
                 prefetch=64, max_workers=16, version=None):
        self.output_root = Path(output_root)
        self.split = split
        self.version = version or configs["metadata.json"].get("dataset_version")
        if not self.version:
            raise KeyError(
                "no dataset version: set metadata.json dataset_version or pass --version. "
                "The S3 prefix is versioned so a re-run cannot overwrite a published "
                "dataset, and an unversioned publish would silently do exactly that.")
        self.raw_root = raw_root
        self.configs = configs
        self.include_lidar = include_lidar
        self.samples = samples
        self.limit = limit
        self.upload = upload
        self.topup_class = topup_class
        self.topup_frames = topup_frames
        self.topup_seed = topup_seed
        self.prefetch = prefetch
        self.max_workers = max_workers

        # A top-up ADDS to what is already there -- never restarts the tree.
        self.append = topup_class is not None
        self.tree = self.output_root / split

        scenes = configs["scene_description.json"]["scenes"]
        self.scene_split = {s["scene_id"]: s["split"] for s in scenes}
        self.scene_map = {s["scene_id"]: s["map"] for s in scenes}
        self.scene_tod = {s["scene_id"]: s["time_of_day"] for s in scenes}

        self.runs = configs["metadata.json"]["runs"]
        self.run_ids = self._select_runs(run_ids)
        self.scene_run = {sid: r["run_id"] for r in self.runs for sid in r["scene_ids"]}
        self.scenes = [sid for r in self.runs if r["run_id"] in set(self.run_ids)
                       for sid in r["scene_ids"]]

        self._validate(scenes)
        self._started_utc = None

    def _select_runs(self, run_ids) -> list:
        """
        Runs to process: those flagged with this split, or an explicit list.

        Explicit --runs skips the status check on purpose -- it is how a run
        that capture left 'aborted' (some scenes failed) still gets processed
        for whatever did land.
        """
        if run_ids:
            return list(run_ids)

        missing = [r["run_id"] for r in self.runs if "split" not in r]
        if missing:
            raise KeyError(
                f"runs {missing} have no 'split' flag in metadata.json. The processor "
                "selects work by run-level split; add it, keeping every run split-pure.")

        mine = [r for r in self.runs if r["split"] == self.split]
        chosen = [r["run_id"] for r in mine if r.get("status") == "complete"]
        if chosen:
            return chosen

        # metadata.json's status is only a PROXY for "this run's frames exist".
        # It is also a version-controlled config file, so deploying code overwrites
        # it and wipes the status capture stamped -- the data is still on disk, but
        # the record of it is gone. So fall back to checking the actual precondition
        # rather than the proxy, loudly enough that a genuinely missing capture is
        # still obvious.
        on_disk = [r["run_id"] for r in mine if self._raw_present(r["scene_ids"])]
        if on_disk:
            print(f"[process] WARNING: runs {on_disk} are not marked complete in "
                  f"metadata.json, but their raw frames ARE in {self.raw_root}. "
                  "Using them. (A code deploy overwrites the status capture stamped.)")
            return on_disk

        states = ", ".join(f"run {r['run_id']}: split={r['split']!r} "
                           f"status={r.get('status')!r}" for r in self.runs)
        raise RuntimeError(
            f"no run for split={self.split!r} is marked complete, and none has raw "
            f"frames in {self.raw_root!r} ({states}). Capture it first, or pass --runs.")

    def _raw_present(self, scene_ids) -> bool:
        """True when every scene of a run has its records file in the raw root."""
        if not self.raw_root:
            return False           # S3 source: no cheap local check, trust the status
        naming = self.configs["CARLA_config.json"]["naming_convention"]
        root = Path(self.raw_root)
        return all((root / naming["scene_dir"].format(scene_id=sid) /
                    kitti.RECORDS_FILENAME).is_file() for sid in scene_ids)

    def _validate(self, scene_entries):
        """
        Fail before any work if the split key is unsound.

        Checks the run-level flag against the per-scene one. They are two
        records of the same fact, and a dataset where they disagree is one
        where the test number quietly means nothing.
        """
        ids = [s["scene_id"] for s in scene_entries]
        dupes = [i for i, n in Counter(ids).items() if n > 1]
        if dupes:
            raise ValueError(f"scene_description has duplicate scene_ids: {dupes}")

        unknown = [sid for sid in self.scenes if sid not in self.scene_split]
        if unknown:
            raise ValueError(
                f"runs {self.run_ids} reference scenes {unknown} that "
                "scene_description does not define.")

        mismatched = {sid: self.scene_split[sid] for sid in self.scenes
                      if self.scene_split[sid] != self.split}
        if mismatched:
            raise ValueError(
                f"runs {self.run_ids} are flagged split={self.split!r} but contain scenes "
                f"assigned elsewhere in scene_description: {mismatched}. Runs must be "
                "split-pure -- re-partition the runs rather than mixing.")

        if not self.scenes:
            raise RuntimeError(f"runs {self.run_ids} select no scenes")

    def run(self) -> dict:
        """Write (or extend) this split's tree, verify it, then publish."""
        self._started_utc = datetime.now(timezone.utc).isoformat()
        action = f"top-up {self.topup_class}" if self.append else "write"
        print(f"[process] === split {self.split} : {action} : runs {self.run_ids} "
              f"scenes {self.scenes} ===", flush=True)

        stats = kitti.write_kitti(
            self.tree, run_ids=self.run_ids, configs=self.configs,
            include_lidar=self.include_lidar, limit=self.limit,
            prefetch=self.prefetch, max_workers=self.max_workers,
            scene_ids=self.scenes, raw_root=self.raw_root, append=self.append,
            select_class=self.topup_class, select_n=self.topup_frames,
            select_seed=self.topup_seed,
        )
        print(f"[process] {stats['frames']} frames written "
              f"({stats['frames_in_tree']} in tree)")

        # Count what is ON DISK, not what the writer believes it wrote. After an
        # append this also re-reads the frames from earlier rounds, so the
        # histogram is always the whole tree rather than the last round's slice.
        per_scene = self._histogram_from_tree()
        histogram = Counter()
        for counts in per_scene.values():
            histogram.update(counts)

        if self.samples:
            # Clear before rendering, for the same reason write_kitti clears the tree:
            # renders are named after the frame they came from, so a rewrite that
            # renumbers scenes or changes the class map leaves the previous run's
            # renders sitting beside the new ones. They then get published as part of
            # this dataset while showing frames it does not contain and classes it does
            # not label. Append keeps them, since there the tree is only growing.
            samples_dir = self.output_root / "samples" / self.split
            if not self.append and samples_dir.is_dir():
                shutil.rmtree(samples_dir)
            written = render_samples(self.tree, samples_dir, n=self.samples)
            print(f"[process] {len(written)} sample renders")

        summary = self._write_summary(per_scene, histogram, stats)
        self._stamp_metadata(per_scene)
        merged = merge_summaries(self.output_root, self.configs)
        report = write_histogram_report(self.output_root, self.configs, merged)
        if self.upload:
            self._upload_tree()

        print(f"[process] done. {self.split}: {summary['frames']} frames, "
              f"{summary['objects']} objects -> {self.tree}")
        print()
        print(report)
        return summary

    def _histogram_from_tree(self) -> dict:
        """{scene_id: Counter(class)} read back from the tree's label files."""
        index = json.loads((self.tree / "frame_index.json").read_text())
        per_scene = defaultdict(Counter)
        for stem, entry in index.items():
            per_scene[entry["scene_id"]].update(
                label["type"] for label in read_labels(self.tree / "label_2" / f"{stem}.txt"))
        return dict(per_scene)

    def _stamp_metadata(self, per_scene: dict):
        """
        Fill each contributing run's class_histogram in metadata.json (step 9.v).

        Counts are attributed back to runs via scene -> run. Writes the RAW file
        so its _notes survive; CONFIGS is the stripped copy and would erase them.
        Only class_histogram is touched -- everything else this pass learns goes
        into dataset_summary.json rather than growing metadata's schema.
        """
        per_run = defaultdict(Counter)
        for scene_id, counts in per_scene.items():
            per_run[self.scene_run[int(scene_id)]].update(counts)

        preset = self.configs["CARLA_config.json"]["class_map"]["active_preset"]
        meta_path = CONFIG_DIR / "metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for entry in meta["runs"]:
            if entry["run_id"] not in per_run:
                continue
            counts = per_run[entry["run_id"]]
            bucket = entry.setdefault("class_histogram", {}).setdefault(preset, {})
            for cls in [c for c in bucket if not c.startswith("_")]:
                bucket[cls] = int(counts.get(cls, 0))
            for cls, n in counts.items():          # classes the schema predates
                bucket[cls] = int(n)
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(f"[process] stamped class_histogram for runs {sorted(per_run)}")

    def _write_summary(self, per_scene: dict, histogram: Counter, stats: dict) -> dict:
        """
        <tree>/dataset_summary.json -- this split's slice of the data card.

        Records the filter thresholds BY VALUE: 'visible' is defined by code
        constants, so without them the counts here are not reproducible.
        """
        rounds = []
        summary_path = self.tree / "dataset_summary.json"
        if summary_path.is_file():
            rounds = json.loads(summary_path.read_text()).get("rounds", [])
        rounds.append({
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "runs": self.run_ids,
            "frames_added": stats["frames"],
            "topup_class": self.topup_class,
            "topup_frames": self.topup_frames,
            "git_commit": _git_commit(),
        })

        scenes = sorted(int(s) for s in per_scene)
        summary = {
            "dataset_name": self.configs["metadata.json"]["dataset_name"],
            "split": self.split,
            "frames": sum(1 for _ in (self.tree / "label_2").glob("*.txt")),
            "objects": sum(histogram.values()),
            "empty_frames": stats["empty_frames"],
            "scenes": scenes,
            "maps": sorted({self.scene_map[s] for s in scenes}),
            "time_of_day": sorted({self.scene_tod[s] for s in scenes}),
            "class_histogram": dict(sorted(histogram.items())),
            "per_scene": {str(k): dict(sorted(v.items())) for k, v in sorted(per_scene.items())},
            "dropped_last_round": stats["dropped"],
            "provenance": {
                "python_version": platform.python_version(),
                "config_hashes": {n: _sha256_file(CONFIG_DIR / n) for n in _HASHED_CONFIGS},
                "class_map_preset": self.configs["CARLA_config.json"]["class_map"]["active_preset"],
                "filter_thresholds": {k: getattr(kitti, k) for k in _THRESHOLD_KEYS},
                "include_lidar": self.include_lidar,
            },
            "rounds": rounds,
        }
        summary_path.write_text(json.dumps(summary, indent=1), encoding="utf-8")
        return summary
    def _upload_tree(self):
        """
        Publish this split to s3://<bucket>/processed/<dataset_name>/<split>/.

        The only thing this pipeline puts in S3. Raw frames never go up, so what
        is stored is the training data itself plus the summary that describes
        it -- regenerable from raw and a commit, but raw lives only as long as
        the instance does.
        """
        bucket = kitti._bucket(self.configs)
        # VERSIONED prefix. Without it every publish lands on the same keys and
        # overwrites the last one, so there is no way back to a dataset a model was
        # actually trained on -- and no way to compare two rounds of the top-up loop.
        # The version is stable across the three per-split invocations of one build
        # because it comes from config (or one --version passed to all three).
        dataset_key = (f"processed/{self.configs['metadata.json']['dataset_name']}/"
                       f"{self.version}/")
        root_key = f"{dataset_key}{self.split}/"
        s3 = boto3.client("s3", config=Config(
            retries={"total_max_attempts": 5, "mode": "adaptive"},
            max_pool_connections=_UPLOAD_WORKERS))

        # Three things land, at three different depths. The S3 layout mirrors the local
        # one exactly, so a `aws s3 sync` of the dataset prefix reproduces the directory
        # you had on the box.
        #   <dataset>/<split>/...            the KITTI tree
        #   <dataset>/samples/<split>/...    the annotated renders
        #   <dataset>/{dataset_summary,class_histogram}
        # The last two are re-uploaded on every split, which is the point: whoever pulls
        # the bucket sees a histogram matching whatever is currently in it.
        samples_dir = self.output_root / "samples" / self.split
        jobs = [(p, root_key + p.relative_to(self.tree).as_posix())
                for p in self.tree.rglob("*") if p.is_file()]
        jobs += [(p, f"{dataset_key}samples/{self.split}/{p.relative_to(samples_dir).as_posix()}")
                 for p in samples_dir.rglob("*") if p.is_file()] if samples_dir.is_dir() else []
        jobs += [(self.output_root / n, dataset_key + n)
                 for n in ("dataset_summary.json", "class_histogram.md")
                 if (self.output_root / n).is_file()]

        with ThreadPoolExecutor(max_workers=_UPLOAD_WORKERS) as pool:
            list(pool.map(lambda j: s3.upload_file(str(j[0]), bucket, j[1]), jobs))

        # Verify by listing rather than N HEADs, over ONE scope covering every key we
        # just wrote. Scoping the check to the split tree is what previously reported
        # the dataset-root files as missing when they had uploaded fine; listing the
        # whole dataset prefix costs one request per 1000 keys and cannot drift out of
        # step when another artifact location is added.
        uploaded = set()
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=dataset_key):
            uploaded.update(o["Key"] for o in page.get("Contents", []))

        ours = {key for _, key in jobs}
        missing = ours - uploaded
        if missing:
            raise RuntimeError(f"{len(missing)}/{len(jobs)} objects missing after upload "
                               f"(e.g. {next(iter(missing))})")
        print(f"[process] uploaded {len(jobs)} files to s3://{bucket}/{dataset_key} "
              f"({self.split} tree + samples + dataset summary)")

        # PRUNE. Uploading alone makes the prefix a UNION of every publish ever made
        # to it, not a copy of this one: a rewrite that renames or drops frames leaves
        # the old objects sitting there, and the prefix ends up holding two datasets
        # wearing one name. Deleting what we did not just write makes it a mirror.
        #
        # Scoped to THIS split's two prefixes -- never the other splits, never the
        # dataset-root summaries -- and skipped entirely on append, where the whole
        # point is to add to what is already there. Runs after verification, so a
        # failed upload can never delete the good copy it failed to replace.
        if self.append:
            return
        owned = (root_key, f"{dataset_key}samples/{self.split}/")
        stale = sorted(k for k in uploaded
                       if k.startswith(owned) and k not in ours)
        if not stale:
            return
        for i in range(0, len(stale), 1000):        # delete_objects caps at 1000
            s3.delete_objects(Bucket=bucket, Delete={
                "Objects": [{"Key": k} for k in stale[i:i + 1000]], "Quiet": True})
        print(f"[process] pruned {len(stale)} stale objects under {root_key} "
              f"(e.g. {stale[0]})")



def merge_summaries(output_root: Path, configs: dict) -> dict:
    """
    Refresh output_root/dataset_summary.json across whichever splits exist.

    Each invocation only knows its own split, but the data card needs the
    whole picture. Merging what is on disk means the top-level view is
    correct after every round without any invocation having to see the rest.
    """
    splits, totals = {}, Counter()
    for path in sorted(output_root.glob("*/dataset_summary.json")):
        data = json.loads(path.read_text())
        splits[data["split"]] = {k: data[k] for k in
                                 ("frames", "objects", "scenes", "maps",
                                  "time_of_day", "class_histogram")}
        totals.update(data["class_histogram"])
    merged = {
        "dataset_name": configs["metadata.json"]["dataset_name"],
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "splits": splits,
        "totals": {
            "frames": sum(s["frames"] for s in splits.values()),
            "objects": sum(s["objects"] for s in splits.values()),
            "class_histogram": dict(sorted(totals.items())),
        },
    }
    (output_root / "dataset_summary.json").write_text(
        json.dumps(merged, indent=1), encoding="utf-8")
    return merged

def _expected_classes(configs: dict) -> list:
    """
    Every class the active class_map can emit, whether or not any appeared.

    Derived from the map rather than from the counts, because the classes that
    matter most in this report are the ones with ZERO instances -- a class that
    never shows up is invisible in a histogram built from what was found.
    """
    preset = configs["CARLA_config.json"]["class_map"]
    preset = preset["presets"][preset["active_preset"]]
    return sorted(set(preset["vehicle_by_base_type"].values())
              | set(preset["sign_by_speed_kph"].values()))

def write_histogram_report(output_root: Path, configs: dict, merged: dict) -> str:
    """
    class_histogram.md -- the step 9.v table, in the shape the decision needs.

    The histogram exists to answer one question: which classes are too thin to
    train on, and therefore what the next capture round should target. So this
    scores every class against the target and names the gap, rather than just
    listing counts. Two failure modes it is built to surface:

      * a class with NO instances at all (missing from class_map's output
        entirely -- e.g. a posted speed the routes never drive past)
      * a class healthy in total but absent from a split. A class that is fine
        overall and missing from test tells you nothing on test.

    Written after every invocation, so it stays current as splits and top-up
    rounds land one at a time.
    """
    splits = list(merged["splits"])
    totals = merged["totals"]["class_histogram"]
    classes = sorted(set(_expected_classes(configs)) | set(totals))

    head = f"| {'class':<16} | " + " | ".join(f"{s:>6}" for s in splits) + \
           f" | {'total':>6} | status |"
    rule = f"|{'-' * 18}|" + "|".join("-" * 8 for _ in splits) + f"|{'-' * 8}|--------|"
    rows, thin, missing, gaps = [], [], [], {}

    for cls in classes:
        counts = [merged["splits"][s]["class_histogram"].get(cls, 0) for s in splits]
        total = totals.get(cls, 0)

        # Volume and COVERAGE are scored independently. Chaining them (as an
        # if/elif does) lets the louder problem mask the worse one: a class that
        # is both under target and absent from test reports only "UNDER", and the
        # fact that it cannot be evaluated at all goes unsaid.
        flags = []
        if total == 0:
            flags.append("MISSING")
            missing.append(cls)
        else:
            if total < TARGET_MIN:
                flags.append("UNDER")
                thin.append(cls)
            elif total > TARGET_MAX:
                flags.append("over")
            short = {s: c for s, c in zip(splits, counts) if c < TARGET_MIN_PER_SPLIT}
            if short:
                gaps[cls] = short
                flags.append(", ".join(
                    f"{'no ' + s if c == 0 else s + '=' + str(c)}" for s, c in short.items()))
        status = "; ".join(flags) if flags else "ok"
        rows.append(f"| {cls:<16} | " + " | ".join(f"{c:>6}" for c in counts) +
                    f" | {total:>6} | {status} |")

    lines = [
        f"# Class histogram - {merged['dataset_name']}",
        "",
        f"Updated {merged['updated_utc']}. Target {TARGET_MIN}-{TARGET_MAX} "
        f"instances per class. {merged['totals']['frames']} frames, "
        f"{merged['totals']['objects']} objects.",
        "", head, rule, *rows, "",
    ]

    if gaps:
        lines += [
            "## Split coverage gaps (fix these first)", "",
            f"A split with fewer than {TARGET_MIN_PER_SPLIT} instances of a class "
            "cannot measure that class. This is not a volume problem and top-up "
            "frames from the WRONG split will not fix it -- the scenes that feed "
            "that split have to produce the class in the first place.", "",
        ]
        for cls, short in gaps.items():
            where = ", ".join(f"{s}={c}" for s, c in short.items())
            lines.append(f"- **{cls}**: {where}. Add or re-route a scene in the map(s) "
                         f"assigned to that split so it drives past one, give it a "
                         f"split-pure run, and capture.")
        lines.append("")

    if missing or thin:
        lines += ["## Volume top-up", ""]
        for cls in missing:
            lines.append(f"- **{cls}**: no instances at all. No amount of top-up "
                         "helps until a scene actually drives past one -- this is a "
                         "scene_description/route change, not a processing change.")
        for cls in thin:
            lines.append(
                f"- **{cls}**: below target. Capture a split-pure run whose scenes "
                f"produce it, then:\n"
                f"  `python src/process_data/process.py --split train "
                f"--raw-root <scratch> --output-root <root> --runs <N> "
                f"--topup-class {cls} --topup-frames 200`")
        lines += ["", "Topping up drags co-occurring classes along with it "
                      "(frames chosen for a rare sign are still full of cars), so "
                      "re-read this table after each round rather than extrapolating.", ""]

    if not (gaps or missing or thin):
        lines += ["All classes are within target and covered in every split.", ""]

    report = "\n".join(lines)
    (output_root / "class_histogram.md").write_text(report, encoding="utf-8")
    return report

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Phase 1 offline processing (steps 8-10)")
    ap.add_argument("--split", default=None,
                    help="which split to process; selects the runs flagged with it")
    ap.add_argument("--report-only", action="store_true",
                    help="rebuild dataset_summary.json + class_histogram.md from the "
                         "per-split summaries already on disk and exit. Needs no raw "
                         "data and no reprocessing, so the histogram can be regenerated "
                         "after the capture instance is gone.")
    ap.add_argument("--output-root", required=True,
                    help="dataset root; the tree lands in <output-root>/<split>/")
    ap.add_argument("--raw-root", default=None,
                    help="local capture scratch to read (default flow). Omit to read S3, "
                         "which only works for runs captured with --upload-raw.")
    ap.add_argument("--runs", type=int, nargs="+", default=None,
                    help="override run selection (default: runs flagged with --split "
                         "whose status is complete)")
    ap.add_argument("--no-lidar", action="store_true",
                    help="skip velodyne/*.bin. LiDAR is written by default -- the sweep "
                         "is captured either way, so dropping it discards data that "
                         "cost GPU time.")
    ap.add_argument("--version", default=None,
                    help="dataset version for the S3 prefix (default: metadata.json "
                         "dataset_version). Pass the SAME value for every split of one "
                         "build, or the splits land under different versions.")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="cap frames written (subset smoke test)")
    ap.add_argument("--samples", type=int, default=8,
                    help="annotated sample renders: 0 to skip, -1 to render EVERY frame "
                         "(for VLM/automated QA over the whole split)")
    ap.add_argument("--no-upload", action="store_true",
                    help="skip the S3 publish and leave the tree local")
    ap.add_argument("--topup-class", default=None,
                    help="APPEND mode: keep only frames containing this class")
    ap.add_argument("--topup-frames", type=int, default=None,
                    help="randomly sample this many matching frames")
    ap.add_argument("--topup-seed", type=int, default=0,
                    help="seed for the top-up sample, so a round is reproducible")
    args = ap.parse_args()

    if args.report_only:
        root = Path(args.output_root)
        print(write_histogram_report(root, CONFIGS, merge_summaries(root, CONFIGS)))
        sys.exit(0)
    if not args.split:
        ap.error("--split is required (or pass --report-only)")

    summary = Processor(
        args.output_root, args.split, raw_root=args.raw_root, run_ids=args.runs,
        include_lidar=not args.no_lidar, limit=args.max_frames, samples=args.samples,
        version=args.version,
        upload=not args.no_upload, topup_class=args.topup_class,
        topup_frames=args.topup_frames, topup_seed=args.topup_seed,
    ).run()
    json.dump({"split": summary["split"], "frames": summary["frames"],
               "objects": summary["objects"]}, sys.stdout, indent=1)
    print()
