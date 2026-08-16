
#################################
"""
This file is meant to house code dedicated towards visualising labelled frames.

Functions within:

def render_kitti_frame(kitti_root, stem, out_path, label_dir=..., ...) -> Path
    - Draw one frame's boxes onto its image: the projected 3D cuboid, the 2D box,
      and the class name. Reads image_2/, label_2/ and calib/ straight off disk.

def render_samples(kitti_root, out_dir, n=..., ...) -> list[Path]
    - Render a spread of frames (busiest first) for eyeballing a fresh dataset
      and for the README's annotated samples.

Why render from the KITTI FILES rather than from the capture records: it makes
this the round-trip check on the writer. Rebuilding a cuboid out of
(h, w, l, x, y, z, rotation_y) and P2 exercises the dimension ORDER, the
bottom-centre location convention and the yaw convention -- three things that
produce a perfectly well-formed label file when they are wrong. It also means
the same function draws model PREDICTIONS in later phases, since those come out
in KITTI label format too: point label_dir at them and pass a score threshold.
"""


### Imports below

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


### Drawing below

# Deterministic per-class colours, so the same class reads the same across frames.
_COLORS = {
    "car": (0, 220, 60),
    "truck": (255, 170, 0),
    "van": (0, 190, 255),
    "motorcycle": (255, 0, 200),
    "vehicle": (0, 220, 60),          # the class_map 'coarse' preset
}
_SIGN_COLOR = (255, 70, 70)
_BOX2D_COLOR = (255, 255, 255)

# Cuboid edges over the KITTI corner order built in _corners_from_label:
# 0-3 are the bottom face, 4-7 the top, and i -> i+4 the verticals.
_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0),
          (4, 5), (5, 6), (6, 7), (7, 4),
          (0, 4), (1, 5), (2, 6), (3, 7)]
# The +l/2 end of the box, i.e. the face the object points at. Drawn with a cross
# because a wireframe box is SYMMETRIC: a 180 degree yaw error looks perfect
# without it, and that is the most common way a rotation convention goes wrong.
_FRONT_FACE = (0, 1, 5, 4)


def _color_for(cls: str):
    return _COLORS.get(cls, _SIGN_COLOR if cls.startswith("speed_sign") else (200, 200, 200))


def read_calib(path) -> np.ndarray:
    """P2 (3x4) from a KITTI calib file."""
    for line in Path(path).read_text().splitlines():
        if line.startswith("P2:"):
            return np.array([float(v) for v in line.split()[1:]]).reshape(3, 4)
    raise KeyError(f"no P2 row in {path}")


def read_labels(path) -> list:
    """
    Parse a KITTI label (or prediction) file.

    15 columns is ground truth; a 16th is the confidence score that detectors
    append, so prediction files parse through the same reader.
    """
    out = []
    for line in Path(path).read_text().splitlines():
        f = line.split()
        if not f:
            continue
        out.append({
            "type": f[0],
            "truncated": float(f[1]),
            "occluded": int(float(f[2])),
            "alpha": float(f[3]),
            "box2d": [float(v) for v in f[4:8]],
            "dimensions": [float(v) for v in f[8:11]],    # h, w, l
            "location": [float(v) for v in f[11:14]],     # x, y, z (bottom centre)
            "rotation_y": float(f[14]),
            "score": float(f[15]) if len(f) > 15 else None,
        })
    return out


def _corners_from_label(label: dict) -> np.ndarray:
    """
    The 8 camera-frame corners of a KITTI 3D box. (8, 3)

    Built exactly as the KITTI devkit does: length along local x, width along
    local z, height DOWN from the location (y is +down and location is the
    bottom face, so the top of the box is at -h).
    """
    h, w, l = label["dimensions"]
    x = np.array([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2])
    y = np.array([0, 0, 0, 0, -h, -h, -h, -h], dtype=float)
    z = np.array([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])

    ry = label["rotation_y"]
    rot = np.array([[np.cos(ry), 0.0, np.sin(ry)],
                    [0.0, 1.0, 0.0],
                    [-np.sin(ry), 0.0, np.cos(ry)]])
    return (rot @ np.stack([x, y, z])).T + np.array(label["location"])


def _project(corners: np.ndarray, P2: np.ndarray):
    """Camera-frame corners -> pixels, plus each corner's depth for behind-camera culling."""
    hom = np.hstack([corners, np.ones((len(corners), 1))])
    uvw = hom @ P2.T
    depth = uvw[:, 2]
    safe = np.where(np.abs(depth) < 1e-6, 1e-6, depth)
    return np.stack([uvw[:, 0] / safe, uvw[:, 1] / safe], axis=-1), depth


def draw_boxes(image: Image.Image, labels: list, P2: np.ndarray,
               draw_2d: bool = True, draw_3d: bool = True,
               score_threshold: float = None) -> Image.Image:
    """
    Draw KITTI labels onto an image. Returns the same image, drawn on in place.

    Kept source-agnostic on purpose: `labels` is whatever read_labels produced,
    so ground truth and detector output render through one path.
    """
    draw = ImageDraw.Draw(image)

    for lab in labels:
        if lab["type"] == "DontCare":
            continue
        if score_threshold is not None and (lab["score"] or 0.0) < score_threshold:
            continue
        color = _color_for(lab["type"])

        if draw_3d:
            uv, depth = _project(_corners_from_label(lab), P2)
            in_front = depth > 0.1
            for a, b in _EDGES:
                if in_front[a] and in_front[b]:
                    draw.line([tuple(uv[a]), tuple(uv[b])], fill=color, width=2)
            # Cross on the facing end -- this is what makes a flipped yaw visible.
            if all(in_front[i] for i in _FRONT_FACE):
                a, b, c, d = (tuple(uv[i]) for i in _FRONT_FACE)
                draw.line([a, c], fill=color, width=1)
                draw.line([b, d], fill=color, width=1)

        x1, y1, x2, y2 = lab["box2d"]
        if draw_2d:
            draw.rectangle([x1, y1, x2, y2], outline=_BOX2D_COLOR, width=1)

        tag = lab["type"] if lab["score"] is None else f"{lab['type']} {lab['score']:.2f}"
        draw.text((x1 + 2, max(0.0, y1 - 11)), tag, fill=color)

    return image


### Frame + sample rendering below

def render_kitti_frame(kitti_root, stem: str, out_path, label_dir: str = "label_2",
                       draw_2d: bool = True, draw_3d: bool = True,
                       score_threshold: float = None):
    """
    Render one frame of a KITTI tree to out_path.

    Args:
        kitti_root: a tree written by parser.write_kitti (image_2/, label_2/, calib/).
        stem: frame stem, e.g. '000007'.
        label_dir: which label directory to read. Point it at a predictions
            directory to render detector output over the same images.
        score_threshold: drop predictions below this confidence. None keeps all,
            which is what ground truth wants.
    """
    root = Path(kitti_root)
    image = Image.open(root / "image_2" / f"{stem}.png").convert("RGB")
    labels = read_labels(root / label_dir / f"{stem}.txt")
    P2 = read_calib(root / "calib" / f"{stem}.txt")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    draw_boxes(image, labels, P2, draw_2d, draw_3d, score_threshold).save(out_path)
    return out_path


def render_samples(kitti_root, out_dir, n: int = 8, label_dir: str = "label_2", **kwargs):
    """
    Render a spread of frames from a tree: the busiest ones first.

    This is the step 8.v sanity pass -- run it on a fresh dataset BEFORE trusting
    a mass write, since a coordinate bug produces a clean-looking label file and
    only shows up drawn on pixels. Doubles as the source of the README's
    annotated sample images.

    Returns the list of paths written.
    """
    root, out_dir = Path(kitti_root), Path(out_dir)
    stems = sorted(p.stem for p in (root / label_dir).glob("*.txt"))
    stems.sort(key=lambda s: -len(read_labels(root / label_dir / f"{s}.txt")))
    return [render_kitti_frame(root, s, out_dir / f"{s}.png", label_dir, **kwargs)
            for s in stems[:n]]
