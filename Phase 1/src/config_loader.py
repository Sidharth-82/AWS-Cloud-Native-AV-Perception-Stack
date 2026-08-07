from pathlib import Path
import json
from utils import strip_all_documentation

# Anchored to this file, not the working directory, so the loader works no
# matter where python is invoked from.
CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

# The config SET: the constant inputs that parametrize a run. dataset_example
# is a schema DOC, not runtime config, so it is deliberately absent.
CONFIG_FILES = (
    "CARLA_config.json",
    "ego_config.json",
    "scene_description.json",
    "metadata.json",
)

def load_config(name: str) -> dict:
    """
    Load ONE config from Phase 1/config/ with documentation keys stripped.

    Stripping happens in memory only. The file on disk keeps its '_' notes --
    they are the schema contract (conventions, decode formulas, the unverified
    instance-seg warning), not cruft to be purged.

    Args:
        name (str): File name within config folder
    """
    path = CONFIG_DIR / name
    with path.open(encoding="utf-8") as f:
        return strip_all_documentation(json.load(f))


def load_configs() -> dict:
    """
    Load the whole config SET from Phase 1/config/, stripped, keyed by filename.

    The set fed to CARLA is the same set offline processing reads, so we load
    the local source-of-truth copies instead of pulling them from S3. Small and
    read-only -- fine to hold in memory for the whole session.

    Returns:
        dict: {filename: parsed_stripped_config} for each file in CONFIG_FILES.
    """
    return {name: load_config(name) for name in CONFIG_FILES}


# Loaded once at import. The process shares this single read-only bundle; every
# function that needs config defaults to it.
CONFIGS = load_configs()