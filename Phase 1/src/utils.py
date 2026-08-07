import json


# Per-scene records file living at the scene's S3 prefix, one frame-record per
# line (schema = dataset_example.json).
# TODO(verify): confirm the capture-side writer emits this exact name/format.
RECORDS_FILENAME = "records.jsonl"


### Functions Below

def strip_all_documentation(node) -> dict:
    """
    Recursively drop documentation keys (leading '_') from parsed JSON.

    Returns a new structure; the input is left untouched. Descends into lists
    as well as dicts: the configs nest dicts inside arrays (scenes, actors,
    runs), and stopping at the array boundary silently leaves notes behind.
    """
    if isinstance(node, dict):
        return {
            key: strip_all_documentation(val)
            for key, val in node.items()
            if not key.startswith("_")
        }
    if isinstance(node, list):
        return [strip_all_documentation(val) for val in node]
    return node
