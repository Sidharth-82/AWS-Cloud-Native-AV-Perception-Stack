from pathlib import Path
import json
import carla
import numpy as np


def _v3(v) -> dict:
    """carla.Vector3D / carla.Location -> {x, y, z}."""
    return {"x": v.x, "y": v.y, "z": v.z}


def _rot(r) -> dict:
    """carla.Rotation -> {roll, pitch, yaw}."""
    return {"roll": r.roll, "pitch": r.pitch, "yaw": r.yaw}


class Serializer:
    def __init__(self, output_root, scene_id, naming_convention, sensor_encoding):
        self.output_root = Path(output_root)
        self.scene_id = scene_id
        self.naming_convention = naming_convention      # write_frame builds frame_id/paths from this
        self.sensor_encoding = sensor_encoding

        # "scene_001" and the scene's output dir — records.jsonl + all sensor files live here.
        self.scene_dir = naming_convention["scene_dir"].format(scene_id=scene_id)
        self.scene_path = self.output_root / self.scene_dir

        self._records_f = None   # opened in __enter__, closed in __exit__
    
    def __enter__(self):
        # Create the scene folder and open records.jsonl once for the whole scene.
        # mode "w" so a re-run overwrites the scene cleanly (idempotent).
        self.scene_path.mkdir(parents=True, exist_ok=True)
        self._records_f = open(self.scene_path / "records.jsonl", "w", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, tb):
        # Always flush + close the records file, even if a frame write raised.
        if self._records_f is not None:
            self._records_f.close()
        return False   # don't suppress exceptions
    
    def write_frame(self, carla_frame, timestamp, snapshot):
        frame_str = self.naming_convention["frame_id"].format(scene_id=self.scene_id, carla_frame=carla_frame)  # "001_012345"
        sensor_val, ego, actors, signs = snapshot
        sensor_files = {}
        for name, data in sensor_val.items():
            if "cam" in name:  sensor_files[name] = self.save_image_to_disk(data, name, frame_str)
            elif "lidar" in name:   sensor_files[name] = self.save_lidar_to_disk(data, name, frame_str)
            # skip imu/gnss — they're ego-state, not in the record's sensor_files
        self.save_frame_to_records(carla_frame, frame_str, timestamp, sensor_files, ego, actors, signs)

    
    def save_image_to_disk(self, image:carla.Image, name:str, frame_str:str) -> str:
        rel = self.naming_convention["sensor_file"].format(scene_dir = self.scene_dir, sensor_name = name, frame_id = frame_str, ext="png")
        path = self.output_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save_to_disk(str(path), carla.ColorConverter.Raw)
        return rel
    
    def save_lidar_to_disk(self, lidar, name, frame_str) -> str:
        rel = self.naming_convention["sensor_file"].format(scene_dir = self.scene_dir, sensor_name = name, frame_id = frame_str, ext="npy" )
        path = self.output_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        
        pts = np.frombuffer(lidar.raw_data, dtype=np.float32).reshape(-1, 4)
        np.save(path, pts)
        
        return rel
    
    def _actor_record(self, actor) -> dict:
        aid, bbox, tf, vel, ang, type_id, base_type = actor

        # Vehicle bbox is in the actor's LOCAL frame; compose its center into world.
        center = carla.Location(bbox.location.x, bbox.location.y, bbox.location.z)
        tf.transform(center)   # local -> world, in place

        return {
            "id": aid,
            "global_actor_id": self.naming_convention["global_actor_id"].format(
                scene_id=self.scene_id, actor_id=aid),
            "class": base_type,                 # RAW base_type; KITTI writer applies class_map offline
            "source": "carla_actor",
            "blueprint_id": type_id,
            "base_type": base_type,
            "velocity": {"linear": _v3(vel), "angular": _v3(ang)},
            "bounding_box_world": {
                "center": _v3(center),
                "half_extent": _v3(bbox.extent),   # CARLA extent is already half-dims
                "rotation": _rot(tf.rotation),
            },
        }

    def _sign_record(self, sign) -> dict:
        # sign tuple = (id, bounding_box, listed_speed_kph); env-object bbox is ALREADY world-frame.
        sid, bbox, speed = sign
        return {
            "id": sid,
            "global_actor_id": self.naming_convention["global_actor_id"].format(
                scene_id=self.scene_id, actor_id=sid),
            "class": "speed_sign",
            "source": "environment_object + map_landmark",
            "blueprint_id": None,
            "base_type": None,
            "listed_speed_kph": speed,
            "velocity": None,                    # signs are static
            "bounding_box_world": {
                "center": _v3(bbox.location),
                "half_extent": _v3(bbox.extent),
                "rotation": _rot(bbox.rotation),
            },
        }

    def save_frame_to_records(self, carla_frame, frame_str, timestamp, sensor_files, ego, actors, signs):
        ego_tf, ego_vel, ego_ang = ego
        record = {
            "scene_id": self.scene_id,
            "carla_frame": carla_frame,
            "frame_id": frame_str,
            "timestamp_sim_s": timestamp,
            "sensor_files": sensor_files,
            "ego_pose": {
                "position": _v3(ego_tf.location),
                "rotation": _rot(ego_tf.rotation),
                "velocity": {"linear": _v3(ego_vel), "angular": _v3(ego_ang)},
            },
            # Vehicles + signs share one array — signs are entries with class="speed_sign".
            "actors": [self._actor_record(a) for a in actors]
                    + [self._sign_record(s) for s in signs],
        }
        self._records_f.write(json.dumps(record) + "\n")