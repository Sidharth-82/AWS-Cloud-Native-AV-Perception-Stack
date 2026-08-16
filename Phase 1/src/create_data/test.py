import carla
from utils import CONFIGS
from scene import Scene


if __name__ == "__main__":
    client = carla.Client("localhost", 2000); client.set_timeout(20.0)
    scene_cfg = CONFIGS["scene_description.json"]["scenes"][0]   # scene 1
    with Scene(client, scene_cfg) as s:
        s.start()
        frame_id, ts, (sensor_val, ego, actors, signs) = s.tick()
        print(frame_id, ts, list(sensor_val), "actors:", len(actors), "signs:", len(signs))