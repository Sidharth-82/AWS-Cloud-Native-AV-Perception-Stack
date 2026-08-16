import carla
from carla import command
from queue import Queue
from config_loader import (
    CONFIGS
)

import random

class Scene:
    ##Class Variables that don't change in between instances
    ego_config = CONFIGS["ego_config.json"]["ego_vehicle"]
    sensor_config = CONFIGS["ego_config.json"]["sensor_rig"]["sensors"]
    capture_every_n_ticks = CONFIGS["CARLA_config.json"]["capture"]["capture_every_n_ticks"]
    THRESHOLD_M = 5
    
    class SetupError(Exception):
        def __init__(self, message):
            super().__init__(message)
            self.message = message

        def __str__(self):
            return f"{self.message} (Error Reason: {self.__cause__})"

    ##Functions
    def __init__(self, carla_client: carla.Client, scene_cfg:dict):
        self.carla_client = carla_client
        self.world = carla_client.get_world()
        self.scene_id = scene_cfg["scene_id"]
        self.scene_name = scene_cfg["scene_name"]
        self.split = scene_cfg["split"]
        self.map = scene_cfg["map"]
        self.spawn_point_index = scene_cfg["route"]["spawn_point_index"]
        self.weather = CONFIGS["CARLA_config.json"]["weather_presets"][scene_cfg["weather_preset"]] 
        self.traffic = CONFIGS["CARLA_config.json"]["traffic_density_presets"][scene_cfg["traffic_preset"]]
        self.tod = scene_cfg["time_of_day"]
        self.ego_target_speed = scene_cfg["ego_target_speed_kph"]
        self.duration_s = scene_cfg["duration_s"]
        self.expected_frames = scene_cfg["expected_frames"]
        self.traffic_manager_seed = scene_cfg["seeds"]["traffic_manager"]
        self.spawn_rng_seed = scene_cfg["seeds"]["spawn_rng"]
        self.storage_address = scene_cfg["storage"]["s3_prefix"]
        
        self.tm = None
        self.tm_port = None
        self.ego_actor = None
        self.sensor_actors = {}
        self.traffic_actors = []
        self.queues = {} #{sensor_name: queue.Queue}
        self._original_settings = None
        self.sign_data = []
        
    def _cleanup(self):
        if self._original_settings is not None: 
            self.world.apply_settings(self._original_settings)
            
            
        actors = self.traffic_actors.copy()
                
        if self.ego_actor is not None:
            actors.append(self.ego_actor.id)
                
            
        batch = [command.SetAutopilot(aid, False, self.tm_port) for aid in actors]
        resp = self.carla_client.apply_batch_sync(batch, False)

        for sensor in self.sensor_actors.values():
            if sensor.is_listening():
                sensor.stop()
        
        batch = [command.DestroyActor(actor) for actor in self.sensor_actors.values()]
        resp = self.carla_client.apply_batch_sync(batch, False)
        
        
        batch = [command.DestroyActor(aid) for aid in actors]
        resp = self.carla_client.apply_batch_sync(batch, False)

    
    def __enter__(self):
        # save original world settings; set sync mode; spawn ego+traffic+sensors
        try:
            self.world_setup()
        except Exception as e:
            self._cleanup()
            raise self.SetupError("Setup failed") from e
        return self
    def __exit__(self, exc_type, exc, tb):
        # ALWAYS runs: destroy sensors, destroy actors, restore world settings
        # Revert synchronous_mode to False. Destroy sensors -> vehicles. use apply_batch([DestroyActor(x) ...])
        # 
        self._cleanup()
        return False   # False = don't suppress an exception, just clean up
    
    def world_setup(self):
        ## Set map and weather and get info about map.
        ## Only reload when the map CHANGES: CARLA leaks across load_world calls and
        ## crashes after ~5 reloads, so reloading the same map every scene (scenes 1-4
        ## are all Town04) is what tipped scene 5's Town06 load over. The previous
        ## scene's _cleanup already destroyed its actors, so reusing a same-map world
        ## is safe; sync settings + weather are re-applied below regardless.
        current_map = self.world.get_map().name.split("/")[-1]   # e.g. "Town04"
        if current_map != self.map:
            print(f"[scene {self.scene_id}] loading map {self.map} (was {current_map}) ...", flush=True)
            self.world = self.carla_client.load_world(self.map)
        else:
            print(f"[scene {self.scene_id}] reusing loaded map {self.map} (no reload)", flush=True)
        bp_lib = self.world.get_blueprint_library()
        spawns = self.world.get_map().get_spawn_points() # list[carla.Transform]
        self.world.set_weather(carla.WeatherParameters(**self.weather))
        print(f"[scene {self.scene_id}] map ready; {len(spawns)} spawn points", flush=True)
        
        ##Settings + sync mode
        
        self._original_settings = self.world.get_settings()
        s = self.world.get_settings()
        s.synchronous_mode = True
        s.fixed_delta_seconds = 0.05
        self.world.apply_settings(s)
        
        ##Traffic Manager
        self.tm = self.carla_client.get_trafficmanager(8000)
        self.tm.set_synchronous_mode(True)
        self.tm.set_random_device_seed(self.traffic_manager_seed)
        self.tm_port = self.tm.get_port()
        
        ## Spawn Ego + Sensors
        ego_bp = bp_lib.find(self.ego_config["blueprint"])
        ego_bp.set_attribute("role_name", "hero")
        self.ego_actor = self.world.spawn_actor(ego_bp, spawns[self.spawn_point_index])
        print(f"[scene {self.scene_id}] ego spawned at spawn index {self.spawn_point_index}", flush=True)
        
        for name, sensor in self.sensor_config.items():
            if "camera" in sensor["type"]:
                sensor_bp = bp_lib.find(sensor["type"])
                sensor_bp.set_attribute("image_size_x", str(sensor["width"]))
                sensor_bp.set_attribute("image_size_y", str(sensor["height"]))
                sensor_bp.set_attribute("fov", str(sensor["fov_deg"]))
                coord = None
                if sensor.get("co_located_with"):
                    coord = self.sensor_config[sensor["co_located_with"]]["T_sensor_to_ego"]
                else:
                    coord = sensor["T_sensor_to_ego"]
            elif "lidar" in sensor["type"]:
                sensor_bp = bp_lib.find(sensor["type"])
                sensor_bp.set_attribute("channels", str(sensor["channels"]))
                sensor_bp.set_attribute("range", str(sensor["range_m"]))
                sensor_bp.set_attribute("points_per_second", str(sensor["points_per_second"]))
                sensor_bp.set_attribute("rotation_frequency", str(sensor["rotation_frequency"]))
                sensor_bp.set_attribute("upper_fov", str(sensor["upper_fov_deg"]))
                sensor_bp.set_attribute("lower_fov", str(sensor["lower_fov_deg"]))
                coord = None
                if sensor.get("co_located_with"):
                    coord = self.sensor_config[sensor["co_located_with"]]["T_sensor_to_ego"]
                else:
                    coord = sensor["T_sensor_to_ego"]
            elif "other" in sensor["type"]:
                sensor_bp = bp_lib.find(sensor["type"])
                coord = {
                    "position": { "x": 0.0, "y": 0.0, "z": 0.0 },
                    "rotation": { "roll": 0.0, "pitch": 0.0, "yaw": 0.0 }
                    }
                    
            x, y, z = coord["position"]["x"], coord["position"]["y"], coord["position"]["z"]
            roll, pitch, yaw = coord["rotation"]["roll"], coord["rotation"]["pitch"], coord["rotation"]["yaw"]
            
            tf = carla.Transform(carla.Location(x,y,z), carla.Rotation(pitch, yaw, roll))
            actor = self.world.spawn_actor(sensor_bp, tf, attach_to=self.ego_actor)
            self.sensor_actors[name] = actor
            
            q = Queue()
            self.queues[name] = q
            actor.listen(lambda data, q=q: q.put((data.frame, data)))
        print(f"[scene {self.scene_id}] {len(self.sensor_actors)} sensors attached", flush=True)
        
        ## Traffic Batch Spawn
        random.seed(self.spawn_rng_seed)
        ## Only spawn vehicles whose base_type the dataset can label. The exclude list let
        ## through blueprints with an empty/other base_type (buses, cars CARLA never tagged)
        ## -> unmappable "" classes. Whitelisting the class_map's vehicle base_types drops
        ## bicycles AND those, so every spawned vehicle maps cleanly and
        ## actor.attributes["base_type"] is always valid downstream (get_actors unchanged).
        class_map = CONFIGS["CARLA_config.json"]["class_map"]
        allowed_base_types = set(class_map["presets"][class_map["active_preset"]]["vehicle_by_base_type"])
        vehicle_bps = [bp for bp in bp_lib.filter("vehicle.*")
                    if bp.get_attribute("base_type").as_str() in allowed_base_types]
        
        points = [p for i, p in enumerate(spawns) if i != self.spawn_point_index]
        random.shuffle(points)
        n = min(self.traffic["num_vehicles"], len(points))
        
        batch = [command.SpawnActor(random.choice(vehicle_bps), points[i]) for i in range(n)]
        resp = self.carla_client.apply_batch_sync(batch, False)
        self.traffic_actors = [r.actor_id for r in resp if not r.error]
        print(f"[scene {self.scene_id}] traffic spawned: {len(self.traffic_actors)}/{n}", flush=True)

        ##Lights
        self._set_night_lights()

        ##Signs
        signs = self.world.get_environment_objects(carla.CityObjectLabel.TrafficSigns)
        
        sign_landmark_obj = list({lm.id: lm for lm in self.world.get_map().get_all_landmarks_of_type("274")}.values())
        
        if sign_landmark_obj:
            for sign in signs:
                sign_pos = sign.bounding_box.location
                nearest = min(sign_landmark_obj, key=lambda lm: sign_pos.distance(lm.transform.location))
                nearest_d = sign_pos.distance(nearest.transform.location)
                if nearest_d <= self.THRESHOLD_M:
                    self.sign_data.append((sign.id, sign.bounding_box, nearest.value))
        print(f"[scene {self.scene_id}] world_setup complete; {len(self.sign_data)} signs", flush=True)

        

    def _set_night_lights(self):
        """
        Switch headlights and tail lights on for every vehicle in a night scene.

        CARLA does not do this on its own -- autopilot drives with the lights off --
        and the clear_night preset puts the sun at -90 deg, so there is no light
        source in the scene at all. The captured night frames came out effectively
        black (mean luminance ~4/255, with ~75% of pixels under 10). The LABELS were
        still correct, since they come from the sim rather than the image, but the
        images carried almost nothing for a camera detector to learn from.

        Ego headlights matter most: they are what lights the road and the vehicles
        ahead for the front camera. Position lights on the traffic matter too -- they
        are how a car ahead is visible at all before the headlights reach it.

        Set explicitly rather than through tm.update_vehicle_lights(): the Traffic
        Manager drives lighting from its own internal state, which would make a
        scene's exposure depend on TM history instead of on the scene config, and
        reproducibility from config is the whole point of the capture design.
        """
        if self.tod != "night":
            return

        lights = carla.VehicleLightState(
            carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam)
        self.ego_actor.set_light_state(lights)
        batch = [command.SetVehicleLightState(aid, lights) for aid in self.traffic_actors]
        self.carla_client.apply_batch_sync(batch, False)
        print(f"[scene {self.scene_id}] night scene: lights on for "
              f"{len(self.traffic_actors) + 1} vehicles", flush=True)

    def start(self):
        print(f"[scene {self.scene_id}] start(): enabling autopilot", flush=True)
        self.tm.global_percentage_speed_difference(self.traffic["tm_global_speed_difference_pct"])
        batch = [command.SetAutopilot(aid, True, self.tm_port) for aid in self.traffic_actors + [self.ego_actor.id]]
        resp = self.carla_client.apply_batch_sync(batch, False)

    def tick(self):
        carla_frame = 0
        
        for t in range(self.capture_every_n_ticks): 
            carla_frame = self.world.tick()
            i = 0
            if t < self.capture_every_n_ticks-1:
                while i < len(self.sensor_config.keys()):
                    name = list(self.sensor_config.keys())[i]
                    q = self.queues[name]
                    f, data = q.get(block=True, timeout=2.0)
                    if f != carla_frame:
                        continue
                    i+=1

        snapshot = self.snapshot(carla_frame)
        
        timestamp = self.world.get_snapshot().timestamp.elapsed_seconds
        
        return carla_frame, timestamp, snapshot
        
    def get_actors(self):
        return [(actor.id, actor.bounding_box, actor.get_transform(), actor.get_velocity(), actor.get_angular_velocity(), actor.type_id, actor.attributes.get("base_type")) for actor in self.world.get_actors(self.traffic_actors)]

    
    def get_signs(self):
        return self.sign_data
        
    def get_ego(self):
        return (self.ego_actor.get_transform(), self.ego_actor.get_velocity(), self.ego_actor.get_angular_velocity())
    
    def snapshot(self, carla_frame: int):
        sensor_val = {}
        i = 0
        while i < len(self.sensor_config.keys()):
            name = list(self.sensor_config.keys())[i]
            q = self.queues[name]
            f, data = q.get(block=True, timeout=2.0)
            if f != carla_frame:
                continue
            
            sensor_val[name] = data
            i += 1
            
        actors = self.get_actors()
        signs = self.get_signs()
        ego_pose = self.get_ego()
        
        return sensor_val, ego_pose, actors, signs
    
