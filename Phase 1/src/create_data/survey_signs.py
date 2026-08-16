
#################################
"""
Rank a map's spawn points by how many speed-limit signs the route ahead passes.

Signs are the starved class, and which spawn point you pick decides how many a
scene ever drives past -- but nothing in the config says where the signs are, so
picking a spawn_point_index by hand is guessing with GPU time. This answers it
from the map itself, before any capture runs.

Method: for each spawn point, walk the road graph forward with waypoint.next()
for ROUTE_M metres and count the distinct speed-limit signs that come within
RADIUS_M of the path. That is an approximation of the drive -- the Traffic
Manager picks its own turns at junctions, so a real route diverges -- but it
ranks spawn points correctly, which is all it is for.

Run it per map (one load per process, like capture, to stay clear of CARLA's
cross-reload crash):

    python src/create_data/survey_signs.py --map Town04
    python src/create_data/survey_signs.py --map Town05 --top 15
"""


### Imports below

import argparse
import json
from collections import Counter

import carla

from config_loader import CONFIGS


SERVER = CONFIGS["CARLA_config.json"]["server"]
SIGN_TYPE = CONFIGS["CARLA_config.json"]["landmarks"]["speed_limit_signal_type"]
THRESHOLD_M = 5.0     # sign mesh <-> landmark association, same as scene.py

ROUTE_M = 4000.0      # how far ahead to walk. Half a 300 s / 100 kph drive; the
                      # ranking is what matters, not the absolute count.
STEP_M = 10.0         # waypoint stride along the route
RADIUS_M = 40.0       # a sign this close to the path will pass through the frame


def sign_points(world):
    """[(location, posted_kph)] for every speed-limit sign mesh in the map."""
    meshes = world.get_environment_objects(carla.CityObjectLabel.TrafficSigns)
    landmarks = list({lm.id: lm for lm in
                      world.get_map().get_all_landmarks_of_type(SIGN_TYPE)}.values())
    if not landmarks:
        return []

    out = []
    for mesh in meshes:
        loc = mesh.bounding_box.location
        nearest = min(landmarks, key=lambda lm: loc.distance(lm.transform.location))
        if loc.distance(nearest.transform.location) <= THRESHOLD_M:
            out.append((loc, int(round(nearest.value))))
    return out


def signs_ahead(carla_map, spawn, signs):
    """Distinct signs within RADIUS_M of the route walked forward from `spawn`."""
    wp = carla_map.get_waypoint(spawn.location)
    if wp is None:
        return Counter()

    seen, walked = {}, 0.0
    while walked < ROUTE_M:
        nxt = wp.next(STEP_M)
        if not nxt:
            break                      # dead end: route is shorter than ROUTE_M
        wp = nxt[0]
        walked += STEP_M
        here = wp.transform.location
        for idx, (loc, value) in enumerate(signs):
            if idx not in seen and loc.distance(here) <= RADIUS_M:
                seen[idx] = value
    return Counter(seen.values())


def main():
    ap = argparse.ArgumentParser(description="rank spawn points by sign exposure")
    ap.add_argument("--map", required=True, help="e.g. Town04")
    ap.add_argument("--top", type=int, default=10, help="how many spawn points to print")
    ap.add_argument("--out", default=None, help="write the full ranking to this JSON file")
    args = ap.parse_args()

    client = carla.Client(SERVER["host"], SERVER["port"])
    client.set_timeout(SERVER["timeout_s"])
    world = client.load_world(args.map)
    carla_map = world.get_map()

    signs = sign_points(world)
    spawns = carla_map.get_spawn_points()
    totals = Counter(v for _, v in signs)
    print(f"{args.map}: {len(spawns)} spawn points, {len(signs)} speed-limit signs "
          f"{dict(sorted(totals.items()))}")
    if not signs:
        print("  no speed-limit landmarks in this map -- no route here will ever "
              "produce a sign label.")
        return

    ranked = []
    for i, spawn in enumerate(spawns):
        counts = signs_ahead(carla_map, spawn, signs)
        if counts:
            ranked.append({"spawn_point_index": i,
                           "signs_ahead": sum(counts.values()),
                           "by_value": dict(sorted(counts.items()))})
    ranked.sort(key=lambda r: -r["signs_ahead"])

    print(f"\n  {'spawn':>6}  {'signs':>5}  by posted value")
    for row in ranked[:args.top]:
        print(f"  {row['spawn_point_index']:>6}  {row['signs_ahead']:>5}  {row['by_value']}")
    if not ranked:
        print("  no spawn point's route came within "
              f"{RADIUS_M:.0f} m of a sign in {ROUTE_M:.0f} m")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"map": args.map, "sign_totals": dict(sorted(totals.items())),
                       "route_m": ROUTE_M, "radius_m": RADIUS_M,
                       "ranking": ranked}, f, indent=1)
        print(f"\nfull ranking -> {args.out}")


if __name__ == "__main__":
    main()
