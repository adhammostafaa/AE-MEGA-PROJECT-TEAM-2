import time
import math
import collections
import collections.abc
collections.MutableMapping = collections.abc.MutableMapping
from dronekit import connect, VehicleMode, LocationGlobalRelative, Command
from pymavlink import mavutil

CONNECTION_STRING = "udp:127.0.0.1:14551"

TARGET_ALT = 30 

WAYPOINTS = [
    (38.3149255, -76.5420377),
    (38.3141342, -76.5422738),
    (38.3140585, -76.5446556),
    (38.3157505, -76.5540433),
    (38.3178718, -76.5504277),
    (38.3179223, -76.5463293),
    (38.3156074, -76.5448547),
    (38.3168027, -76.5506959),
    (38.3161209, -76.5518117),
    (38.3152875, -76.5471125),
]

OBSTACLES = [
    (38.31409620, -76.54346470, 4.0, 100.0),
    (38.31676480, -76.54519200, 4.0, 100.0),
    (38.31646180, -76.55125310, 4.0, 100.0),
]

SAFETY_BUFFER = 15.0  
CHECK_INTERVAL = 1.0  

def ground_distance_m(lat1, lon1, lat2, lon2):
    R = 6371000.0  
    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    x = dlon * math.cos((lat1_r + lat2_r) / 2)
    y = dlat
    return math.sqrt(x * x + y * y) * R


def bearing_deg(lat1, lon1, lat2, lon2):
    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(lat2_r)
    y = (math.cos(lat1_r) * math.sin(lat2_r) -
         math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon))
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def offset_position(lat, lon, distance_m, bearing_degrees):
    R = 6371000.0
    br = math.radians(bearing_degrees)
    lat_r = math.radians(lat)
    lat2 = math.asin(
        math.sin(lat_r) * math.cos(distance_m / R) +
        math.cos(lat_r) * math.sin(distance_m / R) * math.cos(br)
    )
    lon2 = math.radians(lon) + math.atan2(
        math.sin(br) * math.sin(distance_m / R) * math.cos(lat_r),
        math.cos(distance_m / R) - math.sin(lat_r) * math.sin(lat2)
    )
    return math.degrees(lat2), math.degrees(lon2)


def find_blocking_obstacle(lat, lon, alt):
    for (o_lat, o_lon, o_radius, o_height) in OBSTACLES:
        horizontal_dist = ground_distance_m(lat, lon, o_lat, o_lon)
        too_close_horizontally = horizontal_dist < (o_radius + SAFETY_BUFFER)
        too_low = alt < o_height
        if too_close_horizontally and too_low:
            return (o_lat, o_lon, o_radius, o_height)
    return None

def arm_and_takeoff(vehicle, target_altitude):
    print("Waiting for vehicle to be armed")
    while not vehicle.is_armable:
        time.sleep(1)

    print("Arming motors...")
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True
    while not vehicle.armed:
        time.sleep(1)

    print("Taking off...")
    vehicle.simple_takeoff(target_altitude)
    while True:
        alt = vehicle.location.global_relative_frame.alt
        print(f"  altitude: {alt:.1f} m")
        if alt >= target_altitude * 0.95:
            print("Reached cruise altitude.")
            break
        time.sleep(1)


def upload_mission(vehicle):
    cmds = vehicle.commands
    cmds.download()
    cmds.wait_ready()
    cmds.clear()
    for lat, lon in WAYPOINTS:
        cmds.add(Command(
            0, 0, 0,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
            mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
            0, 0, 0, 0, 0, 0,
            lat, lon, TARGET_ALT
        ))
    cmds.upload()
    print(f"Uploaded {len(WAYPOINTS)} waypoints.")


def fly_mission_with_avoidance(vehicle):
    vehicle.mode = VehicleMode("AUTO")
    time.sleep(1)
    print(f"Mode is now: {vehicle.mode.name}")
    vehicle.commands.next = 0
    total_wp = len(WAYPOINTS)

    while True:
        current_wp_number = vehicle.commands.next   
        next_wp_index = current_wp_number - 1 
        if next_wp_index >= total_wp:
            print("Mission complete.")
            break

        cur = vehicle.location.global_relative_frame
        print(f"  WP {next_wp_index+1}/{total_wp} | mode={vehicle.mode.name} | lat={cur.lat:.6f} lon={cur.lon:.6f} alt={cur.alt:.1f}")
        target_lat, target_lon = WAYPOINTS[min(next_wp_index, total_wp - 1)]

        obstacle = find_blocking_obstacle(cur.lat, cur.lon, cur.alt)
        if obstacle:
            o_lat, o_lon, o_radius, o_height = obstacle
            print(f"Obstacle detected near waypoint {next_wp_index + 1}: "
                  f"center=({o_lat:.6f},{o_lon:.6f}) radius={o_radius}m")

            path_bearing = bearing_deg(cur.lat, cur.lon, target_lat, target_lon)
            sidestep_bearing = (path_bearing + 90) % 360
            divert_distance = o_radius + SAFETY_BUFFER + 10

            divert_lat, divert_lon = offset_position(
                cur.lat, cur.lon, divert_distance, sidestep_bearing
            )

            print("  Taking manual control (GUIDED) to divert...")
            vehicle.mode = VehicleMode("GUIDED")
            while vehicle.mode.name != "GUIDED":
                time.sleep(0.5)
            vehicle.simple_goto(LocationGlobalRelative(divert_lat, divert_lon, TARGET_ALT))

            while True:
                cur = vehicle.location.global_relative_frame
                if find_blocking_obstacle(cur.lat, cur.lon, cur.alt) is None:
                    break
                time.sleep(CHECK_INTERVAL)

            print("  Clear of obstacle. Resuming mission (AUTO).")
            vehicle.mode = VehicleMode("AUTO")

        time.sleep(CHECK_INTERVAL)
def disable_gcs_failsafe(vehicle, max_attempts=20, wait=2):
    print("Disabling GCS failsafe")
    for attempt in range(max_attempts):
        try:
            vehicle.parameters['FS_GCS_ENABLE'] = 0
            time.sleep(wait)
            current = vehicle.parameters.get('FS_GCS_ENABLE', None)
            if current == 0:
                print("  FS_GCS_ENABLE confirmed set to 0.")
                return
        except Exception as e:
            print(f"  attempt {attempt+1} failed ({e}), retrying")
        time.sleep(1)
    print("WARNING: could not confirm FS_GCS_ENABLE was set. Continuing anyway")

if __name__ == "__main__":
    print(f"Connecting to vehicle on {CONNECTION_STRING}...")
    vehicle = connect(CONNECTION_STRING, wait_ready=True, heartbeat_timeout=90)
    print("Waiting for vehicle initialization...")
    vehicle.wait_ready(timeout=90)
    print("Vehicle is fully ready.")
    try:
        disable_gcs_failsafe(vehicle) 
        arm_and_takeoff(vehicle, TARGET_ALT)
        upload_mission(vehicle)
        fly_mission_with_avoidance(vehicle)
    finally:
        print("Closing connection.")
        vehicle.close()
