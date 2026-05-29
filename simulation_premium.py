import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button

# =====================================================
# STATE
# =====================================================

START_LAT = 21.25
START_LON = 81.63
START_ALT = 1000

TARGET_LAT = 21.26
TARGET_LON = 81.64

lat = START_LAT
lon = START_LON
alt = START_ALT

current_heading = 0

wind_strength = 5
flight_time = 0

mission_status = "ACTIVE"
report_printed = False

path_lat = []
path_lon = []

alt_history = []
error_history = []
distance_history = []

events = []

# =====================================================
# RL MODEL
# =====================================================

Q = np.load("q_table.npy")

error_bins = np.linspace(-180, 180, 10)
dist_bins = np.linspace(0, 0.01, 10)

actions = [-10, 0, 10]

def rl_controller(error, distance):

    e_bin = np.digitize(error, error_bins)
    d_bin = np.digitize(distance, dist_bins)

    e_bin = max(0, min(9, e_bin))
    d_bin = max(0, min(9, d_bin))

    action_idx = np.argmax(Q[e_bin, d_bin])

    return actions[action_idx]

# =====================================================
# HELPERS
# =====================================================

def log_event(msg):

    if len(events) > 8:
        events.pop(0)

    events.append(msg)

def calculate_bearing(lat1, lon1, lat2, lon2):

    dLon = math.radians(lon2 - lon1)

    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)

    y = math.sin(dLon) * math.cos(lat2)

    x = (
        math.cos(lat1) * math.sin(lat2)
        - math.sin(lat1) * math.cos(lat2) * math.cos(dLon)
    )

    return (math.degrees(math.atan2(y, x)) + 360) % 360

def heading_error(target, current):

    return (target - current + 180) % 360 - 180

def move(lat, lon, alt, heading):

    descent_rate = 2
    glide_ratio = 5

    alt -= descent_rate

    step = descent_rate * glide_ratio * 0.00001

    lat += step * math.cos(math.radians(heading))
    lon += step * math.sin(math.radians(heading))

    return lat, lon, alt

def get_wind(x, y):

    return wind_strength * math.sin(x * 40 + y * 40)

# =====================================================
# TERRAIN
# =====================================================

x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)

X, Y = np.meshgrid(x, y)

terrain = (
    0.8 * np.exp(-(X**2 + Y**2))
    + 0.6 * np.exp(-((X - 2)**2 + (Y + 1)**2))
    + 0.5 * np.exp(-((X + 1.5)**2 + (Y - 2)**2))
    + 0.3 * np.sin(3 * X)
    + 0.2 * np.cos(4 * Y)
)

# =====================================================
# UI
# =====================================================

plt.style.use("dark_background")

fig = plt.figure(figsize=(15, 8))

ax_map = fig.add_subplot(121)
ax_error = fig.add_subplot(222)
ax_alt = fig.add_subplot(224)

ax_wind = plt.axes([0.25, 0.02, 0.45, 0.03])
wind_slider = Slider(
    ax_wind,
    "Wind",
    0,
    15,
    valinit=5
)

ax_button = plt.axes([0.82, 0.92, 0.12, 0.05])
reset_button = Button(ax_button, "Reset")

# =====================================================
# UPDATE
# =====================================================

def update(frame):

    global lat
    global lon
    global alt
    global current_heading
    global wind_strength
    global flight_time
    global mission_status
    global report_printed

    wind_strength = wind_slider.val

    if mission_status != "LANDED":
        flight_time += 1

    distance = math.sqrt(
        (lat - TARGET_LAT) ** 2 +
        (lon - TARGET_LON) ** 2
    )

    distance_m = distance * 111000

    target_heading = calculate_bearing(
        lat,
        lon,
        TARGET_LAT,
        TARGET_LON
    )

    error = heading_error(
        target_heading,
        current_heading
    )

    if mission_status != "LANDED":

        turn = rl_controller(error, distance)

        current_heading += turn * 0.1

        current_heading += (
            get_wind(lat, lon) * 0.1
        )

        current_heading %= 360

        lat, lon, alt = move(
            lat,
            lon,
            alt,
            current_heading
        )

    path_lat.append(lat)
    path_lon.append(lon)

    alt_history.append(alt)
    error_history.append(error)
    distance_history.append(distance_m)

    if alt <= 0 and mission_status != "LANDED":

        mission_status = "LANDED"

        if not report_printed:

            print("\nMISSION REPORT")
            print("---------------------")
            print(f"Flight Time: {flight_time}s")
            print(f"Landing Error: {distance_m:.2f} m")
            print(f"Wind: {wind_strength:.1f}")
            print("Controller: RL")
            print("---------------------")

            report_printed = True

    if distance < 0.0003 and alt < 50:
        mission_status = "TARGET LOCKED"

    # =================================================
    # MAP
    # =================================================

    ax_map.clear()

    ax_map.imshow(
        terrain,
        extent=[81.62, 81.65, 21.24, 21.27],
        cmap="terrain",
        alpha=0.35,
        origin="lower"
    )

    ax_map.set_xlim(81.62, 81.65)
    ax_map.set_ylim(21.24, 21.27)

    # wind arrows
    for wx in np.linspace(81.62, 81.65, 6):
        for wy in np.linspace(21.24, 21.27, 6):

            u = 0.0002 * math.cos(wx * 50)
            v = 0.0002 * math.sin(wy * 50)

            ax_map.arrow(
                wx,
                wy,
                u,
                v,
                color="yellow",
                alpha=0.3,
                head_width=0.00012
            )

    # predicted path
    future_lon = []
    future_lat = []

    temp_lat = lat
    temp_lon = lon

    for _ in range(50):

        temp_lat += (
            0.00001 *
            math.cos(
                math.radians(current_heading)
            )
        )

        temp_lon += (
            0.00001 *
            math.sin(
                math.radians(current_heading)
            )
        )

        future_lat.append(temp_lat)
        future_lon.append(temp_lon)

    ax_map.plot(
        future_lon,
        future_lat,
        "--",
        color="yellow",
        linewidth=2,
        alpha=0.7
    )

    # actual path
    ax_map.plot(
        path_lon,
        path_lat,
        color="cyan",
        linewidth=2
    )

    # no fly zone
    nofly = plt.Circle(
        (81.635, 21.255),
        0.002,
        color="red",
        alpha=0.25
    )

    ax_map.add_patch(nofly)

    # landing zone
    landing_zone = plt.Circle(
        (TARGET_LON, TARGET_LAT),
        0.001,
        fill=False,
        color="lime",
        linewidth=2
    )

    ax_map.add_patch(landing_zone)

    # aircraft
    ax_map.scatter(
        lon,
        lat,
        s=300,
        marker=(3, 0, current_heading),
        color="cyan"
    )

    # target
    ax_map.scatter(
        TARGET_LON,
        TARGET_LAT,
        s=150,
        color="lime"
    )

    # HUD
    hud_y = 21.269

    score = max(
        0,
        100 - (distance_m / 20)
    )

    ax_map.text(
        81.6215,
        hud_y,
        "🛰 AUTONOMOUS PARAFOIL MISSION",
        color="cyan",
        fontsize=10,
        fontweight="bold"
    )

    ax_map.text(
        81.6215,
        hud_y - 0.002,
        f"Altitude: {alt:.0f} m"
    )

    ax_map.text(
        81.6215,
        hud_y - 0.004,
        f"Heading: {current_heading:.1f}°"
    )

    ax_map.text(
        81.6215,
        hud_y - 0.006,
        f"Distance: {distance_m:.1f} m"
    )

    ax_map.text(
        81.6215,
        hud_y - 0.008,
        f"Wind: {wind_strength:.1f}"
    )

    ax_map.text(
        81.6215,
        hud_y - 0.010,
        f"Time: {flight_time}s"
    )

    ax_map.text(
        81.6215,
        hud_y - 0.012,
        f"Status: {mission_status}",
        color="orange"
    )

    ax_map.text(
        81.6215,
        hud_y - 0.014,
        f"Mission Score: {score:.0f}",
        color="lime"
    )

    ax_map.set_title(
        "Mission Control Dashboard",
        color="cyan",
        fontsize=14
    )

    # =================================================
    # ERROR GRAPH
    # =================================================

    ax_error.clear()

    ax_error.plot(
        error_history,
        color="cyan"
    )

    ax_error.set_title(
        "Heading Error"
    )

    ax_error.grid(True)

    # =================================================
    # ALTITUDE GRAPH
    # =================================================

    ax_alt.clear()

    ax_alt.plot(
        alt_history,
        color="orange"
    )

    ax_alt.set_title(
        "Altitude"
    )

    ax_alt.grid(True)

# =====================================================
# RESET
# =====================================================

def reset(event):

    global lat
    global lon
    global alt
    global current_heading
    global flight_time
    global mission_status
    global report_printed

    lat = START_LAT
    lon = START_LON
    alt = START_ALT

    current_heading = 0

    flight_time = 0
    mission_status = "ACTIVE"
    report_printed = False

    path_lat.clear()
    path_lon.clear()

    alt_history.clear()
    error_history.clear()
    distance_history.clear()

    events.clear()

reset_button.on_clicked(reset)

# =====================================================
# RUN
# =====================================================

ani = FuncAnimation(
    fig,
    update,
    interval=100
)

plt.show()