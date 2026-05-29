import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button

# -------------------------------
# 🌍 STATE
# -------------------------------
lat, lon, alt = 21.25, 81.63, 1000
target_lat, target_lon = 21.26, 81.64

current_heading = 0
wind_strength = 5

path_lat, path_lon = [], []
alt_history, error_history = [], []

# -------------------------------
# 🧠 LOAD RL MODEL
# -------------------------------
Q = np.load("q_table.npy")

error_bins = np.linspace(-180, 180, 10)
dist_bins = np.linspace(0, 0.01, 10)
actions = [-10, 0, 10]

def rl_controller(error, distance):
    e_bin = int(np.digitize(error, error_bins))
    d_bin = int(np.digitize(distance, dist_bins))

    e_bin = min(max(e_bin, 0), 9)
    d_bin = min(max(d_bin, 0), 9)

    action_idx = np.argmax(Q[e_bin, d_bin])
    return actions[action_idx]

# -------------------------------
# 🧭 FUNCTIONS
# -------------------------------
def calculate_bearing(lat1, lon1, lat2, lon2):
    dLon = math.radians(lon2 - lon1)
    lat1, lat2 = math.radians(lat1), math.radians(lat2)

    y = math.sin(dLon) * math.cos(lat2)
    x = math.cos(lat1)*math.sin(lat2) - math.sin(lat1)*math.cos(lat2)*math.cos(dLon)

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

# -------------------------------
# 🌪️ WIND
# -------------------------------
def get_wind(x, y):
    return wind_strength * math.sin(x*40 + y*40)

# -------------------------------
# 📋 EVENTS
# -------------------------------
events = []
flight_time = 0
mission_status = "ACTIVE"

def log_event(msg):
    if len(events) > 6:
        events.pop(0)
    events.append(msg)

# -------------------------------
# 🎮 UI
# -------------------------------
plt.style.use('dark_background')
fig = plt.figure(figsize=(12,6))

ax_map = fig.add_subplot(121)
ax_plot = fig.add_subplot(222)
ax_alt = fig.add_subplot(224)

ax_wind = plt.axes([0.25, 0.02, 0.5, 0.02])
wind_slider = Slider(ax_wind, 'Wind', 0, 15, valinit=5)

ax_button = plt.axes([0.8, 0.9, 0.1, 0.05])
reset_button = Button(ax_button, 'Reset')

# -------------------------------
# 🔁 UPDATE
# -------------------------------
def update(frame):
    global lat, lon, alt, current_heading
    global wind_strength
    global flight_time
    global mission_status

    wind_strength = wind_slider.val
    flight_time += 1

    if alt <= 0:
        log_event("🟢 LANDED")
        return

    distance = math.sqrt((lat-target_lat)**2 + (lon-target_lon)**2)
    distance_m = distance * 111000

    target_heading = calculate_bearing(lat, lon, target_lat, target_lon)
    error = heading_error(target_heading, current_heading)

    # 🧠 RL CONTROL
    turn = rl_controller(error, distance)

    current_heading += turn * 0.1
    current_heading %= 360

    # wind effect
    current_heading += get_wind(lat, lon) * 0.1

    lat, lon, alt = move(lat, lon, alt, current_heading)

    path_lat.append(lat)
    path_lon.append(lon)
    alt_history.append(alt)
    error_history.append(error)

    if distance < 0.0003 and alt < 50:
        log_event("🎯 TARGET LOCKED")
        return

    # ---------------- MAP ----------------
    ax_map.clear()
    ax_map.set_xlim(81.62, 81.65)
    ax_map.set_ylim(21.24, 21.27)

    ax_map.plot(path_lon, path_lat, color='cyan')
    ax_map.scatter(
    lon,
    lat,
    s=150,
    marker=">",
    color="cyan"
)
    ax_map.scatter(target_lon, target_lat, color='red')
landing_zone = plt.Circle(
    (target_lon, target_lat),
    0.001,
    color='red',
    fill=False,
    linewidth=2
)

ax_map.add_patch(landing_zone)

ax_map.text(
        81.622,
        21.266,
        f"Altitude: {alt:.0f} m",
        color='white'
    )

ax_map.text(
        81.622,
        21.264,
        f"Time: {flight_time}s",
        color='orange'
    )

ax_map.text(
        81.622,
        21.262,
        f"Wind: {wind_strength:.1f}",
        color='cyan'
    )

ax_map.text(
        81.622,
        21.260,
        f"Status: {mission_status}",
        color='lime'
    )

for i, e in enumerate(events):
        ax_map.text(81.622, 21.264 - i*0.002, e, color='lime', fontsize=8)

ax_map.set_title(
    "🛰️ Autonomous Parafoil Mission Control",
    color="cyan",
    fontsize=14
   )

    # ---------------- PLOTS ----------------
ax_plot.clear()
ax_plot.plot(error_history, color='cyan')
ax_plot.set_title("Error")

ax_alt.clear()
ax_alt.plot(alt_history, color='orange')
ax_alt.set_title("Altitude")

# -------------------------------
# RESET
# -------------------------------
def reset(event):
    global lat, lon, alt, current_heading

    lat, lon, alt = 21.25, 81.63, 1000
    current_heading = 0

    path_lat.clear()
    path_lon.clear()
    alt_history.clear()
    error_history.clear()
    events.clear()

reset_button.on_clicked(reset)

# -------------------------------
# RUN
# -------------------------------
ani = FuncAnimation(fig, update, interval=100)
plt.show()