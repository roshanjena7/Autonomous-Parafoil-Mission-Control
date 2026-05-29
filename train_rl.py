import math
import random
import numpy as np

# -------------------------------
# 🎯 Q TABLE
# -------------------------------
Q = np.zeros((10, 10, 3))  # error, distance, action

error_bins = np.linspace(-180, 180, 10)
dist_bins = np.linspace(0, 0.01, 10)

actions = [-10, 0, 10]

alpha = 0.1
gamma = 0.9
epsilon = 0.2

# -------------------------------
# 🧠 HELPERS
# -------------------------------
def discretize(val, bins):
    return int(np.digitize(val, bins))

def clamp(x):
    return min(max(x, 0), 9)

def distance(lat, lon):
    return math.sqrt((lat-21.26)**2 + (lon-81.64)**2)

def step(lat, lon, heading, action):
    heading += action
    lat += 0.0001 * math.cos(math.radians(heading))
    lon += 0.0001 * math.sin(math.radians(heading))
    return lat, lon, heading

# -------------------------------
# 🚀 TRAINING
# -------------------------------
for episode in range(3000):

    lat, lon = 21.25, 81.63
    heading = 0

    for t in range(200):

        dist = distance(lat, lon)

        error = (math.degrees(math.atan2(21.26-lat, 81.64-lon)) - heading + 180) % 360 - 180

        e_bin = clamp(discretize(error, error_bins))
        d_bin = clamp(discretize(dist, dist_bins))

        # choose action
        if random.random() < epsilon:
            action_idx = random.randint(0, 2)
        else:
            action_idx = np.argmax(Q[e_bin, d_bin])

        action = actions[action_idx]

        new_lat, new_lon, new_heading = step(lat, lon, heading, action)
        new_dist = distance(new_lat, new_lon)

        reward = -new_dist

        ne_bin = clamp(discretize(error, error_bins))
        nd_bin = clamp(discretize(new_dist, dist_bins))

        Q[e_bin, d_bin, action_idx] += alpha * (
            reward + gamma * np.max(Q[ne_bin, nd_bin]) - Q[e_bin, d_bin, action_idx]
        )

        lat, lon, heading = new_lat, new_lon, new_heading

print("Training done")

np.save("q_table.npy", Q)
print("Model saved: q_table.npy")