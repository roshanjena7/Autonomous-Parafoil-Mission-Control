import math
import random
import torch
import torch.nn as nn
import torch.optim as optim

# -------------------------------
# 🧠 MODEL
# -------------------------------
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.net(x)


model = DQN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

actions = [-10, 0, 10]

# -------------------------------
# 🌍 ENVIRONMENT FUNCTIONS
# -------------------------------
def get_distance(lat, lon):
    return math.sqrt((lat - 21.26)**2 + (lon - 81.64)**2)

def step_env(lat, lon, heading, action):
    heading += action
    lat += 0.0001 * math.cos(math.radians(heading))
    lon += 0.0001 * math.sin(math.radians(heading))
    return lat, lon, heading

def get_state(error, dist):
    return torch.tensor([error / 180.0, dist * 100], dtype=torch.float32)

# -------------------------------
# 🚀 TRAINING LOOP
# -------------------------------
episodes = 2000
epsilon = 0.2
gamma = 0.9

for ep in range(episodes):

    lat, lon = 21.25, 81.63
    heading = 0

    for t in range(200):

        dist = get_distance(lat, lon)

        error = (math.degrees(math.atan2(21.26 - lat, 81.64 - lon)) - heading + 180) % 360 - 180

        state = get_state(error, dist)

        # choose action
        if random.random() < epsilon:
            action_idx = random.randint(0, 2)
        else:
            with torch.no_grad():
                q_values = model(state)
                action_idx = torch.argmax(q_values).item()

        action = actions[action_idx]

        # step environment
        new_lat, new_lon, new_heading = step_env(lat, lon, heading, action)
        new_dist = get_distance(new_lat, new_lon)

        reward = -new_dist

        new_error = (math.degrees(math.atan2(21.26 - new_lat, 81.64 - new_lon)) - new_heading + 180) % 360 - 180
        next_state = get_state(new_error, new_dist)

        # Q update
        q_values = model(state)
        next_q = model(next_state)

        target = q_values.clone().detach()
        target[action_idx] = reward + gamma * torch.max(next_q)

        loss = loss_fn(q_values, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lat, lon, heading = new_lat, new_lon, new_heading

    if ep % 200 == 0:
        print(f"Episode {ep} done")

print("✅ Training complete")

torch.save(model.state_dict(), "dqn_model.pth")
print("💾 Model saved: dqn_model.pth")