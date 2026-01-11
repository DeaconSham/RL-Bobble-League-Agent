import numpy as np
import torch
from neural_network import actor_critic_neural_network

model = actor_critic_neural_network(obs_dim=17, act_dim=6)
obs = np.random.randn(17).astype(np.float32)
action = model.act(obs)

print(f"Observation: {obs}")
print(f"Action: {action}")