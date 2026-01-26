import numpy as np
import gymnasium as gym

class BobbleFuckEnv(gym.Env):
    
    def __init__(self):
        self.observation_space = gym.spaces.Dict({
            # friendly team player positions
            "friendly1": gym.spaces.Box(low=np.array([-500, -350]), high=np.array([500, 350]), dtype=np.float32),
            "friendly2": gym.spaces.Box(low=np.array([-500, -350]), high=np.array([500, 350]), dtype=np.float32),
            "friendly3": gym.spaces.Box(low=np.array([-500, -350]), high=np.array([500, 350]), dtype=np.float32),

            # opposing team player positions
            "enemy1": gym.spaces.Box(low=np.array([-500, -350]), high=np.array([500, 350]), dtype=np.float32),
            "enemy2": gym.spaces.Box(low=np.array([-500, -350]), high=np.array([500, 350]), dtype=np.float32),
            "enemy3": gym.spaces.Box(low=np.array([-500, -350]), high=np.array([500, 350]), dtype=np.float32),

            # position of center of ball
            "ball": gym.spaces.Box(low=np.array([-500, -350]), high=np.array([500, 350]), dtype=np.float32),
        })

        self.action_space = gym.spaces.Dict({
            # friendly player launch power and directions
            "friendly1": gym.spaces.Box(low=np.array([0, 0]), high=np.array([1, 360]), dtype=np.float32),
            "friendly2": gym.spaces.Box(low=np.array([0, 0]), high=np.array([1, 360]), dtype=np.float32),
            "friendly3": gym.spaces.Box(low=np.array([0, 0]), high=np.array([1, 360]), dtype=np.float32),
        })


        