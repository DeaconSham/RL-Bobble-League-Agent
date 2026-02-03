import numpy as np
import gymnasium as gym


class bobble_fuck_env(gym.Env):
    """
    Docstring for bobble_fuck_env:
    
    Gymnasium environment for Bobble League reinforcement learning.
    
    Observation Space:
        - Ball position (x, y): 2 values
        - 3 Friendly player positions (x, y): 6 values
        - 3 Enemy player positions (x, y): 6 values
        Total: 14-dimensional observation vector
    
    Action Space:
        - 3 Friendly player actions (power, angle): 6 values
        Total: 6-dimensional continuous action vector
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self):
        """
        Docstring for __init__:
        
        Initialize the Bobble League environment.
        """
        super().__init__()
        
        # Observation space: 14-dim vector
        # [ball_x, ball_y, f1_x, f1_y, f2_x, f2_y, f3_x, f3_y, e1_x, e1_y, e2_x, e2_y, e3_x, e3_y]
        self.observation_space = gym.spaces.Box(
            low=np.array([-500, -350] * 7, dtype=np.float32),
            high=np.array([500, 350] * 7, dtype=np.float32),
            dtype=np.float32
        )
        
        # Action space: 6-dim vector
        # [f1_power, f1_angle, f2_power, f2_angle, f3_power, f3_angle]
        # Power: [0, 1], Angle: [0, 360]
        self.action_space = gym.spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([1, 360, 1, 360, 1, 360], dtype=np.float32),
            dtype=np.float32
        )
        
        # Episode tracking
        self.ep_len = 0
        self.max_ep_len = 1000
        
    def reset(self):
        """
        Docstring for reset:
        
        Reset the environment to initial state.
        Uses the Gym API for compatibility with training loop.
        """
        self.ep_len = 0
        
        # TODO: Get initial state from Godot
        # For now, return a PLACEHOLDER observation
        obs = np.zeros(14, dtype=np.float32)
        
        return obs
    
    def step(self, action):
        """
        Docstring for step:
        
        Execute one step in the environment.
        Uses the old Gym API for compatibility with training loop.
        
        Args:
            action: Action to take (6-dim numpy array)
            
        Returns:
            observation: Current observation after action (14-dim numpy array)
            reward: Reward received (float)
            done: Whether episode ended (bool)
            info: Additional information (dict)
        """
        # TODO: Send action to Godot and get next state
        # For now, return placeholder values
        
        self.ep_len += 1
        
        # PLACEHOLDER observation
        obs = np.zeros(14, dtype=np.float32)
        
        # PLACEHOLDER reward
        reward = 0.0
        
        # Episode ends PLACEHOLDER
        done = (self.ep_len >= self.max_ep_len)
        
        # Additional info PLACEHOLDER
        info = {}
        
        return obs, reward, done, info
    
    def render(self, mode='human'):
        """
        Docstring for render:
        
        Render the environment (should be handled by Godot).
        
        Args:
            mode: Rendering mode
        """
        # Rendering handled by Godot
        pass
    
    def close(self):
        """
        Docstring for close:
        
        Clean up environment resources.
        """
        pass


# Helper function to create environment (used by training loop)
def bobble_fuck_env_helper():
    """
    Docstring for bobble_fuck_env_helper:
    
    Factory function to create bobble_fuck_env instance.
    Following OpenAI Spinning Up convention of using env_fn.
    
    Returns:
        env: bobble_fuck_env instance
    """
    return bobble_fuck_env()