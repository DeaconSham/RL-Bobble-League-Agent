import numpy as np
import gymnasium as gym
from shared_memory_link import SharedMemoryLink


class BobbleGameEnv(gym.Env):
    """
    Gymnasium environment for Bobble League reinforcement learning.
    Communicates with Godot via shared memory bridge.
    
    Observation Space:
        - Ball position (x, z): 2 values
        - 3 Team A player positions (x, z): 6 values
        - 3 Team B (default AI) player positions (x, z): 6 values
        Total: 14-dimensional observation vector
    
    Action Space:
        - 3 Team B player actions (power x, power z): 6 values
        Total: 6-dimensional continuous action vector
    
    Note: For self-play (both teams), use SelfPlayEnv instead.
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self):
        """
        Initialize the Bobble League environment.
        Creates shared memory bridge to communicate with Godot.
        """
        super().__init__()
        
        # Observation space: 14-dim vector
        # [ball_x, ball_z, a1_x, a1_z, a2_x, a2_z, a3_x, a3_z, b1_x, b1_z, b2_x, b2_z, b3_x, b3_z]
        self.observation_space = gym.spaces.Box(
            low=np.array([-500, -350] * 7, dtype=np.float32),
            high=np.array([500, 350] * 7, dtype=np.float32),
            dtype=np.float32
        )
        
        # Action space: 6-dim vector (Team B only)
        # [b1_x, b1_z, b2_x, b2_z, b3_x, b3_z]
        self.action_space = gym.spaces.Box(
            low=-1.0 * np.ones(6, dtype=np.float32),
            high=1.0 * np.ones(6, dtype=np.float32),
            dtype=np.float32
        )
        
        # Episode tracking
        self.ep_len = 0
        self.max_ep_len = 1000
        
        # Shared memory bridge (attach to Godot-created shared memory)
        self.bridge = SharedMemoryLink(create=False)
        
    def reset(self):
        """
        Reset the environment to initial state.
        Signals Godot to reset the game and waits for initial observation.
        """
        self.ep_len = 0
        
        self.bridge.request_reset()
        self.bridge.wait_for_reset()
        
        obs = self.bridge.read_observation()
        
        return obs
    
    def step(self, action):
        """
        Execute one step in the environment.
        Sends action to Godot via shared memory and waits for the result.
        
        Args:
            action: Action to take (6-dim numpy array for Team B)
            
        Returns:
            observation: Current observation after action (14-dim numpy array)
            reward: Reward received (float)
            done: Whether episode ended (bool)
            info: Additional information (dict)
        """
        self.bridge.write_action(action)
        
        self.bridge.request_step()
        self.bridge.wait_for_step()
        
        self.ep_len += 1
        
        obs = self.bridge.read_observation()
        reward = self.bridge.read_reward()
        done = self.bridge.read_done() or (self.ep_len >= self.max_ep_len)
        info = {}
        
        return obs, reward, done, info
    
    def render(self, mode='human'):
        """Rendering handled by Godot."""
        pass
    
    def close(self):
        """Clean up environment resources."""
        self.bridge.close()


def make_bobble_env():
    """
    Factory function to create BobbleGameEnv instance.
    Following OpenAI Spinning Up convention of using env_fn.
    
    Returns:
        env: BobbleGameEnv instance
    """
    return BobbleGameEnv()
