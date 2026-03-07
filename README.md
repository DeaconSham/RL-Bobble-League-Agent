# RL Model for Bobble League

A Reinforcement Learning (PPO) agent that trains to play Bobble League by communicating with a Godot environment via shared memory.

## AI vs Static Method

Currently, the training supports learning against static opponents (Team B is controlled by the RL agent while Team A is static). *Note: Self-play is not implemented yet.*

### Prerequisites

1. **Python 3.12.4** or compatible environment.
2. Install the required Python packages:
   ```bash
   pip install -r Agent/requirements.txt
   ```
3. **Godot Engine** (to run the game environment).

### How to Run the Training

1. **Start the Godot Environment:**
   Launch Godot and open the project. Run the main game scene.
   *The Godot game acts as the environment server and will initialize the shared memory bridge necessary for AI communication.*

2. **Start the Python Agent:**
   Once the Godot instance is running and waiting, open your terminal, navigate to the `Agent` directory, and run the training loop:
   ```bash
   cd Agent
   python training_loop.py
   ```

3. **Observe Training:**
   The Python agent will attach to the shared memory initialized by Godot. PPO training will begin, and you should observe the agent controlling Team B players in the game window.
