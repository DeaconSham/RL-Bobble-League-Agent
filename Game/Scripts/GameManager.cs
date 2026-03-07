using Godot;
using System.Collections.Generic;
using System;

// manages game flow, player input, and RL step synchronization
public partial class GameManager : Node
{
    // team identifiers used across game logic
    private const int TeamAId = 1;
    private const int TeamBId = 2;

    // scene references
    [Export] Camera3D Camera;
    [Export] Node Players;
    [Export] RigidBody3D Ball;
    [Export] Node Spawns;
    [Export] float SelectDistance;
    [Export] float DragDistance;
    Player _launchPlayer = null;

    // player spawning
    [Export] Material TeamAMaterial;
    [Export] Material TeamBMaterial;
    [Export] PackedScene PlayerPrefab;

    // play mode determines input handling and RL integration
    public enum PlayMode {
        AI_VS_AI,
        HUMAN_VS_AI,
        HUMAN_VS_HUMAN
    }

    [Export] public PlayMode CurrentPlayMode;
    bool _humanWaitingForAI = false;

    int _turn = 1;
    int _currentTeam = TeamAId;

    int _scoreA = 0;
    int _scoreB = 0;

    // RL training state
    DataManager _dataManager;
    bool _isSimulating = false;
    float _stepReward = 0f;
    bool _currentDone = false;

    // handles mouse input for human players (drag to aim, right click to submit)
    public override void _Input(InputEvent @event) {
        if (@event is InputEventMouseButton mouseButton) {
            // Human vs AI input handling
            if (CurrentPlayMode == PlayMode.HUMAN_VS_AI) {
                if (_humanWaitingForAI) return;
                bool stationaryBall = Ball == null || Ball.LinearVelocity.Length() < 0.01;
                if (mouseButton.ButtonIndex == MouseButton.Left && mouseButton.Pressed) {
                    if (stationaryBall && StationaryPlayers()) {
                        _launchPlayer = PlayerClicked(TeamAId);
                        if (_launchPlayer != null) {
                            _launchPlayer.DrawArrow();
                        }
                    }
                }

                if (mouseButton.ButtonIndex == MouseButton.Left && mouseButton.IsReleased()) {
                    _launchPlayer = null;
                }

                // Right click confirms human turn and passes to AI
                if (mouseButton.ButtonIndex == MouseButton.Right && mouseButton.IsReleased()) {
                    if (stationaryBall && StationaryPlayers()) {
                        _humanWaitingForAI = true;
                    }
                }
            }
            
            // Human vs Human input handling
            if (CurrentPlayMode == PlayMode.HUMAN_VS_HUMAN) {
                bool stationaryBall = StationaryBall();
                if (mouseButton.ButtonIndex == MouseButton.Left && mouseButton.Pressed) {
                    if (stationaryBall && StationaryPlayers()) {
                        _launchPlayer = PlayerClicked(_currentTeam);
                        if (_launchPlayer != null) {
                            _launchPlayer.DrawArrow();
                        }
                    }
                }
            
                if (mouseButton.ButtonIndex == MouseButton.Left && mouseButton.IsReleased()) {
                    _launchPlayer = null;
                }
            
                if (mouseButton.ButtonIndex == MouseButton.Right && mouseButton.IsReleased()) {
                    GD.Print("Current team: " + _currentTeam);
                    if (_currentTeam == TeamAId) {
                        if (stationaryBall) {
                            // Clear Team A arrows before switching to Team B
                            foreach (Player player in Players.GetChildren()) {
                                if (player.Team == TeamAId) {
                                    player.ClearArrow();
                                }
                            }
                            _currentTeam = TeamBId;
                        }
                    }
                    else {
                        if (stationaryBall && StationaryPlayers()) {
                            Turn();
                            _currentTeam = TeamAId;
                        }
                    }
                }
            }
        }
    }

    // updates the aiming arrow direction while dragging a player
    public override void _Process(double delta) {
        if (CurrentPlayMode is PlayMode.HUMAN_VS_HUMAN or PlayMode.HUMAN_VS_AI) {
            if (_launchPlayer != null) {
                Vector3 launchVector = (new Vector3(_launchPlayer.Position.X, 0, _launchPlayer.Position.Z) - MousePosition()) / DragDistance;
                _launchPlayer.LaunchInput = launchVector;
            }
        }
    }

    // finds closest player to mouse on the desired team within select distance
    Player PlayerClicked(int desiredTeam) {
        List<Player> desiredPlayerList = new List<Player>();
        foreach (Player player in Players.GetChildren()) {
            if (player.Team == desiredTeam) {
                desiredPlayerList.Add(player);
            }
        }

        Vector3 mouse = MousePosition();
        
        float closestDistance = float.MaxValue;
        Player closestPlayer = null;
        for (int i = 0; i < desiredPlayerList.Count; i++) {
            // DistanceSquaredTo avoids unnecessary sqrt
            float distance = desiredPlayerList[i].Position.DistanceSquaredTo(mouse);
            if (distance < closestDistance) {
                closestDistance = distance;
                closestPlayer = desiredPlayerList[i];
            }
        }

        if (closestDistance < SelectDistance) {
            return closestPlayer;
        }
        else {
            return null;
        }
    }

    // projects mouse screen position onto the XZ ground plane
    Vector3 MousePosition() {
        Vector2 mousePosition = GetWindow().GetMousePosition();
        Plane plane = Plane.PlaneXZ;
        Vector3? intersect = plane.IntersectsRay(Camera.Position, Camera.ProjectRayNormal(mousePosition));
        if (intersect != null) {
            return intersect.Value;
        }
        else {
            return new Vector3(0, 0, 0);
        }
    }

    // launches all players and advances the turn counter
    void Turn() {
        GD.Print("Turn: " + _turn);
        
        foreach (Player player in Players.GetChildren()) {
            player.Launch();
        }
        
        _turn++;
    }

    [Export] TextureRect GoalOverlay;
    
    // called by TeamGoal when ball enters a goal, updates score and reward
    public void OnGoalScored(int team) {
        if (team == TeamAId) {
            _scoreA++;
            // Agent is Team A, so scoring gives +1 reward
            _stepReward += 1.0f;
        }
        else {
            _scoreB++;
            // Getting scored on is -1 reward
            _stepReward -= 1.0f;
        }
        GD.Print("Team " + team + " scored");
        Reset();
        GoalOverlay.SetVisible(true);
        if (GoalOverlay.Texture is AnimatedTexture animatedTexture) {
            animatedTexture.CurrentFrame = 0;
        }
    }

    // spawns initial players when scene loads
    public override void _EnterTree() {
        Spawn();
    }

    // destroys all players, resets ball, and respawns
    void Reset() {
        foreach (var node in Players.GetChildren()) {
            var player = (Player)node;
            player.QueueFree();
        }
        Ball.AngularVelocity = Vector3.Zero;
        Ball.LinearVelocity = Vector3.Zero;
        Ball.Position = new Vector3(0, 0.5f, 0);
        Spawn();
    }
    
    // checks if all players have stopped moving
    bool StationaryPlayers() {
        bool value = true;
        foreach (Node node in Players.GetChildren()) {
            if (node.IsQueuedForDeletion()) continue;
            if (node is Player p && p.LinearVelocity.Length() >= 0.01f) {
                value = false;
                break;
            }
        }
        return value;
    }

    // checks if the ball has stopped moving
    bool StationaryBall() {
        return Ball == null || Ball.LinearVelocity.Length() < 0.01f;
    }

    // instantiates players at spawn points with team materials
    void Spawn() {
        foreach (var node in Spawns.GetChildren()) {
            var spawn = (PlayerSpawn)node;
            Player player = PlayerPrefab.Instantiate<Player>();
            player.Position = new Vector3(spawn.Position.X, 1.001f, spawn.Position.Z);
            
            MeshInstance3D mesh = player.GetChild(0) as MeshInstance3D;
            if (spawn.Team == TeamAId) { mesh.SetSurfaceOverrideMaterial(0, TeamAMaterial);}
            else if (spawn.Team == TeamBId) { mesh.SetSurfaceOverrideMaterial(0, TeamBMaterial);}
            
            player.Team = spawn.Team;
            
            Players.AddChild(player);
        }
    }

    // main RL loop: waits for python action, applies it, then exports results after simulation
    public override void _PhysicsProcess(double delta) {
        if (_isSimulating) {
            if (StationaryBall() && StationaryPlayers()) {
                float finalReward = (_stepReward) + GetDistanceReward();
                
                _dataManager.Export(_currentDone, finalReward);
                _stepReward = 0;
                _currentDone = false;
                _isSimulating = false;
            }
            return;
        }

        // Wait for human to submit their turn before processing AI requests
        if (CurrentPlayMode == PlayMode.HUMAN_VS_AI && !_humanWaitingForAI) {
            return;
        }

        bool resetRequested;
        float[] action = _dataManager.ProcessRequests(out resetRequested);

        if (resetRequested) {
            Reset();
            _dataManager.Export();
            _stepReward = 0;
            _currentDone = false;
            _humanWaitingForAI = false;
            return;
        }

        // Only process step requests if Python provides an action
        if (action != null) {
            if (CurrentPlayMode == PlayMode.AI_VS_AI) {
                ApplyRLAction(action);
                _isSimulating = true;
            } 
            else if (CurrentPlayMode == PlayMode.HUMAN_VS_AI) {
                // Apply AI's predicted actions only to Team B
                ApplyRLActionHalf(action, TeamBId);
                
                // Human's inputs (Team A) are already set via LaunchInput from _Process
                foreach (Node node in Players.GetChildren()) {
                    if (node is Player p) p.Launch();
                }
                
                _isSimulating = true;
                _humanWaitingForAI = false;
            }
        }
    }

    // reward shaping based on ball distance to opponent goal
    float GetDistanceReward() {
        if (Ball == null) return 0f;
        // From Team A POV: opponent goal is at Z=-33. 
        // Closer to opponent goal (Z = -33) gives positive reward.
        // Closer to own goal (Z = 33) gives negative reward.
        return -Ball.Position.Z / 330.0f;
    }

    // applies python's action array to players (6 for single-agent, 12 for self-play)
    void ApplyRLAction(float[] action) {
        int actionIdx = 0;
        int playersToControl = action.Length / 2;

        if (playersToControl == 3) {
            // Phase 1 (Single Agent): 6 floats = AI controls Team B only, Team A stays static
            foreach (Node node in Players.GetChildren()) {
                if (node.IsQueuedForDeletion()) continue;
                if (node is Player player && player.Team == TeamBId) {
                    if (actionIdx < action.Length) {
                        float x = action[actionIdx];
                        float y = action[actionIdx + 1];
                        actionIdx += 2;
                        player.LaunchInput = new Vector3(x, 0, y);
                        player.Launch();
                    }
                }
            }
        }
        else if (playersToControl == 6) {
            // Phase 3 (Self-Play): 12 floats = both Team A and Team B
            foreach (Node node in Players.GetChildren()) {
                if (node.IsQueuedForDeletion()) continue;
                if (node is Player player) {
                    if (actionIdx < action.Length) {
                        float x = action[actionIdx];
                        float y = action[actionIdx + 1];
                        actionIdx += 2;
                        player.LaunchInput = new Vector3(x, 0, y);
                        player.Launch();
                    }
                }
            }
        }
        else {
            GD.PrintErr($"Unsupported action array length: {action.Length}");
        }
    }

    // applies python's action array only to the specified team (human vs AI mode)
    void ApplyRLActionHalf(float[] action, int teamTarget) {
        int actionIdx = 0;
        foreach (Node node in Players.GetChildren()) {
            if (node.IsQueuedForDeletion()) continue;
            if (node is Player player) {
                if (actionIdx < action.Length) {
                    float x = action[actionIdx];
                    float y = action[actionIdx + 1];
                    actionIdx += 2;
                    
                    if (player.Team == teamTarget) {
                        player.LaunchInput = new Vector3(x, 0, y);
                    }
                }
            }
        }
    }
}
