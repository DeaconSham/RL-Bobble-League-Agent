using Godot;
using System.Collections.Generic;
using System;

public partial class GameManager : Node
{
    [Export] Camera3D camera;
    [Export] Node players;
    [Export] RigidBody3D ball;
    [Export] float selectDistance;
    [Export] float dragDistance;
    Player launchPlayer = null;

    int turn = 1;
    int currentTeam = 1;

    List<Player> TeamOne = new List<Player>();
    List<Player> TeamTwo = new List<Player>();

    int scoreA = 0;
    int scoreB = 0;
    
    public override void _Input(InputEvent @event)
    {
        if (@event is InputEventMouseButton mouseButton) {
            bool stationaryBall = ball.LinearVelocity.Length() < 0.1;
            if (mouseButton.ButtonIndex == MouseButton.Left && mouseButton.Pressed) {
                if (stationaryBall) {
                    launchPlayer = playerClicked(currentTeam);
                    if (launchPlayer != null) {
                        launchPlayer.DrawArrow();
                    }
                }
            }
            
            if (mouseButton.ButtonIndex == MouseButton.Left && mouseButton.IsReleased()) {
                launchPlayer = null;
            }
            
            if (mouseButton.ButtonIndex == MouseButton.Right && mouseButton.IsReleased()) {
                GD.Print("team: " + currentTeam);
                if (currentTeam == 1) {
                    if (stationaryBall) {
                        foreach (Player player in TeamOne) {
                            player.ClearArrow();
                        }
                        currentTeam = 2;
                    }
                }
                else {
                    if (stationaryBall) {
                        Turn();
                        currentTeam = 1;
                    }
                }
            }
        }
    }

    public override void _Process(double delta) {
        if (launchPlayer != null) {
            Vector3 launchVector = (new Vector3(launchPlayer.Position.X, 0, launchPlayer.Position.Z) - MousePosition()) / dragDistance;
            launchPlayer.input = launchVector;
        }
    }

    // kind of shit but i didnt want to deal with godot raycast and colliders
    Player playerClicked(int desiredTeam) {
        // Compile the list of players
        
        // poopy ass coded keeps getting rearanged
        foreach (Player player in players.GetChildren()) {
            if (player.team == 1) {
                TeamOne.Add(player);
            }
        }
        foreach (Player player in players.GetChildren()) {
            if (player.team == 2) {
                TeamTwo.Add(player);
            }
        }
        
        List<Player> desiredPlayerList = new List<Player>();
        if (desiredTeam == 1) {
            desiredPlayerList = TeamOne;
        }
        else {
            desiredPlayerList = TeamTwo;
        }

        Vector3 mouse = MousePosition();
        
        // Closest player
        float closestDistance = float.MaxValue;
        Player closestPlayer = null;
        for (int i = 0; i < desiredPlayerList.Count; i++) {
            // faster calculation and doesnt matter lol idk ok cool
            // is a root calculation taht expensive ??
            float distance = desiredPlayerList[i].Position.DistanceSquaredTo(mouse);
            if (distance < closestDistance) {
                closestDistance = distance;
                closestPlayer = desiredPlayerList[i];
            }
        }

        if (closestDistance < selectDistance) {
            return closestPlayer;
        }
        else {
            return null;
        }
    }

    // Mouse position on xy plane
    Vector3 MousePosition() {
        Vector2 mousePosition = GetWindow().GetMousePosition();
        Plane plane = Plane.PlaneXZ;
        Vector3? intersect = plane.IntersectsRay(camera.Position, camera.ProjectRayNormal(mousePosition));
        if (intersect != null) {
            return intersect.Value;
        }
        else {
            return new Vector3(0, 0, 0);
        }
    }

    void Turn() {
        GD.Print("turn: " + turn);
        
        foreach (Player player in players.GetChildren()) {
            player.Launch();
        }
        
        turn++;
    }
        
    public void SCOREEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE(int team) {
        if (team == 1) {
            scoreA++;
        }
        else {
            scoreB++;
        }
        GD.Print("team: " + team + "scored");
        Reset();
    }
    
    [Export] Node spawns;

    [Export] Material TeamA;
    [Export] Material TeamB;
    [Export] PackedScene PlayerPrefab;

    public override void _EnterTree() {
        Spawn();
    }

    void Reset() {
        foreach (var node in players.GetChildren()) {
            var player = (Player)node; // yeah bro because casting would totally save us from an error bro again bro how many times do we have to tell you old man *farts*
            player.QueueFree();
        }
        ball.AngularVelocity = Vector3.Zero;
        ball.LinearVelocity = Vector3.Zero;
        ball.Position = new Vector3(0, 0.5f, 0);
        Spawn();
    }

    void Spawn() {
        foreach (var node in spawns.GetChildren()) {
            var spawn = (PlayerSpawn)node;
            Player player = PlayerPrefab.Instantiate<Player>();
            player.Position = new Vector3(spawn.Position.X, 1, spawn.Position.Z);
            
            MeshInstance3D mesh = player.GetChild(0) as MeshInstance3D;
            if (spawn.Team == 1) { mesh.SetSurfaceOverrideMaterial(0, TeamA);}
            else if (spawn.Team == 2) { mesh.SetSurfaceOverrideMaterial(0, TeamB);}
            
            player.team = spawn.Team;
            
            players.AddChild(player);
        }
    }
}
