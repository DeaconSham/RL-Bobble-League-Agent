using Godot;
using System.Collections.Generic;
using System;

public partial class LaunchManager : Node
{
    [Export] Camera3D camera;
    [Export] Node players;
    [Export] float selectDistance;
    [Export] float dragDistance;
    Player launchPlayer = null;

    int turn = 1;
    int currentTeam = 1;

    List<Player> TeamOne = new List<Player>();
    List<Player> TeamTwo = new List<Player>();
    
    override public void _EnterTree() {
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
    }
    
    public override void _Input(InputEvent @event)
    {
        if (@event is InputEventMouseButton mouseButton)
        {
            if (mouseButton.ButtonIndex == MouseButton.Left && mouseButton.Pressed) {
                launchPlayer = playerClicked(currentTeam);
                launchPlayer.DrawArrow();
            }
            
            if (mouseButton.ButtonIndex == MouseButton.Left && mouseButton.IsReleased()) {
                launchPlayer = null;
            }
            
            if (mouseButton.ButtonIndex == MouseButton.Right && mouseButton.IsReleased()) {
                GD.Print("team: " + currentTeam);
                if (currentTeam == 1) {
                    foreach (Player player in TeamOne) {
                        player.ClearArrow();
                    }
                    currentTeam = 2;
                }
                else {
                    Turn();
                    currentTeam = 1;
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
}
