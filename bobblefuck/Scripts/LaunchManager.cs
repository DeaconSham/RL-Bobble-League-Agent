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
    
    public override void _Input(InputEvent @event)
    {
        if (@event is InputEventMouseButton mouseButton)
        {
            if (mouseButton.ButtonIndex == MouseButton.Left && mouseButton.Pressed) {
                launchPlayer = playerClicked(1);
                launchPlayer.DrawArrow();
            }
            
            if (mouseButton.ButtonIndex == MouseButton.Left && mouseButton.IsReleased()) {
                launchPlayer.Launch();
                launchPlayer = null;
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
        int playerAmount = players.GetChildCount();
        foreach (Player player in players.GetChildren()) {
            if (player.team == desiredTeam) {
                desiredPlayerList.Add(player);
            }
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

    void DrawArrows() {
        
    }
}
