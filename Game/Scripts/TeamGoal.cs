using Godot;
using System;

public partial class TeamGoal : Area3D
{
    [Export] GameManager GameManager; 
    [Export] int Team;
    
    public override void _Ready() {
        BodyEntered += OnBodyEntered;
    }

    private void OnBodyEntered(Node3D body) {
        if (body.IsInGroup("ball")) {
            GameManager.OnGoalScored(Team);
        }
    }
}
