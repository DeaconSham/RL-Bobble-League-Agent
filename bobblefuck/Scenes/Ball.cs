using Godot;
using System;

public partial class Ball : RigidBody3D {
    [Export] float forfuckssake; // temp fix
    public override void _PhysicsProcess(double delta) {
        if (LinearVelocity.Y != MathF.Abs(LinearVelocity.Y)) {
            if (Position.Y < 0.55f && MathF.Abs(LinearVelocity.Y) < forfuckssake) {
                LinearVelocity = new Vector3(LinearVelocity.X, 0, LinearVelocity.Z);
                Position = new Vector3(Position.X, 0.5f, Position.Z);
            }
        }
    }
}
