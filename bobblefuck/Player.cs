using Godot;
using System;

public partial class Player : RigidBody3D
{
    [Export] float launchForce;
    public int team = 1;

    private float velocity = 0;

    public void Launch(Vector3 input) {
        ApplyImpulse(input.Normalized() * launchForce);
    }
}
