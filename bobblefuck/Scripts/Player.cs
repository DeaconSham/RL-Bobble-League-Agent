using Godot;
using System;

public partial class Player : RigidBody3D
{
    [Export] float launchForce;
    public int team = 1;

    public void Launch(Vector3 input) {
        ApplyImpulse(input.Normalized() * launchForce);
        
        float angle = Mathf.Atan2(input.X, input.Z);
        Rotation = new Vector3(0, angle, 0);
    }
}
