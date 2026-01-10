using Godot;
using System;

public partial class Player : RigidBody3D
{
    [ExportGroup("Physics")] [Export] float launchForce;

    [ExportGroup("Visuals")] 
    [Export] float baseWidth;
    [Export] float baseWidthCoef;
    [Export] float tipSize; // 🥵

    public int team = 1;

    public Vector3 input;

    [Export] Node3D ArrowBase;
    [Export] Node3D ArrowTip;

    public void Launch() {
        ApplyImpulse(input.Normalized() * launchForce);

        float angle = Mathf.Atan2(input.X, input.Z);
        Rotation = new Vector3(0, angle, 0);
        
        ClearArrow();
    }

    void DrawArrow() {
        ArrowBase.Scale = new Vector3(baseWidth, 1, Mathf.Clamp(input.Length(), 0, 1) * baseWidthCoef);
        ArrowTip.Scale = Vector3.One * tipSize;
        
        float angle = Mathf.Atan2(input.X, input.Z);
        ArrowBase.GlobalRotation = new Vector3(0, angle, 0);
        ArrowTip.GlobalRotation = new Vector3(0, float.RadiansToDegrees(ArrowBase.GlobalRotation.Y), 0);
        
        ArrowBase.GlobalPosition = Position + (input.Normalized() * Mathf.Clamp(input.Length(), 0, 1)) + input.Normalized() * 0.1f;
        ArrowTip.GlobalPosition = Position + (input.Normalized() * Mathf.Clamp(input.Length(), 0, 1) * 2) + input.Normalized() * 0.1f;
        
        ArrowBase.Visible = true;
        ArrowTip.Visible = true;
    }

    void ClearArrow() {
        ArrowBase.Visible = false;
        ArrowTip.Visible = false;
    }

    public override void _Process(double delta) {
        DrawArrow();
    }
}
