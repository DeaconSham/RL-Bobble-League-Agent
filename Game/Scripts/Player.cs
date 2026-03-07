using Godot;
using System;

public partial class Player : RigidBody3D
{
    [Export] public int Team = 1;
    
    [ExportGroup("Physics")] 
    [Export] float LaunchForce;
    [Export] float KickForceCoefficient;
    [Export] float VerticalKickForce;
    [Export] float KickSpeedThreshold;

    [ExportGroup("Visuals")] 
    [Export] float BaseWidth;
    [Export] float BaseWidthCoefficient;
    [Export] float TipSize;


    public Vector3 LaunchInput;

    [Export] Node3D ArrowBase;
    [Export] Node3D ArrowTip;
    bool _isHidden = true;

    public void Launch() {
        ApplyImpulse(LaunchInput.Normalized() * LaunchForce);
        
        if (LaunchInput.Length() > 0) {
            float angle = Mathf.Atan2(LaunchInput.X, LaunchInput.Z);
            Rotation = new Vector3(0, angle, 0);
        }
        
        _isHidden = false;
        ClearArrow();
        
        LaunchInput = new Vector3(0, 0, 0);
    }

    public void DrawArrow() {
        _isHidden = false;
        ArrowBase.Scale = new Vector3(BaseWidth, 1, Mathf.Clamp(LaunchInput.Length(), 0, 1) * BaseWidthCoefficient);
        ArrowTip.Scale = Vector3.One * TipSize;
        
        float angle = Mathf.Atan2(LaunchInput.X, LaunchInput.Z);
        ArrowBase.GlobalRotation = new Vector3(0, angle, 0);
        ArrowTip.GlobalRotation = new Vector3(0, ArrowBase.GlobalRotation.Y + float.DegreesToRadians(180), 0);
        
        ArrowBase.GlobalPosition = Position + (LaunchInput.Normalized() * Mathf.Clamp(LaunchInput.Length(), 0, 1)) + LaunchInput.Normalized() * 0.1f;
        ArrowTip.GlobalPosition = Position + (LaunchInput.Normalized() * Mathf.Clamp(LaunchInput.Length(), 0, 1) * 2) + LaunchInput.Normalized() * 0.6f;
        
        ArrowBase.Visible = true;
        ArrowTip.Visible = true;
    }

    public void ClearArrow() {
        ArrowBase.Visible = false;
        ArrowTip.Visible = false;
        _isHidden = true;
    }

    public override void _Process(double delta) {
        if (_isHidden == false) {
            DrawArrow();
        }
    }

    void Kick(Node3D node) {
        GD.Print(LinearVelocity);
        if (node.IsInGroup("ball")) {
            RigidBody3D ball = (RigidBody3D)node;
            if ((this.LinearVelocity - ball.LinearVelocity).Length() > KickSpeedThreshold) {
                Vector3 launchVector = LinearVelocity.Normalized() * KickForceCoefficient + new Vector3(0, VerticalKickForce, 0);
                ball.ApplyCentralImpulse(launchVector);
                GD.Print("kick performed");
            }
        }
    }
}
