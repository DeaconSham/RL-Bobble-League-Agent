using Godot;
using System;
using System.Collections.Generic;

public partial class ExporterTest : Node3D {
    [Export] JsonExporter Exporter;
    
    [Export] Node players;
    [Export] RigidBody3D ball;
    
    List<Player> playerList = new List<Player>();

    override public void _EnterTree() {
        foreach (var node in players.GetChildren()) {
            var player = (Player)node;
            playerList.Add(player);
        }
    }

    public override void _Process(double delta) {
        if (Input.IsKeyPressed(Key.Enter)) {
            RandomisePosition(new RandomNumberGenerator());
            Print();
            Exporter.DoTheThing(1, 2);
        }
    }

    void RandomisePosition(RandomNumberGenerator rnd) {
        foreach (Player player in playerList) {
            player.Position = new Vector3(rnd.RandfRange(-10, 10), 0, rnd.RandfRange(-10, 10));
        }
        ball.Position = new Vector3(rnd.RandfRange(-10, 10), 0, rnd.RandfRange(-10, 10));
    }

    void Print() {
        GD.Print("Team 1:");
        for (int i = 0; i < 4; i++) {
            GD.Print(playerList[i].Position);
        }
        GD.Print("Team 2:");
        for (int i = 4; i < 8; i++) {
            GD.Print(playerList[i].Position);
        }
        GD.Print("ball: " + ball.Position);
    }
}
