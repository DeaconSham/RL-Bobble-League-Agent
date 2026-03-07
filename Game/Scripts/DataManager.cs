using Godot;
using System;
using System.IO;

public partial class DataManager : Node {

    private const int TeamAId = 1;
    private const int TeamBId = 2;

    SharedMemory _memory;

    [Export] Node Players;
    [Export] RigidBody3D Ball;

    public void Export(bool done = false, float reward = 0f) {
        SendToMemory(GetTeamPositions(TeamAId), GetTeamPositions(TeamBId), new Vector2(Ball.Position.X, Ball.Position.Z), done, reward);
    }

    float[] GetTeamPositions(int team) {
        int teamSize = Players.GetChildCount() / 2;
        float[] teamArray = new float[teamSize * 2];

        int idx = 0;
        for (int i = 0; i < Players.GetChildCount(); i++) {
            Player player = (Player)(Players.GetChild(i));
            if (player.Team == team) {
                teamArray[idx] = player.Position.X;
                teamArray[idx + 1] = player.Position.Z;
                idx += 2;
            }
        }

        return teamArray;
    }

    unsafe void SendToMemory(float[] teamAPositions, float[] teamBPositions, Vector2 ballPosition, bool done = false, float reward = 0f) {
        float[] teamPositions = new float[teamAPositions.Length + teamBPositions.Length];
        Array.Copy(teamAPositions, 0, teamPositions, 0, teamAPositions.Length);
        Array.Copy(teamBPositions, 0, teamPositions, teamAPositions.Length, teamBPositions.Length);

        float[] ballPositionArray = { ballPosition.X, ballPosition.Y };
        float[] positionArray = new float[ballPositionArray.Length + teamPositions.Length];
        Array.Copy(teamPositions, 0, positionArray, 0, teamPositions.Length);
        Array.Copy(ballPositionArray, 0, positionArray, teamPositions.Length, ballPositionArray.Length);

        _memory.WriteReward(reward);
        _memory.SendStepResults(positionArray, 0, done);
    }

    unsafe void Import() {

    }

    public override void _EnterTree() {
        _memory = new SharedMemory();
    }

    public float[] ProcessRequests(out bool resetRequested) {
        return _memory.ProcessRequests(out resetRequested);
    }

}
