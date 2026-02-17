using Godot;
using System;
using System.IO;

public partial class DataManager : Node {
    
    // ……………………………………//"~-,_)/ _/''\....('_,-~-|' . „„--„
    
    // ………………………………-~*")"/'~)||||||/'\'\"/'~-, /'||\:,-~"""'\ .'/' '(, ,-~-, \,
    
    // …………………………….._„-~" |||||||||||||||||||||||'\/'|||||||||||||\\-~"||"~--,/',,--"|//'"|(
    
    // ………………………………'\;;;(/'|||||||||||||||||||||||||||||||||||||||||||||||||||(,-~"|||||||(__"~-|
    
    // …………………………__….'\,/||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||-~-„_)
    
    // ……………………_„„-~/||'\;;;/'|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||("*~-„
    
    // …………………...(;;///|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||'\
    
    // ……………………///||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||)
    
    // ………………….||||||||||||||||||||||||||||||||||||\\\\\|||||||||\\\\||||||||||||||||||///|||||||||||||||||||||||||||(
    
    // ……………….,/"'|||||||||||||||||||||||||||||||/' . . . . . . .\\\\\\\||||////// . . . . \\///// \\||||||||||) ,
    
    // ……………….|||||||||||||||||||||||||||||||||/' . . . . . . . . . . . . . . . . . . . . . . . . . '\\||||||/ /)
    
    // ……………..._)||||||||||||||||||||||||||||/' . . . . . . . . . . . . . . . . . . . . . . . . . . . .'\|||||||/
    
    // ……………,/'|||||||||||||||||||||||||||||||| . . . . . . . . . . . . . . . . . . . . . . . . . . . . . '\|||||(;;;;)
    
    // ……………(|||||||||||||||||||||||||||||||| . . . . . . . . . . . . ,, . . .~--- . .__ .-~~ ~ . . . '|||||/""
    
    // …………….."\||||||||||||||||||||||||||||\\;;; . . . . . . .-~' . . . . . . . . . . . . . . . . . . . . \|||(,/'
    
    // ………………..||||||||||||||||||||||||||||||// . . . . . . . . . . . . . . . . . . . . . . . . . . „„„„\\\|||||)
    
    // ………………..||||||||||||||||||||||||||||/ . . . . . . . _„„„„„_______ . . . . . . . .:;;||||///~"'||/
    
    // ………………'\|||//||||||||||||||||||||:: . . . . . . .;;/'||||||||||||||||||||||//; . . . . . . /'""__ . . .'|
    
    // ………………../' :/""~|||||||||||| . . . . . . :::: . . .__„„„„„__:::;;;; . . . . ;;-~*|*~-„ . /
    
    // ………………..|.:|: . .:'\|||||||| . . . . . . . _ „-~"-~~*"¯¯ ;;::::: . . . . .;;'*~'~*"';; '||
    
    // ………………..|.:|:. ./// ||||||| . . . . . . . . . . .' -- -- - ~' ::::::: . . . . .|: .' ~ . .:::;|
    
    // ………………..'|:'|: ::|\ '|||||| . . . . . . . . . . . . . . . . . . .::: . . . . . .'| . . . . . . .|
    
    // ………………...'\:'\::: .¯|||||||| .: : : : : :: . . . . . . . . . . . :: . . . . . . . . . . . . . |
    
    // …………………..'\:: . . '|||||||||| : : : :: :: : :: . . . . . . . .::;;:: .: : . . .| . . . . . |
    
    // …………………….'\,_ .||||||||||| : : : : : : : : : : : . . . . .( . . .___ . . _ |:: . . .|'
    
    // ……………………….¯¯¯'\||||||: : : : : : : : : : : . . . . . . ."~"""""~-~"~' :: . ./'           hes and importer exporter guy 
    
    // ……………………………/'| : : : : : : : : : : : : : . . . . . . / : : : ;;; : : : : : :'|
    
    // …………………………,/';;'|:'\ : : : : : : : : : : : : . . . . . .. . . ;;;;;::: : : : : '|
    
    // ………………………,/';;;;;;'|::'~, : : : : : : : : : : : ;;_„„„___„„„___„„„„---; : :'|a
    
    // ………………….,-~*;;;;;;;;;|::::::'~, : : : : : : : : :;; : : '~,_ . . . . . . ,/' : :,/''*~-„_g
    
    // …………_„„„--~";;;;;;;;;;;;;;;|:::::::::: ' , : : : : : : : :.. . . . .¯¯""""""¯ : : /';;;;;;;;;;¯"~-,a
    
    // …_„„-~"¯;;;;;;;;;;;;;;;;;;;;;;;;;'\:::::::::::::;;'\: : : : : : : : : : : : . . . . . : : /';;;;;;;;;;;;;;;;;;;;"~-,_
    
    // *¯;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;'\::::::: : : :::::'~-, : : : : . . . . . . . . . : :|;;;;;;;;;;;;;;;;;;;;;;;;;;;;¯"~-,
    
    // ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;'\::: : : : : : ::: : :'~-„__ . . . . . . . _„~';;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;'*~-,
    
    // ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;'\::::::::: : : : :::: . . .¯¯¯¯¯¯¯¯¯/;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

    SharedMemory memory;
    
    // godgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgod!
    [Export] Node players;
    [Export] RigidBody3D ball;

    public void DoTheThing(int scoreA, int scoreB) {
        // the scores can be included maybe idk will have to fist myself before i know
        Export(TeamPositions(1), TeamPositions(2), new Vector2(ball.Position.X, ball.Position.Z));
    }
    
    float[] TeamPositions(int team) {
        int teamSize = players.GetChildCount() / 2;
        float[] teamArray = new float[teamSize*2];
        
        for (int i = 0; i < players.GetChildCount(); i+=2) {
            Player player = (Player)(players.GetChild(i)); // yeah bro because casting would totally save us from an error bro
            if (player.team == team) {
                teamArray[i] = player.Position.X;
                teamArray[i+1] = player.Position.Z;
            }
        }
        
        return teamArray;
    }

    unsafe void Export(float[] team1Positions, float[] team2Positions, Vector2 ballPosition) {
        float[] teamPositions = new float[team1Positions.Length + team2Positions.Length];
        Array.Copy(team1Positions, 0, teamPositions, 0, team1Positions.Length);
        Array.Copy(team2Positions, 0, teamPositions, team1Positions.Length, team2Positions.Length);
        
        float[] ballPositionArray = {ballPosition.X, ballPosition.Y};
        float[] positionArray = new float[ballPositionArray.Length + teamPositions.Length];
        Array.Copy(teamPositions, 0, positionArray, 0, teamPositions.Length);
        Array.Copy(ballPositionArray, 0, positionArray, teamPositions.Length, ballPositionArray.Length);
        
        memory.SendStepResults(positionArray, 0,  true);
    }

    public override void _EnterTree() {
        memory = new SharedMemory();
    }
}
