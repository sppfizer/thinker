namespace PRM.Core.Models;

/// <summary>Ball = a token in flight through the diamond grid.</summary>
public class Ball
{
    public int   TokenId         { get; init; }   // index into Vocabulary
    public int   ContextPosition { get; init; }   // 0..WindowSize-1 (or -1 for prediction ball)
    public float Position        { get; set; }    // x-position in grid (pixels / units)
    public float Velocity        { get; set; }    // horizontal velocity
    public float Mass            { get; init; }   // pre-computed from token frequency
    public int   EntryRow        { get; init; }   // row it was dropped at (always 0)
    public bool  Active          { get; set; } = true;
    public bool  Stuck           { get; set; }
    public int   LastNailCol     { get; set; } = -1; // column touched in latest deflection
    public int   LastNailTIdx    { get; set; } = -1; // token routing index used
    public List<int> ContactNailIds { get; } = [];

    public Ball(int tokenId, float position, float mass, int contextPosition = 0)
    {
        TokenId         = tokenId;
        Position        = position;
        Mass            = mass;
        ContextPosition = contextPosition;
    }
}
