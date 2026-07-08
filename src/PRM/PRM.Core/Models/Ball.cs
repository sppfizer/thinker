namespace PRM.Core.Models;

/// <summary>Ball = a token in flight through the diamond grid.</summary>
public class Ball
{
    public int   TokenId  { get; init; }   // index into Vocabulary
    public float Position { get; set; }    // x-position in grid (pixels / units)
    public float Velocity { get; set; }    // horizontal velocity
    public float Mass     { get; init; }   // pre-computed from token frequency
    public int   EntryRow { get; init; }   // row it was dropped at (always 0)
    public bool  Active   { get; set; } = true;

    public Ball(int tokenId, float position, float mass)
    {
        TokenId  = tokenId;
        Position = position;
        Mass     = mass;
    }
}
