using PRM.Core.Models;

namespace PRM.Core.Engine;

/// <summary>
/// Computes the phase-aware magnetic force at any position and row in the diamond.
/// Widening phase: force fans outward (weakens toward midpoint).
/// Narrowing phase: force converges inward (strengthens toward output).
/// </summary>
public class MagnetField
{
    private readonly DiamondConfig _cfg;

    public MagnetField(DiamondConfig cfg) => _cfg = cfg;

    /// <summary>
    /// Returns the magnetic force vector (horizontal push/pull) on a ball at (row, x)
    /// toward target x-position xTarget.
    /// </summary>
    public float Force(int row, float x, float xTarget)
    {
        float delta = xTarget - x;

        if (row <= _cfg.WideningRows)
        {
            // Widening phase: force weakens as depth increases (fans outward)
            float depthFrac = (float)row / _cfg.WideningRows;   // 0 → 1
            float scale     = 1f - depthFrac;                    // 1 at top → 0 at midpoint
            return delta * scale;
        }
        else
        {
            // Narrowing phase: force strengthens as output approaches
            float depthFrac = (float)(row - _cfg.WideningRows) / _cfg.NarrowingRows;  // 0 → 1
            return delta * depthFrac;
        }
    }
}
