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
            // Widening phase: constant moderate force so every row receives training
            // signal. Previous design faded to 0 at the midpoint, leaving the widest
            // (most-nailed) rows completely untrained — a major convergence killer.
            return delta * 0.4f;
        }
        else
        {
            // Narrowing phase: ramps from 0.4 → 1.0 so the model converges firmly
            // toward the target slot as it approaches the output row.
            float depthFrac = (float)(row - _cfg.WideningRows) / Math.Max(_cfg.NarrowingRows, 1);
            return delta * (0.4f + 0.6f * depthFrac);
        }
    }
}
