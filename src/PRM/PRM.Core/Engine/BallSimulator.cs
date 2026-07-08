using PRM.Core.Models;
using System.Runtime.CompilerServices;

namespace PRM.Core.Engine;

/// <summary>
/// Simulates one forward pass of balls through the diamond grid.
/// Handles: nail deflection, ball-ball gravity, elastic collision, open-border removal.
/// In training mode also applies nail updates via the magnetic force.
/// </summary>
public class BallSimulator
{
    private readonly DiamondConfig _cfg;
    private readonly Nail[,]       _nails;   // [row, column]
    private readonly int[]         _rowCols; // number of nail columns per row
    private readonly MagnetField?  _magnet;

    public BallSimulator(DiamondConfig cfg, Nail[,] nails, MagnetField? magnet = null)
    {
        _cfg    = cfg;
        _nails  = nails;
        _magnet = magnet;
        _rowCols = new int[cfg.TotalRows];
        for (int r = 0; r < cfg.TotalRows; r++)
            _rowCols[r] = (int)(GridWidth(r) / cfg.NailSpacing) + 1;
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// <summary>
    /// Run a full forward pass.  Returns surviving balls after all rows.
    /// If learningRate > 0 and magnet is set, nails are updated in place.
    /// </summary>
    public List<Ball> Simulate(List<Ball> balls, float targetSlotCentre = 0f, float learningRate = 0f)
    {
        var active = new List<Ball>(balls);

        for (int row = 0; row < _cfg.TotalRows && active.Count > 0; row++)
        {
            float left  = LeftBorder(row);
            float right = RightBorder(row);

            // 1. Nail deflection
            foreach (var ball in active)
                ApplyNailDeflection(ball, row);

            // 2. Ball–ball gravity + collision (within proximity band)
            ApplyBallInteractions(active);

            // 3. Training: nail update via magnetic force
            if (learningRate > 0f && _magnet != null)
                ApplyNailUpdates(active, row, targetSlotCentre, learningRate);

            // 4. Integrate velocity
            foreach (var ball in active)
                ball.Position += ball.Velocity * _cfg.DeltaTime;

            // 5. Remove out-of-bounds balls
            active.RemoveAll(b => b.Position < left || b.Position > right);
        }

        return active;
    }

    // ── Grid geometry ─────────────────────────────────────────────────────────

    public float GridWidth(int row)
    {
        float expansionRate   = (row <= _cfg.WideningRows)
            ? (_cfg.MaxWidth  - _cfg.EntryWidth) / _cfg.WideningRows
            : 0f;
        float contractionRate = (_cfg.MaxWidth - _cfg.EntryWidth) / _cfg.NarrowingRows;

        return row <= _cfg.WideningRows
            ? _cfg.EntryWidth + row * expansionRate
            : _cfg.MaxWidth   - (row - _cfg.WideningRows) * contractionRate;
    }

    public float LeftBorder(int row)  => (_cfg.MaxWidth - GridWidth(row)) / 2f;
    public float RightBorder(int row) => LeftBorder(row) + GridWidth(row);

    // ── Physics ───────────────────────────────────────────────────────────────

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void ApplyNailDeflection(Ball ball, int row)
    {
        int col = NailColumn(ball.Position, row);
        if (col < 0 || col >= _rowCols[row]) return;

        float deflect = _nails[row, col].Tilt * _cfg.DeflectionAlpha / Math.Max(ball.Mass, 0.01f);
        ball.Position += deflect;
    }

    private void ApplyBallInteractions(List<Ball> balls)
    {
        for (int i = 0; i < balls.Count; i++)
        for (int j = i + 1; j < balls.Count; j++)
        {
            float d = Math.Abs(balls[i].Position - balls[j].Position);
            if (d > _cfg.ProximityBand) continue;

            // Gravity
            float g = _cfg.GravityG * balls[i].Mass * balls[j].Mass / (d * d + 1e-6f);
            float dir = Math.Sign(balls[j].Position - balls[i].Position);
            balls[i].Velocity += g * dir  * _cfg.DeltaTime;
            balls[j].Velocity -= g * dir  * _cfg.DeltaTime;

            // Elastic collision
            if (d < _cfg.CollisionRadius)
            {
                float mi = balls[i].Mass, mj = balls[j].Mass, vi = balls[i].Velocity, vj = balls[j].Velocity;
                balls[i].Velocity = ((mi - mj) * vi + 2f * mj * vj) / (mi + mj);
                balls[j].Velocity = ((mj - mi) * vj + 2f * mi * vi) / (mi + mj);
            }
        }
    }

    private void ApplyNailUpdates(List<Ball> balls, int row, float targetCentre, float lr)
    {
        foreach (var ball in balls)
        {
            int col = NailColumn(ball.Position, row);
            if (col < 0 || col >= _rowCols[row]) continue;

            float force = _magnet!.Force(row, ball.Position, targetCentre);
            float delta = lr * ball.Mass * force / Math.Max(_nails[row, col].Diameter, 0.01f);

            // Clamp tilt to [-1, 1]
            _nails[row, col].Tilt = Math.Clamp(_nails[row, col].Tilt + delta, -1f, 1f);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private int NailColumn(float x, int row)
        => (int)((x - LeftBorder(row)) / _cfg.NailSpacing);
}
