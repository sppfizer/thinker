using PRM.Core.Models;
using System.Runtime.CompilerServices;

namespace PRM.Core.Engine;

/// <summary>
/// Simulates one forward pass of balls through the staggered diamond grid.
///
/// Nail grid layout (hexagonal stagger):
///   Even rows: nail base positions at x = gridLeft + 0, gridLeft + 2, gridLeft + 4, ...
///   Odd  rows: nail base positions at x = gridLeft + 1, gridLeft + 3, gridLeft + 5, ...
///   (offset = NailSpacing / 2 = 1.0 for the default NailSpacing = 2.0)
///
///   Each nail's ACTUAL centre = (base_x + tokenOffX, base_y + tokenOffY)
///   where (tokenOffX, tokenOffY) is a per-token 2D vector inside the unit circle.
///   At max offset (|v|=1) a nail just touches its nearest neighbour. ✓
///
/// Deflection model:
///   Horizontal:  Δx  = offX * α   / mass   (clamped to ±maxStepX per row)
///   Momentum:    Δvx = offY * αY  * radius / mass   (multi-row horizontal nudge)
///
/// Training:
///   Nail 2D offsets are updated via the magnetic force toward the target slot.
///   After each update the vector is projected back onto the unit circle.
/// </summary>
public class BallSimulator
{
    private readonly DiamondConfig _cfg;
    private readonly Nail[,]       _nails;      // shared physical props (radius, resistance)
    private readonly float[,,]     _tokenOffX;  // [row, col, pos*vocabSize+tokenId] — horizontal offset
    private readonly float[,,]     _tokenOffY;  // [row, col, pos*vocabSize+tokenId] — vertical-to-momentum
    private readonly int[]         _rowCols;
    private readonly MagnetField?  _magnet;
    private readonly int           _vocabSize;
    private readonly int           _windowSize;

    // ── Constructor ───────────────────────────────────────────────────────────

    public BallSimulator(DiamondConfig cfg, Nail[,] nails, MagnetField? magnet = null, int vocabSize = 1)
    {
        _cfg        = cfg;
        _nails      = nails;
        _magnet     = magnet;
        _vocabSize  = vocabSize;
        _windowSize = Math.Max(cfg.InputWindowSize, 1);

        int maxCols = nails.GetLength(1);
        _rowCols = new int[cfg.TotalRows];
        for (int r = 0; r < cfg.TotalRows; r++)
            _rowCols[r] = (int)(GridWidth(r) / cfg.NailSpacing) + 2; // +2 for stagger headroom

        // Position-aware routing: each context position has its own routing table.
        // Extra slot (index vocabSize per position) for the prediction ball (tokenId = -1).
        // Dimension 3 = windowSize * (vocabSize + 1)
        int keys = _windowSize * (vocabSize + 1);
        _tokenOffX = new float[cfg.TotalRows, maxCols, keys];
        _tokenOffY = new float[cfg.TotalRows, maxCols, keys];

        // Small random initial offsets to break symmetry (within ±0.05 each axis)
        var rng = new Random(7);
        for (int r = 0; r < cfg.TotalRows; r++)
        for (int c = 0; c < maxCols; c++)
        for (int t = 0; t < keys; t++)
        {
            float ox = (float)(rng.NextDouble() * 0.1 - 0.05);
            float oy = (float)(rng.NextDouble() * 0.1 - 0.05);
            // Ensure unit-circle from the start
            ProjectUnitCircle(ref ox, ref oy);
            _tokenOffX[r, c, t] = ox;
            _tokenOffY[r, c, t] = oy;
        }
    }

    // ── Public API ────────────────────────────────────────────────────────────

    public List<Ball> Simulate(List<Ball> balls, float targetSlotCentre = 0f, float learningRate = 0f)
    {
        var active = new List<Ball>(balls);

        for (int row = 0; row < _cfg.TotalRows && active.Count > 0; row++)
        {
            float left  = LeftBorder(row);
            float right = RightBorder(row);

            // 1. Per-token 2D nail deflection (position + velocity)
            foreach (var ball in active)
                ApplyNailDeflection(ball, row);

            // 2. Ball–ball gravity + elastic collision
            if (_cfg.GravityG > 0f || _cfg.CollisionRadius > 0f)
                ApplyBallInteractions(active);

            // 3. Training: update per-token 2D nail offsets toward target
            if (learningRate > 0f && _magnet != null)
                ApplyNailUpdates(active, row, targetSlotCentre, learningRate);

            // 4. Integrate velocity → position; clamp velocity to prevent explosion
            const float MaxVel = 5f;
            foreach (var ball in active)
            {
                if (float.IsNaN(ball.Velocity) || float.IsInfinity(ball.Velocity)) ball.Velocity = 0f;
                ball.Velocity  = Math.Clamp(ball.Velocity, -MaxVel, MaxVel);
                ball.Position += ball.Velocity * _cfg.DeltaTime;
                if (float.IsNaN(ball.Position) || float.IsInfinity(ball.Position))
                    ball.Position = (left + right) / 2f;
            }

            // 5. Thinking phase: bounce back inside. Summarizing phase: drop off.
            ResolveBounds(active, row, left, right);
        }

        return active;
    }

    /// <summary>
    /// Like Simulate but records ball positions at every row for visualisation.
    /// Returns a GridTrace with geometry + trajectory data; does NOT apply training.
    /// </summary>
    public GridTrace SimulateWithTrace(List<Ball> inputBalls)
    {
        int totalRows = _cfg.TotalRows;

        var gridLefts      = new float[totalRows];
        var gridRights     = new float[totalRows];
        var nailBaseXsList = new float[totalRows][];
        var nailOffXsList  = new float[totalRows][];
        var nailOffYsList  = new float[totalRows][];
        var nailRadiiList  = new float[totalRows][];
        var nailResistList = new float[totalRows][];
        var rowNailCounts  = new int[totalRows];
        var rowFrames      = new PRM.Core.Models.BallFrame[totalRows + 1][];

        // Deep-copy input balls so the original state is unchanged
        var active = inputBalls.Select(b =>
            new Ball(b.TokenId, b.Position, b.Mass, b.ContextPosition)).ToList();

        for (int row = 0; row < totalRows; row++)
        {
            gridLefts[row]  = LeftBorder(row);
            gridRights[row] = RightBorder(row);
            int nailCount   = _rowCols[row];
            rowNailCounts[row] = nailCount;

            // Snapshot BEFORE deflection (entry position for this row)
            rowFrames[row] = active.Select(b =>
                new PRM.Core.Models.BallFrame(b.TokenId, b.Position, b.Velocity, b.Mass, b.ContextPosition)).ToArray();

            // Record nail base positions + averaged offset + physical properties
            var baseXs  = new float[nailCount];
            var offXs   = new float[nailCount];
            var offYs   = new float[nailCount];
            var radii   = new float[nailCount];
            var resists = new float[nailCount];
            for (int c = 0; c < nailCount; c++)
            {
                baseXs[c] = NailBaseX(row, c);
                int rc = Math.Min(row, _nails.GetLength(0) - 1);
                int cc = Math.Min(c,   _nails.GetLength(1) - 1);
                radii[c]   = _nails[rc, cc].Radius;
                resists[c] = _nails[rc, cc].Resistance;
                if (active.Count > 0)
                {
                    foreach (var b in active)
                    {
                        int tIdx = TokenIndex(b);
                        if (tIdx < _tokenOffX.GetLength(2))
                        {
                            offXs[c] += _tokenOffX[row, c, tIdx];
                            offYs[c] += _tokenOffY[row, c, tIdx];
                        }
                    }
                    offXs[c] /= active.Count;
                    offYs[c] /= active.Count;
                }
            }
            nailBaseXsList[row] = baseXs;
            nailOffXsList[row]  = offXs;
            nailOffYsList[row]  = offYs;
            nailRadiiList[row]  = radii;
            nailResistList[row] = resists;

            // Apply physics (no training)
            foreach (var ball in active) ApplyNailDeflection(ball, row);
            if (_cfg.GravityG > 0f || _cfg.CollisionRadius > 0f) ApplyBallInteractions(active);

            const float MaxVel = 5f;
            float left  = LeftBorder(row);
            float right = RightBorder(row);
            foreach (var ball in active)
            {
                if (float.IsNaN(ball.Velocity) || float.IsInfinity(ball.Velocity)) ball.Velocity = 0f;
                ball.Velocity  = Math.Clamp(ball.Velocity, -MaxVel, MaxVel);
                ball.Position += ball.Velocity * _cfg.DeltaTime;
                if (float.IsNaN(ball.Position) || float.IsInfinity(ball.Position))
                    ball.Position = (left + right) / 2f;
            }
            ResolveBounds(active, row, left, right);
        }

        // Final snapshot: positions after all rows (output positions)
        rowFrames[totalRows] = active.Select(b =>
            new PRM.Core.Models.BallFrame(b.TokenId, b.Position, b.Velocity, b.Mass, b.ContextPosition)).ToArray();

        return new PRM.Core.Models.GridTrace
        {
            TotalRows      = totalRows,
            WideningRows   = _cfg.WideningRows,
            MaxWidth       = _cfg.MaxWidth,
            GridLefts      = gridLefts,
            GridRights     = gridRights,
            NailBaseXs     = nailBaseXsList,
            NailOffXs      = nailOffXsList,
            NailOffYs      = nailOffYsList,
            NailRadii      = nailRadiiList,
            NailResistances= nailResistList,
            RowNailCounts  = rowNailCounts,
            RowFrames      = rowFrames,
        };
    }

    /// <summary>
    /// Post-fallthrough reinforcement: use the successful balls with the fewest bumps
    /// to stiffen the nails they contacted during this sample.
    /// </summary>
    public int ReinforceContacts(IEnumerable<Ball> balls, float targetCentre, float learningRate)
    {
        var winners = balls
            .Where(b => b.TokenId >= 0 && !b.Stuck && Math.Abs(b.Position - targetCentre) <= Math.Max(_cfg.NailSpacing * 0.6f, 0.5f))
            .ToList();

        if (winners.Count == 0) return 0;

        int minContacts = winners.Min(b => b.ContactNailIds.Count);
        var focus = winners.Where(b => b.ContactNailIds.Count == minContacts).ToList();
        var uniqueNails = focus.SelectMany(b => b.ContactNailIds).Distinct().ToArray();

        float massFactor = focus.Count > 0 ? focus.Average(b => MathF.Sqrt(Math.Max(b.Mass, 0.01f))) : 1f;
        foreach (int nailId in uniqueNails)
        {
            int row = nailId / 10_000;
            int col = nailId % 10_000;
            if (row < 0 || row >= _nails.GetLength(0) || col < 0 || col >= _nails.GetLength(1)) continue;

            var nail = _nails[row, col];
            float boost = learningRate * massFactor * 0.05f;
            nail.Resistance = Math.Clamp(nail.Resistance + boost, 0.05f, 2.5f);
            nail.Density    = Math.Clamp(nail.Density + boost * 0.5f, 0.1f, 4.0f);
            _nails[row, col] = nail;
        }

        return uniqueNails.Length;
    }

    // ── Grid geometry ─────────────────────────────────────────────────────────

    public float GridWidth(int row)
    {
        float contractionRate = (_cfg.MaxWidth - _cfg.EntryWidth) / _cfg.NarrowingRows;
        return row <= _cfg.WideningRows
            ? _cfg.EntryWidth + row * (_cfg.MaxWidth - _cfg.EntryWidth) / _cfg.WideningRows
            : _cfg.MaxWidth   - (row - _cfg.WideningRows) * contractionRate;
    }

    public float LeftBorder(int row)  => (_cfg.MaxWidth - GridWidth(row)) / 2f;
    public float RightBorder(int row) => LeftBorder(row) + GridWidth(row);

    /// <summary>
    /// X base position of nail at (row, col) in the staggered grid.
    ///   Even rows: base_x = gridLeft + col * NailSpacing
    ///   Odd  rows: base_x = gridLeft + col * NailSpacing + NailSpacing/2
    /// </summary>
    public float NailBaseX(int row, int col)
    {
        float stagger = (row % 2 == 1) ? _cfg.NailSpacing / 2f : 0f;
        return LeftBorder(row) + stagger + col * _cfg.NailSpacing;
    }

    // ── Offset persistence ────────────────────────────────────────────────────

    public float[,,] GetTokenOffX() => _tokenOffX;
    public float[,,] GetTokenOffY() => _tokenOffY;

    public void SetTokenOffsets(float[,,] srcX, float[,,] srcY)
    {
        CopyInto(srcX, _tokenOffX);
        CopyInto(srcY, _tokenOffY);
    }

    // ── Physics ───────────────────────────────────────────────────────────────

    private void ApplyNailDeflection(Ball ball, int row)
    {
        int col = NailColumn(ball.Position, row);
        if (col < 0 || col >= _rowCols[row]) return;

        int tIdx = TokenIndex(ball);
        int nailId = NailKey(row, col);
        float offX   = _tokenOffX[row, col, tIdx];
        float offY   = _tokenOffY[row, col, tIdx];
        float radius = _nails[row, col].Radius;

        if (ball.ContactNailIds.Count == 0 || ball.ContactNailIds[^1] != nailId)
            ball.ContactNailIds.Add(nailId);

        // Deflection routing can optionally weight by inverse mass.
        // 0 = flat, 0.5 = sqrt-IDF, 1 = inverse-mass.
        float idf = MathF.Pow(1f / Math.Max(ball.Mass, 0.01f), _cfg.DeflectionIdfPower);

        // Horizontal position change (width-normalised to prevent overshooting)
        float rowWidth  = Math.Max(GridWidth(row), 1f);
        float maxStepX  = rowWidth / _cfg.TotalRows * _cfg.DeflectionAlpha;
        float rawStepX  = offX * _cfg.DeflectionAlpha * idf;
        ball.Position  += Math.Clamp(rawStepX, -maxStepX, maxStepX);

        // Horizontal momentum nudge from the Y-component of the 2D offset
        ball.Velocity  += offY * _cfg.DeflectionAlphaY * radius * idf;
    }

    private void ApplyBallInteractions(List<Ball> balls)
    {
        for (int i = 0; i < balls.Count; i++)
        for (int j = i + 1; j < balls.Count; j++)
        {
            float d = Math.Abs(balls[i].Position - balls[j].Position);
            if (d > _cfg.ProximityBand) continue;

            if (_cfg.GravityG > 0f)
            {
                float g   = _cfg.GravityG * balls[i].Mass * balls[j].Mass / (d * d + 1e-6f);
                float dir = Math.Sign(balls[j].Position - balls[i].Position);
                balls[i].Velocity += g * dir  * _cfg.DeltaTime;
                balls[j].Velocity -= g * dir  * _cfg.DeltaTime;
            }

            if (_cfg.CollisionRadius > 0f && d < _cfg.CollisionRadius)
            {
                float mi = balls[i].Mass, mj = balls[j].Mass;
                float vi = balls[i].Velocity, vj = balls[j].Velocity;
                balls[i].Velocity = ((mi - mj) * vi + 2f * mj * vj) / (mi + mj);
                balls[j].Velocity = ((mj - mi) * vj + 2f * mi * vi) / (mi + mj);
            }
        }
    }

    private void ApplyNailUpdates(List<Ball> balls, int row, float targetCentre, float lr)
    {
        foreach (var ball in balls)
        {
            if (ball.TokenId < 0) continue;   // skip neutral probe ball — no semantics to train

            int col = NailColumn(ball.Position, row);
            if (col < 0 || col >= _rowCols[row]) continue;

            int tIdx = TokenIndex(ball);
            var nail = _nails[row, col];
            float resistance = nail.Resistance;
            float density = Math.Max(nail.Density, 0.1f);
            float radius = nail.Radius;

            // Magnet force along x-axis toward targetCentre
            float forceX = _magnet!.Force(row, ball.Position, targetCentre);

            // Normalise force by row width so the update magnitude is stable
            float rowWidth   = Math.Max(GridWidth(row), 1f);
            float normForceX = forceX / rowWidth;

            // Small y-axis component: helps build momentum toward target
            float normForceY = normForceX * 0.25f;

            // Punish larger angular faults more heavily so the next identical sample
            // moves the nail in a way that reduces the same error.
            float idealX = normForceX;
            float idealY = normForceY;
            float currentX = _tokenOffX[row, col, tIdx];
            float currentY = _tokenOffY[row, col, tIdx];
            float currentLen = MathF.Max(MathF.Sqrt(currentX * currentX + currentY * currentY), 1e-6f);
            float idealLen   = MathF.Max(MathF.Sqrt(idealX * idealX + idealY * idealY), 1e-6f);
            float dot = (currentX * idealX + currentY * idealY) / (currentLen * idealLen);
            float anglePenalty = 1f + (1f - Math.Clamp(dot, -1f, 1f));

            // Weight and nail inertia both influence how far the nail can move.
            float massFactor = MathF.Sqrt(Math.Max(ball.Mass, 0.01f));
            float inertia    = Math.Max(resistance * density * (1f + radius), 0.05f);
            float deltaX = lr * massFactor * anglePenalty * normForceX / inertia;
            float deltaY = lr * massFactor * anglePenalty * normForceY / inertia;

            float newX = currentX + deltaX;
            float newY = currentY + deltaY;

            // Project back onto the unit circle (the key constraint)
            ProjectUnitCircle(ref newX, ref newY);

            _tokenOffX[row, col, tIdx] = newX;
            _tokenOffY[row, col, tIdx] = newY;

            // Strengthen the nail slightly when the current sample points to it.
            nail.Resistance = Math.Clamp(nail.Resistance + lr * massFactor * 0.01f, 0.05f, 2.0f);
            nail.Density    = Math.Clamp(nail.Density + lr * massFactor * 0.005f, 0.1f, 3.0f);
            _nails[row, col] = nail;
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// <summary>
    /// Find the column index of the nail nearest to x in the staggered row.
    ///   Even rows: col = (x - gridLeft)            / NailSpacing
    ///   Odd  rows: col = (x - gridLeft - spacing/2) / NailSpacing
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private int NailColumn(float x, int row)
    {
        if (float.IsNaN(x) || float.IsInfinity(x)) return -1;
        float stagger = (row % 2 == 1) ? _cfg.NailSpacing / 2f : 0f;
        return (int)((x - LeftBorder(row) - stagger) / _cfg.NailSpacing);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private int TokenIndex(Ball ball)
    {
        // Position-aware routing key: pos × (vocabSize+1) + tokenId
        // Each context window position has its own independent routing table.
        int pos  = Math.Clamp(ball.ContextPosition, 0, _windowSize - 1);
        int slot = (ball.TokenId >= 0 && ball.TokenId < _vocabSize) ? ball.TokenId : _vocabSize;
        return pos * (_vocabSize + 1) + slot;
    }

    /// <summary>
    /// Project (x, y) onto the unit circle: clamp magnitude to ≤ 1.
    /// Vectors already inside the circle are unchanged.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ProjectUnitCircle(ref float x, ref float y)
    {
        float mag = MathF.Sqrt(x * x + y * y);
        if (mag > 1f) { x /= mag; y /= mag; }
    }

    private static void CopyInto(float[,,] src, float[,,] dst)
    {
        int r0 = Math.Min(src.GetLength(0), dst.GetLength(0));
        int c0 = Math.Min(src.GetLength(1), dst.GetLength(1));
        int t0 = Math.Min(src.GetLength(2), dst.GetLength(2));
        for (int r = 0; r < r0; r++)
        for (int c = 0; c < c0; c++)
        for (int t = 0; t < t0; t++)
            dst[r, c, t] = src[r, c, t];
    }

    private void ResolveBounds(List<Ball> balls, int row, float left, float right)
    {
        float jamThreshold = Math.Max(_cfg.NailSpacing * 0.18f, 0.15f);
        float velThreshold = 0.04f;

        foreach (var ball in balls)
        {
            if (ball.Position < left)
            {
                if (row < _cfg.WideningRows)
                {
                    ball.Position = left + (left - ball.Position) * 0.35f;
                    ball.Velocity = Math.Abs(ball.Velocity) * 0.55f;
                }
                else
                {
                    ball.Active = false;
                }
            }
            else if (ball.Position > right)
            {
                if (row < _cfg.WideningRows)
                {
                    ball.Position = right - (ball.Position - right) * 0.35f;
                    ball.Velocity = -Math.Abs(ball.Velocity) * 0.55f;
                }
                else
                {
                    ball.Active = false;
                }
            }

            if (ball.Active && IsStuck(ball, row, jamThreshold, velThreshold))
            {
                ball.Stuck = true;
                ball.Active = false;
            }
        }

        balls.RemoveAll(b => !b.Active);
    }

    private bool IsStuck(Ball ball, int row, float jamThreshold, float velThreshold)
    {
        if (row < _cfg.WideningRows) return false;
        if (Math.Abs(ball.Velocity) > velThreshold) return false;

        int col = NailColumn(ball.Position, row);
        if (col < 0 || col >= _rowCols[row]) return false;

        float nailX = NailBaseX(row, col);
        float dist = Math.Abs(ball.Position - nailX);
        return dist <= jamThreshold;
    }

    private static int NailKey(int row, int col) => row * 10_000 + col;
}
