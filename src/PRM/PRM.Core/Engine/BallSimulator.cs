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
    private readonly float[,,]     _sharedOffX; // [row, col, tokenId] — cross-position routing prior
    private readonly float[,,]     _sharedOffY; // [row, col, tokenId] — cross-position routing prior
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
        _sharedOffX = new float[cfg.TotalRows, maxCols, vocabSize + 1];
        _sharedOffY = new float[cfg.TotalRows, maxCols, vocabSize + 1];

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

        for (int r = 0; r < cfg.TotalRows; r++)
        for (int c = 0; c < maxCols; c++)
        for (int t = 0; t < vocabSize + 1; t++)
        {
            float ox = (float)(rng.NextDouble() * 0.1 - 0.05);
            float oy = (float)(rng.NextDouble() * 0.1 - 0.05);
            ProjectUnitCircle(ref ox, ref oy);
            _sharedOffX[r, c, t] = ox;
            _sharedOffY[r, c, t] = oy;
        }
    }

    // ── Public API ────────────────────────────────────────────────────────────

    public List<Ball> Simulate(List<Ball> balls, float targetSlotCentre = 0f, float learningRate = 0f)
    {
        var active = new List<Ball>(balls);
        bool summaryCreated = false;

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

            AddContextSummaryBalls(active, balls, row, ref summaryCreated);

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

            // 5. Thinking phase: drop off. Summarizing phase: bounce back inside.
            _ = ResolveBounds(active, row, left, right);
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
        var rowEvents      = new List<PRM.Core.Models.BallEvent>[totalRows];

        // Deep-copy input balls so the original state is unchanged
        var active = inputBalls.Select(b =>
            new Ball(b.TokenId, b.Position, b.Mass, b.ContextPosition, b.RelevanceWeight)).ToList();
        bool summaryCreated = false;

        for (int row = 0; row < totalRows; row++)
        {
            rowEvents[row] = [];
            gridLefts[row]  = LeftBorder(row);
            gridRights[row] = RightBorder(row);
            int nailCount   = _rowCols[row];
            rowNailCounts[row] = nailCount;

            // Snapshot BEFORE deflection (entry position for this row)
            rowFrames[row] = active.Select(b =>
                new PRM.Core.Models.BallFrame(b.TokenId, b.Position, b.Velocity, b.Mass, b.ContextPosition)).ToArray();

            // Record nail base positions + dominant token offset + physical properties
            var baseXs  = new float[nailCount];
            var offXs   = new float[nailCount];
            var offYs   = new float[nailCount];
            var radii   = new float[nailCount];
            var resists = new float[nailCount];
            float maxMass = 0f;
            int dominantTIdx = -1;
            foreach (var b in active)
            {
                int tIdx = TokenIndex(b);
                if (tIdx >= 0 && tIdx < _tokenOffX.GetLength(2) && b.Mass > maxMass)
                {
                    maxMass = b.Mass;
                    dominantTIdx = tIdx;
                }
            }
            for (int c = 0; c < nailCount; c++)
            {
                baseXs[c] = NailBaseX(row, c);
                int rc = Math.Min(row, _nails.GetLength(0) - 1);
                int cc = Math.Min(c,   _nails.GetLength(1) - 1);
                radii[c]   = _nails[rc, cc].Radius;
                resists[c] = _nails[rc, cc].Resistance;
                if (dominantTIdx >= 0 && dominantTIdx < _tokenOffX.GetLength(2))
                {
                    int slot = dominantTIdx % (_vocabSize + 1);
                    (offXs[c], offYs[c]) = EffectiveOffset(row, c, dominantTIdx, slot);
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
            AddContextSummaryBalls(active, null, row, ref summaryCreated);

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
            rowEvents[row].AddRange(ResolveBounds(active, row, left, right));
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
            RowEvents      = rowEvents.Select(x => x.ToArray()).ToArray(),
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

    /// <summary>
    /// Wrong-prediction softening: reduce stiffness on contacted nails so they
    /// remain plastic and can be corrected in future samples.
    /// Mirror of ReinforceContacts, called only on miss.
    /// </summary>
    public void SoftenContacts(IEnumerable<Ball> balls, float learningRate)
    {
        // Deduplicate so a nail shared by multiple balls is only softened once per miss
        var uniqueNails = balls
            .Where(b => b.TokenId >= 0 && !b.Stuck)
            .SelectMany(b => b.ContactNailIds)
            .Distinct();

        foreach (int nailId in uniqueNails)
        {
            int row = nailId / 10_000;
            int col = nailId % 10_000;
            if (row < 0 || row >= _nails.GetLength(0) || col < 0 || col >= _nails.GetLength(1)) continue;
            var nail = _nails[row, col];
            float drop = learningRate * 0.05f;
            nail.Resistance = Math.Clamp(nail.Resistance - drop,       0.05f, 2.5f);
            nail.Density    = Math.Clamp(nail.Density    - drop * 0.5f, 0.1f, 4.0f);
            _nails[row, col] = nail;
        }
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
    public float[,,] GetSharedOffX() => _sharedOffX;
    public float[,,] GetSharedOffY() => _sharedOffY;

    public void SetTokenOffsets(float[,,] srcX, float[,,] srcY)
    {
        CopyInto(srcX, _tokenOffX);
        CopyInto(srcY, _tokenOffY);
    }

    public void SetSharedOffsets(float[,,] srcX, float[,,] srcY)
    {
        CopyInto(srcX, _sharedOffX);
        CopyInto(srcY, _sharedOffY);
    }

    // ── Physics ───────────────────────────────────────────────────────────────

    private void ApplyNailDeflection(Ball ball, int row)
    {
        ball.LastNailCol  = -1;
        ball.LastNailTIdx = -1;

        int col = NailColumn(ball.Position, row);
        if (col < 0 || col >= _rowCols[row]) return;

        int tIdx = TokenIndex(ball);
        if (tIdx < 0 || tIdx >= _tokenOffX.GetLength(2)) return;
        int slot = TokenSlot(ball);
        int nailId = NailKey(row, col);
        var (offX, offY) = EffectiveOffset(row, col, tIdx, slot);
        float radius = _nails[row, col].Radius;

        if (ball.ContactNailIds.Count == 0 || ball.ContactNailIds[^1] != nailId)
            ball.ContactNailIds.Add(nailId);

        // Deflection routing can optionally weight by inverse mass.
        // 0 = flat, 0.5 = sqrt-IDF, 1 = inverse-mass.
        float idf = MathF.Pow(1f / Math.Max(ball.Mass, 0.01f), _cfg.DeflectionIdfPower);

        // Horizontal position change.
        // maxStepX = per-row budget so a fully-deflected nail (offX=1) moves the ball
        // one TotalRows-th of the row width per step.  rawStepX uses maxStepX as the
        // scale so offX=1 always reaches the budget cap regardless of grid width.
        // Old code used offX*alpha (constant ≈0.6 units) which was far too small for
        // wide grids (simple_corpus: output zone 418 units, max travel = 21 units → 5%).
        float rowWidth  = Math.Max(GridWidth(row), 1f);
        float maxStepX  = rowWidth / _cfg.TotalRows * _cfg.DeflectionAlpha;
        float rawStepX  = offX * maxStepX * idf;
        ball.LastNailCol  = col;
        ball.LastNailTIdx = tIdx;
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

    private void AddContextSummaryBalls(List<Ball> active, List<Ball>? owner, int row, ref bool summaryCreated)
    {
        if (summaryCreated) return;

        int requested = Math.Max(_cfg.ContextSummaryBallCount, 0);
        if (requested <= 0) return;

        int summaryRow = _cfg.ContextSummaryRow < 0 ? _cfg.WideningRows : _cfg.ContextSummaryRow;
        summaryRow = Math.Clamp(summaryRow, 0, _cfg.TotalRows - 1);
        if (row != summaryRow) return;

        float massScale = Math.Max(_cfg.ContextSummaryMassScale, 0f);
        if (massScale <= 0f) return;

        var candidates = active
            .Where(b => b.TokenId >= 0 && b.Active && !b.Stuck)
            .OrderBy(b => b.Position)
            .ToArray();
        if (candidates.Length == 0) return;

        int summaryCount = Math.Min(requested, candidates.Length);
        for (int groupIndex = 0; groupIndex < summaryCount; groupIndex++)
        {
            int start = groupIndex * candidates.Length / summaryCount;
            int end = (groupIndex + 1) * candidates.Length / summaryCount;
            if (end <= start) continue;

            ReadOnlySpan<Ball> group = candidates.AsSpan(start, end - start);
            float weightedMass = 0f;
            float positionSum = 0f;
            float velocitySum = 0f;
            float relevanceSum = 0f;
            float massSum = 0f;
            int contextPosition = 0;

            foreach (var ball in group)
            {
                float relevance = Math.Clamp(ball.RelevanceWeight, 0f, 1f);
                float weight = Math.Max(ball.Mass * relevance, 1e-6f);
                weightedMass += weight;
                positionSum += ball.Position * weight;
                velocitySum += ball.Velocity * weight;
                relevanceSum += relevance;
                massSum += ball.Mass * relevance;
                contextPosition = Math.Max(contextPosition, ball.ContextPosition);
            }

            float position = positionSum / weightedMass;
            float velocity = velocitySum / weightedMass;
            float relevanceWeight = Math.Clamp(relevanceSum / group.Length, 0f, 1f);
            float mass = Math.Max(massSum / group.Length * massScale, 0.001f);
            var summary = new Ball(-2 - groupIndex, position, mass, contextPosition, relevanceWeight)
            {
                Velocity = velocity
            };

            active.Add(summary);
            owner?.Add(summary);
        }

        summaryCreated = true;
    }

    private void ApplyNailUpdates(List<Ball> balls, int row, float targetCentre, float lr)
    {
        foreach (var ball in balls)
        {
            bool isPredictionProbe = ball.TokenId == -1;
            if (ball.TokenId < 0 && !isPredictionProbe) continue;

            float probeTrainingWeight = Math.Max(_cfg.PredictionProbeTrainingWeight, 0f);
            if (isPredictionProbe && probeTrainingWeight <= 0f) continue;

            int col = ball.LastNailCol;
            if (col < 0 || col >= _rowCols[row]) continue;

            int tIdx = ball.LastNailTIdx;
            if (tIdx < 0 || tIdx >= _tokenOffX.GetLength(2)) continue;
            int slot = TokenSlot(ball);
            if (slot < 0 || slot >= _sharedOffX.GetLength(2)) continue;

            // Magnet force along x-axis toward targetCentre
            float forceX = _magnet!.Force(row, ball.Position, targetCentre);

            // Normalise to the same coordinate space as offX: fraction of per-row budget.
            // offX=1 → ball moves maxStepX = rowWidth/TotalRows*alpha per row.
            // idealX=1 → nail should point at maximum deflection toward target.
            // Normalise by rowWidth/TotalRows so idealX saturates to 1 when delta≥rowWidth/TotalRows,
            // giving proportional guidance for small deltas and full-deflection for large ones.
            float rowWidth   = Math.Max(GridWidth(row), 1f);
            float normForceX = forceX * _cfg.TotalRows / rowWidth;

            // Small y-axis component: helps build momentum toward target
            float normForceY = normForceX * 0.25f;

            // Weight and nail inertia both influence how far the nail can move.
            float relevance = Math.Clamp(ball.RelevanceWeight, 0f, 1f);
            float massFactor = isPredictionProbe
                ? probeTrainingWeight
                : MathF.Sqrt(Math.Max(ball.Mass, 0.01f)) * relevance;
            if (massFactor <= 0f) continue;

            // Error-correction: pull current offset toward the ideal (magnet direction).
            // Natural equilibrium at current==ideal; amplitude shrinks as the nail converges.
            // Clamp scale to [0,1] to prevent overshoot when inertia is at its floor.
            float idealX = normForceX;
            float idealY = normForceY;
            float scale  = UpdateScale(row, col, lr, massFactor);
            ApplyOffsetUpdate(row, col, tIdx, slot, idealX, idealY, scale);
            ApplyDownstreamNailUpdates(row, col, tIdx, slot, idealX, idealY, targetCentre, lr, massFactor);
        }
    }

    private void ApplyDownstreamNailUpdates(
        int sourceRow, int sourceCol, int tIdx, int slot,
        float sourceIdealX, float sourceIdealY, float targetCentre, float lr, float massFactor)
    {
        float influence = Math.Clamp(_cfg.DownstreamNailInfluence, 0f, 1f);
        int rows = Math.Min(
            Math.Max(_cfg.DownstreamNailInfluenceRows, 0),
            Math.Max(_cfg.TotalRows - sourceRow - 1, 0));
        float radiusUnits = Math.Max(_cfg.DownstreamNailInfluenceRadius, 0f);
        if (influence <= 0f || rows <= 0 || radiusUnits <= 0f) return;

        float sourceX = NailBaseX(sourceRow, sourceCol);
        float spacing = Math.Max(_cfg.NailSpacing, 1e-6f);
        int colRadius = Math.Max(1, (int)MathF.Ceiling(radiusUnits) + 1);
        float decay = Math.Max(_cfg.DownstreamNailInfluenceDecay, 0f);
        float directionality = Math.Clamp(_cfg.DownstreamNailTargetDirectionality, 0f, 1f);

        for (int dr = 1; dr <= rows; dr++)
        {
            int row = sourceRow + dr;
            if (row >= _cfg.TotalRows) break;

            float pathBlend = directionality * dr / (rows + 1f);
            float pathCentreX = Lerp(sourceX, targetCentre, pathBlend);
            int centreCol = NailColumn(pathCentreX, row);
            int startCol = Math.Max(0, centreCol - colRadius);
            int endCol = Math.Min(_rowCols[row] - 1, centreCol + colRadius);
            if (startCol > endCol) continue;

            float rowWeight = 1f / (1f + decay * (dr - 1));
            for (int col = startCol; col <= endCol; col++)
            {
                float nailX = NailBaseX(row, col);
                float dxUnits = Math.Abs(nailX - pathCentreX) / spacing;
                if (dxUnits > radiusUnits) continue;

                float lateralWeight = 1f - dxUnits / radiusUnits;
                float weight = influence * rowWeight * lateralWeight;
                if (weight <= 0f) continue;

                float rowWidth = Math.Max(GridWidth(row), 1f);
                float localIdealX = _magnet!.Force(row, nailX, targetCentre) * _cfg.TotalRows / rowWidth;
                float localIdealY = localIdealX * 0.25f;
                float idealX = Lerp(sourceIdealX, localIdealX, directionality);
                float idealY = Lerp(sourceIdealY, localIdealY, directionality);
                float scale = UpdateScale(row, col, lr, massFactor) * weight;
                ApplyOffsetUpdate(row, col, tIdx, slot, idealX, idealY, scale);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float Lerp(float a, float b, float t) => a + (b - a) * t;

    private float UpdateScale(int row, int col, float lr, float massFactor)
    {
        var nail = _nails[row, col];
        float density = Math.Max(nail.Density, 0.1f);
        float inertia = Math.Max(nail.Resistance * density * (1f + nail.Radius), 0.05f);
        return Math.Clamp(lr * massFactor / inertia, 0f, 1f);
    }

    private void ApplyOffsetUpdate(
        int row, int col, int tIdx, int slot,
        float idealX, float idealY, float scale)
    {
        scale = Math.Clamp(scale, 0f, 1f);
        if (scale <= 0f) return;

        float currentX = _tokenOffX[row, col, tIdx];
        float currentY = _tokenOffY[row, col, tIdx];
        float sharedX = _sharedOffX[row, col, slot];
        float sharedY = _sharedOffY[row, col, slot];
        float blend  = Math.Clamp(_cfg.SharedOffsetBlend, 0f, 1f);
        float posScale = scale * (1f - blend);
        float sharedScale = scale * blend;

        float newX   = currentX + posScale * (idealX - currentX);
        float newY   = currentY + posScale * (idealY - currentY);
        float newSharedX = sharedX + sharedScale * (idealX - sharedX);
        float newSharedY = sharedY + sharedScale * (idealY - sharedY);
        ProjectUnitCircle(ref newX, ref newY);
        ProjectUnitCircle(ref newSharedX, ref newSharedY);
        _tokenOffX[row, col, tIdx] = newX;
        _tokenOffY[row, col, tIdx] = newY;
        _sharedOffX[row, col, slot] = newSharedX;
        _sharedOffY[row, col, slot] = newSharedY;
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
        int slot = TokenSlot(ball);
        return pos * (_vocabSize + 1) + slot;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private int TokenSlot(Ball ball) =>
        (ball.TokenId >= 0 && ball.TokenId < _vocabSize) ? ball.TokenId : _vocabSize;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private (float x, float y) EffectiveOffset(int row, int col, int tIdx, int tokenSlot)
    {
        float blend = Math.Clamp(_cfg.SharedOffsetBlend, 0f, 1f);
        if (blend <= 0f)
            return (_tokenOffX[row, col, tIdx], _tokenOffY[row, col, tIdx]);

        float posX = _tokenOffX[row, col, tIdx];
        float posY = _tokenOffY[row, col, tIdx];
        float sharedX = _sharedOffX[row, col, tokenSlot];
        float sharedY = _sharedOffY[row, col, tokenSlot];
        return (posX * (1f - blend) + sharedX * blend,
                posY * (1f - blend) + sharedY * blend);
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

    private List<PRM.Core.Models.BallEvent> ResolveBounds(List<Ball> balls, int row, float left, float right)
    {
        float jamThreshold = Math.Max(_cfg.NailSpacing * 0.18f, 0.15f);
        float velThreshold = 0.04f;
        var events = new List<PRM.Core.Models.BallEvent>();

        foreach (var ball in balls)
        {
            if (ball.Position < left)
            {
                if (row <= _cfg.WideningRows)
                {
                    events.Add(new PRM.Core.Models.BallEvent(ball.TokenId, row, ball.Position, "dropped"));
                    ball.Active = false;
                }
                else
                {
                    ball.Position = left + (left - ball.Position) * 0.35f;
                    ball.Velocity = Math.Abs(ball.Velocity) * 0.55f;
                }
            }
            else if (ball.Position > right)
            {
                if (row <= _cfg.WideningRows)
                {
                    events.Add(new PRM.Core.Models.BallEvent(ball.TokenId, row, ball.Position, "dropped"));
                    ball.Active = false;
                }
                else
                {
                    ball.Position = right - (ball.Position - right) * 0.35f;
                    ball.Velocity = -Math.Abs(ball.Velocity) * 0.55f;
                }
            }

            if (ball.Active && IsStuck(ball, row, jamThreshold, velThreshold))
            {
                ball.Stuck = true;
                events.Add(new PRM.Core.Models.BallEvent(ball.TokenId, row, ball.Position, "stuck"));
                ball.Active = false;
            }
        }

        balls.RemoveAll(b => !b.Active);
        return events;
    }

    private bool IsStuck(Ball ball, int row, float jamThreshold, float velThreshold)
    {
        if (row <= _cfg.WideningRows) return false;
        if (Math.Abs(ball.Velocity) > velThreshold) return false;

        int col = NailColumn(ball.Position, row);
        if (col < 0 || col >= _rowCols[row]) return false;

        float nailX = NailBaseX(row, col);
        float dist = Math.Abs(ball.Position - nailX);
        return dist <= jamThreshold;
    }

    private static int NailKey(int row, int col) => row * 10_000 + col;
}
