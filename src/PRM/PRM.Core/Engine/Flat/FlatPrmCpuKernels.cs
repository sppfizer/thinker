namespace PRM.Core.Engine.Flat;

/// <summary>
/// Deterministic flat-array CPU kernels for the first GPU-planning slice.
/// This mirrors BallSimulator's nail deflection and velocity integration math:
/// ApplyNailDeflection plus the MaxVel/DeltaTime integration block.
///
/// Comparison recipe:
/// 1. Use the same DiamondConfig, balls, nail radii, token offsets, and shared offsets.
/// 2. Keep GravityG and CollisionRadius at 0 and use learningRate 0.
/// 3. Pick a case where balls do not hit bounds or get stuck; bounds/contact history are
///    deliberately outside this first slice.
/// 4. Run BallSimulator.Simulate on copied balls, run this kernel on copied flat arrays,
///    then compare positions/velocities with FlatPrmSelfCheck.
/// </summary>
public static class FlatPrmCpuKernels
{
    public static void RunNailDeflectionIntegrationRows(
        FlatPrmKernelConfig config,
        FlatPrmRowGeometry geometry,
        int rowStart,
        int rowCount,
        Span<float> positions,
        Span<float> velocities,
        ReadOnlySpan<float> masses,
        ReadOnlySpan<int> contextPositions,
        ReadOnlySpan<int> tokenIds,
        ReadOnlySpan<float> tokenOffsetX,
        ReadOnlySpan<float> tokenOffsetY,
        ReadOnlySpan<float> sharedOffsetX,
        ReadOnlySpan<float> sharedOffsetY,
        ReadOnlySpan<float> nailRadii,
        Span<int> lastNailColumns,
        Span<int> lastTokenIndices)
    {
        ValidateInputs(
            config,
            geometry,
            rowStart,
            rowCount,
            positions,
            velocities,
            masses,
            contextPositions,
            tokenIds,
            tokenOffsetX,
            tokenOffsetY,
            sharedOffsetX,
            sharedOffsetY,
            nailRadii,
            lastNailColumns,
            lastTokenIndices);

        int endRow = rowStart + rowCount;
        for (int row = rowStart; row < endRow; row++)
        {
            ApplyNailDeflectionRow(
                config,
                geometry,
                row,
                positions,
                velocities,
                masses,
                contextPositions,
                tokenIds,
                tokenOffsetX,
                tokenOffsetY,
                sharedOffsetX,
                sharedOffsetY,
                nailRadii,
                lastNailColumns,
                lastTokenIndices);

            IntegrateVelocityRow(config, geometry, row, positions, velocities);
        }
    }

    public static void ApplyNailDeflectionRow(
        FlatPrmKernelConfig config,
        FlatPrmRowGeometry geometry,
        int row,
        Span<float> positions,
        Span<float> velocities,
        ReadOnlySpan<float> masses,
        ReadOnlySpan<int> contextPositions,
        ReadOnlySpan<int> tokenIds,
        ReadOnlySpan<float> tokenOffsetX,
        ReadOnlySpan<float> tokenOffsetY,
        ReadOnlySpan<float> sharedOffsetX,
        ReadOnlySpan<float> sharedOffsetY,
        ReadOnlySpan<float> nailRadii,
        Span<int> lastNailColumns,
        Span<int> lastTokenIndices)
    {
        int ballCount = positions.Length;
        int rowNailCount = geometry.RowNailCounts[row];
        float rowWidth = Math.Max(geometry.RowWidths[row], 1f);
        float maxStepX = rowWidth / config.TotalRows * config.DeflectionAlpha;

        for (int ball = 0; ball < ballCount; ball++)
        {
            lastNailColumns[ball] = -1;
            lastTokenIndices[ball] = -1;

            int col = geometry.NailColumn(positions[ball], row);
            if (col < 0 || col >= rowNailCount) continue;

            int slot = TokenSlot(config, tokenIds[ball]);
            int tIdx = TokenIndex(config, contextPositions[ball], slot);
            if (tIdx < 0 || tIdx >= config.TokenKeyCount) continue;

            (float offX, float offY) = EffectiveOffset(
                config,
                row,
                col,
                tIdx,
                slot,
                tokenOffsetX,
                tokenOffsetY,
                sharedOffsetX,
                sharedOffsetY);

            float radius = nailRadii[NailIndex(config, row, col)];
            float idf = MathF.Pow(1f / Math.Max(masses[ball], 0.01f), config.DeflectionIdfPower);
            float rawStepX = offX * maxStepX * idf;

            lastNailColumns[ball] = col;
            lastTokenIndices[ball] = tIdx;
            positions[ball] += Math.Clamp(rawStepX, -maxStepX, maxStepX);
            velocities[ball] += offY * config.DeflectionAlphaY * radius * idf;
        }
    }

    public static void IntegrateVelocityRow(
        FlatPrmKernelConfig config,
        FlatPrmRowGeometry geometry,
        int row,
        Span<float> positions,
        Span<float> velocities)
    {
        float left = geometry.LeftBorders[row];
        float right = geometry.RightBorders[row];
        float fallbackPosition = (left + right) / 2f;

        for (int ball = 0; ball < positions.Length; ball++)
        {
            if (float.IsNaN(velocities[ball]) || float.IsInfinity(velocities[ball]))
                velocities[ball] = 0f;

            velocities[ball] = Math.Clamp(velocities[ball], -config.MaxVelocity, config.MaxVelocity);
            positions[ball] += velocities[ball] * config.DeltaTime;

            if (float.IsNaN(positions[ball]) || float.IsInfinity(positions[ball]))
                positions[ball] = fallbackPosition;
        }
    }

    public static void ApplyTrainingUpdateRowSample(
        FlatPrmKernelConfig config,
        FlatPrmRowGeometry geometry,
        int row,
        int sampleIndex,
        float learningRate,
        float targetCentre,
        ReadOnlySpan<float> positions,
        ReadOnlySpan<float> masses,
        ReadOnlySpan<float> relevanceWeights,
        ReadOnlySpan<int> tokenIds,
        ReadOnlySpan<int> lastNailColumns,
        ReadOnlySpan<int> lastTokenIndices,
        Span<float> tokenOffsetX,
        Span<float> tokenOffsetY,
        Span<float> sharedOffsetX,
        Span<float> sharedOffsetY,
        ReadOnlySpan<float> nailRadii,
        ReadOnlySpan<float> nailResistances,
        ReadOnlySpan<float> nailDensities)
    {
        ValidateTrainingInputs(
            config,
            geometry,
            row,
            sampleIndex,
            positions,
            masses,
            relevanceWeights,
            tokenIds,
            lastNailColumns,
            lastTokenIndices,
            tokenOffsetX,
            tokenOffsetY,
            sharedOffsetX,
            sharedOffsetY,
            nailRadii,
            nailResistances,
            nailDensities);

        if (learningRate <= 0f) return;

        int tokenId = tokenIds[sampleIndex];
        bool isPredictionProbe = tokenId == -1;
        if (tokenId < 0 && !isPredictionProbe) return;

        float probeTrainingWeight = Math.Max(config.PredictionProbeTrainingWeight, 0f);
        if (isPredictionProbe && probeTrainingWeight <= 0f) return;

        int col = lastNailColumns[sampleIndex];
        if (col < 0 || col >= geometry.RowNailCounts[row]) return;

        int tIdx = lastTokenIndices[sampleIndex];
        if (tIdx < 0 || tIdx >= config.TokenKeyCount) return;

        int slot = TokenSlot(config, tokenId);
        if (slot < 0 || slot >= config.TokenSlotCount) return;

        float forceX = MagnetForce(config, row, positions[sampleIndex], targetCentre);
        float rowWidth = Math.Max(geometry.RowWidths[row], 1f);
        float idealX = forceX * config.TotalRows / rowWidth;
        float idealY = idealX * 0.25f;

        float massFactor = isPredictionProbe
            ? probeTrainingWeight
            : MathF.Sqrt(Math.Max(masses[sampleIndex], 0.01f)) *
              Math.Clamp(relevanceWeights[sampleIndex], 0f, 1f);
        if (massFactor <= 0f) return;

        int nailIndex = NailIndex(config, row, col);
        float density = Math.Max(nailDensities[nailIndex], 0.1f);
        float inertia = Math.Max(nailResistances[nailIndex] * density * (1f + nailRadii[nailIndex]), 0.05f);
        float scale = Math.Clamp(learningRate * massFactor / inertia, 0f, 1f);

        ApplyOffsetUpdate(
            config,
            row,
            col,
            tIdx,
            slot,
            idealX,
            idealY,
            scale,
            tokenOffsetX,
            tokenOffsetY,
            sharedOffsetX,
            sharedOffsetY);
    }

    public static int TokenSlot(FlatPrmKernelConfig config, int tokenId) =>
        tokenId >= 0 && tokenId < config.VocabSize ? tokenId : config.VocabSize;

    public static int TokenIndex(FlatPrmKernelConfig config, int contextPosition, int tokenSlot)
    {
        int pos = Math.Clamp(contextPosition, 0, config.WindowSize - 1);
        return pos * config.TokenSlotCount + tokenSlot;
    }

    public static int NailIndex(FlatPrmKernelConfig config, int row, int col) =>
        row * config.MaxColumns + col;

    public static int TokenOffsetIndex(FlatPrmKernelConfig config, int row, int col, int tokenIndex) =>
        ((row * config.MaxColumns + col) * config.TokenKeyCount) + tokenIndex;

    public static int SharedOffsetIndex(FlatPrmKernelConfig config, int row, int col, int tokenSlot) =>
        ((row * config.MaxColumns + col) * config.TokenSlotCount) + tokenSlot;

    public static float MagnetForce(FlatPrmKernelConfig config, int row, float x, float targetCentre)
    {
        float delta = targetCentre - x;
        if (row <= config.WideningRows)
            return delta * 0.4f;

        int narrowingRows = Math.Max(config.TotalRows - config.WideningRows, 1);
        float depthFrac = (float)(row - config.WideningRows) / narrowingRows;
        return delta * (0.4f + 0.6f * depthFrac);
    }

    private static (float X, float Y) EffectiveOffset(
        FlatPrmKernelConfig config,
        int row,
        int col,
        int tokenIndex,
        int tokenSlot,
        ReadOnlySpan<float> tokenOffsetX,
        ReadOnlySpan<float> tokenOffsetY,
        ReadOnlySpan<float> sharedOffsetX,
        ReadOnlySpan<float> sharedOffsetY)
    {
        int offsetIndex = TokenOffsetIndex(config, row, col, tokenIndex);
        float blend = Math.Clamp(config.SharedOffsetBlend, 0f, 1f);
        if (blend <= 0f)
            return (tokenOffsetX[offsetIndex], tokenOffsetY[offsetIndex]);

        int sharedIndex = SharedOffsetIndex(config, row, col, tokenSlot);
        float posX = tokenOffsetX[offsetIndex];
        float posY = tokenOffsetY[offsetIndex];
        float sharedX = sharedOffsetX[sharedIndex];
        float sharedY = sharedOffsetY[sharedIndex];
        return (posX * (1f - blend) + sharedX * blend,
                posY * (1f - blend) + sharedY * blend);
    }

    private static void ApplyOffsetUpdate(
        FlatPrmKernelConfig config,
        int row,
        int col,
        int tokenIndex,
        int tokenSlot,
        float idealX,
        float idealY,
        float scale,
        Span<float> tokenOffsetX,
        Span<float> tokenOffsetY,
        Span<float> sharedOffsetX,
        Span<float> sharedOffsetY)
    {
        scale = Math.Clamp(scale, 0f, 1f);
        if (scale <= 0f) return;

        int offsetIndex = TokenOffsetIndex(config, row, col, tokenIndex);
        int sharedIndex = SharedOffsetIndex(config, row, col, tokenSlot);
        float blend = Math.Clamp(config.SharedOffsetBlend, 0f, 1f);
        float posScale = scale * (1f - blend);
        float sharedScale = scale * blend;

        float newX = tokenOffsetX[offsetIndex] + posScale * (idealX - tokenOffsetX[offsetIndex]);
        float newY = tokenOffsetY[offsetIndex] + posScale * (idealY - tokenOffsetY[offsetIndex]);
        float newSharedX = sharedOffsetX[sharedIndex] + sharedScale * (idealX - sharedOffsetX[sharedIndex]);
        float newSharedY = sharedOffsetY[sharedIndex] + sharedScale * (idealY - sharedOffsetY[sharedIndex]);

        ProjectUnitCircle(ref newX, ref newY);
        ProjectUnitCircle(ref newSharedX, ref newSharedY);

        tokenOffsetX[offsetIndex] = newX;
        tokenOffsetY[offsetIndex] = newY;
        sharedOffsetX[sharedIndex] = newSharedX;
        sharedOffsetY[sharedIndex] = newSharedY;
    }

    private static void ProjectUnitCircle(ref float x, ref float y)
    {
        float mag = MathF.Sqrt(x * x + y * y);
        if (mag > 1f)
        {
            x /= mag;
            y /= mag;
        }
    }

    private static void ValidateInputs(
        FlatPrmKernelConfig config,
        FlatPrmRowGeometry geometry,
        int rowStart,
        int rowCount,
        ReadOnlySpan<float> positions,
        ReadOnlySpan<float> velocities,
        ReadOnlySpan<float> masses,
        ReadOnlySpan<int> contextPositions,
        ReadOnlySpan<int> tokenIds,
        ReadOnlySpan<float> tokenOffsetX,
        ReadOnlySpan<float> tokenOffsetY,
        ReadOnlySpan<float> sharedOffsetX,
        ReadOnlySpan<float> sharedOffsetY,
        ReadOnlySpan<float> nailRadii,
        ReadOnlySpan<int> lastNailColumns,
        ReadOnlySpan<int> lastTokenIndices)
    {
        if (config.TotalRows <= 0) throw new ArgumentOutOfRangeException(nameof(config), "TotalRows must be positive.");
        if (config.MaxColumns <= 0) throw new ArgumentOutOfRangeException(nameof(config), "MaxColumns must be positive.");
        if (geometry.TotalRows < config.TotalRows) throw new ArgumentException("Geometry has fewer rows than config.", nameof(geometry));
        if (geometry.MaxColumns < config.MaxColumns) throw new ArgumentException("Geometry has fewer columns than config.", nameof(geometry));
        if (rowStart < 0 || rowCount < 0 || rowStart + rowCount > config.TotalRows)
            throw new ArgumentOutOfRangeException(nameof(rowCount), "Row range is outside the configured geometry.");

        int ballCount = positions.Length;
        EnsureLength(velocities.Length, ballCount, nameof(velocities));
        EnsureLength(masses.Length, ballCount, nameof(masses));
        EnsureLength(contextPositions.Length, ballCount, nameof(contextPositions));
        EnsureLength(tokenIds.Length, ballCount, nameof(tokenIds));
        EnsureLength(lastNailColumns.Length, ballCount, nameof(lastNailColumns));
        EnsureLength(lastTokenIndices.Length, ballCount, nameof(lastTokenIndices));

        int requiredTokenOffsets = config.TotalRows * config.MaxColumns * config.TokenKeyCount;
        int requiredSharedOffsets = config.TotalRows * config.MaxColumns * config.TokenSlotCount;
        int requiredNails = config.TotalRows * config.MaxColumns;
        EnsureLength(tokenOffsetX.Length, requiredTokenOffsets, nameof(tokenOffsetX));
        EnsureLength(tokenOffsetY.Length, requiredTokenOffsets, nameof(tokenOffsetY));
        EnsureLength(sharedOffsetX.Length, requiredSharedOffsets, nameof(sharedOffsetX));
        EnsureLength(sharedOffsetY.Length, requiredSharedOffsets, nameof(sharedOffsetY));
        EnsureLength(nailRadii.Length, requiredNails, nameof(nailRadii));
    }

    private static void ValidateTrainingInputs(
        FlatPrmKernelConfig config,
        FlatPrmRowGeometry geometry,
        int row,
        int sampleIndex,
        ReadOnlySpan<float> positions,
        ReadOnlySpan<float> masses,
        ReadOnlySpan<float> relevanceWeights,
        ReadOnlySpan<int> tokenIds,
        ReadOnlySpan<int> lastNailColumns,
        ReadOnlySpan<int> lastTokenIndices,
        ReadOnlySpan<float> tokenOffsetX,
        ReadOnlySpan<float> tokenOffsetY,
        ReadOnlySpan<float> sharedOffsetX,
        ReadOnlySpan<float> sharedOffsetY,
        ReadOnlySpan<float> nailRadii,
        ReadOnlySpan<float> nailResistances,
        ReadOnlySpan<float> nailDensities)
    {
        if (config.TotalRows <= 0) throw new ArgumentOutOfRangeException(nameof(config), "TotalRows must be positive.");
        if (config.MaxColumns <= 0) throw new ArgumentOutOfRangeException(nameof(config), "MaxColumns must be positive.");
        if (geometry.TotalRows < config.TotalRows) throw new ArgumentException("Geometry has fewer rows than config.", nameof(geometry));
        if (geometry.MaxColumns < config.MaxColumns) throw new ArgumentException("Geometry has fewer columns than config.", nameof(geometry));
        if (row < 0 || row >= config.TotalRows) throw new ArgumentOutOfRangeException(nameof(row), "Row is outside the configured geometry.");
        if (sampleIndex < 0 || sampleIndex >= positions.Length)
            throw new ArgumentOutOfRangeException(nameof(sampleIndex), "Sample index is outside the input arrays.");

        int ballCount = positions.Length;
        EnsureLength(masses.Length, ballCount, nameof(masses));
        EnsureLength(relevanceWeights.Length, ballCount, nameof(relevanceWeights));
        EnsureLength(tokenIds.Length, ballCount, nameof(tokenIds));
        EnsureLength(lastNailColumns.Length, ballCount, nameof(lastNailColumns));
        EnsureLength(lastTokenIndices.Length, ballCount, nameof(lastTokenIndices));

        int requiredTokenOffsets = config.TotalRows * config.MaxColumns * config.TokenKeyCount;
        int requiredSharedOffsets = config.TotalRows * config.MaxColumns * config.TokenSlotCount;
        int requiredNails = config.TotalRows * config.MaxColumns;
        EnsureLength(tokenOffsetX.Length, requiredTokenOffsets, nameof(tokenOffsetX));
        EnsureLength(tokenOffsetY.Length, requiredTokenOffsets, nameof(tokenOffsetY));
        EnsureLength(sharedOffsetX.Length, requiredSharedOffsets, nameof(sharedOffsetX));
        EnsureLength(sharedOffsetY.Length, requiredSharedOffsets, nameof(sharedOffsetY));
        EnsureLength(nailRadii.Length, requiredNails, nameof(nailRadii));
        EnsureLength(nailResistances.Length, requiredNails, nameof(nailResistances));
        EnsureLength(nailDensities.Length, requiredNails, nameof(nailDensities));
    }

    private static void EnsureLength(int actual, int required, string name)
    {
        if (actual < required)
            throw new ArgumentException($"{name} length {actual} is smaller than required length {required}.", name);
    }
}
