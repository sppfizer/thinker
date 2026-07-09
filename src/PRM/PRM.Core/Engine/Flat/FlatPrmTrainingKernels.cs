namespace PRM.Core.Engine.Flat;

internal static class FlatPrmTrainingKernels
{
    public static void RunTrainingSample(
        FlatPrmKernelConfig config,
        FlatPrmRowGeometry geometry,
        Span<FlatPrmGpuBallState> balls,
        Span<float> tokenOffsetX,
        Span<float> tokenOffsetY,
        Span<float> sharedOffsetX,
        Span<float> sharedOffsetY,
        ReadOnlySpan<float> nailRadii,
        ReadOnlySpan<float> nailResistances,
        ReadOnlySpan<float> nailDensities,
        Span<int> contactColumns,
        float targetCentre,
        float learningRate)
    {
        Validate(config, geometry, balls, tokenOffsetX, tokenOffsetY, sharedOffsetX, sharedOffsetY, nailRadii, nailResistances, nailDensities, contactColumns);
        contactColumns.Fill(-1);

        for (int row = 0; row < config.TotalRows; row++)
        {
            ApplyDeflectionRow(config, geometry, row, balls, tokenOffsetX, tokenOffsetY, sharedOffsetX, sharedOffsetY, nailRadii, contactColumns);
            ApplyBallInteractions(config, balls);
            ApplyTrainingUpdatesRow(config, geometry, row, balls, tokenOffsetX, tokenOffsetY, sharedOffsetX, sharedOffsetY, nailRadii, nailResistances, nailDensities, targetCentre, learningRate);
            IntegrateAndResolveBoundsRow(config, geometry, row, balls);
        }
    }

    private static void ApplyDeflectionRow(
        FlatPrmKernelConfig config,
        FlatPrmRowGeometry geometry,
        int row,
        Span<FlatPrmGpuBallState> balls,
        ReadOnlySpan<float> tokenOffsetX,
        ReadOnlySpan<float> tokenOffsetY,
        ReadOnlySpan<float> sharedOffsetX,
        ReadOnlySpan<float> sharedOffsetY,
        ReadOnlySpan<float> nailRadii,
        Span<int> contactColumns)
    {
        int rowNailCount = geometry.RowNailCounts[row];
        float rowWidth = Math.Max(geometry.RowWidths[row], 1f);
        float maxStepX = rowWidth / config.TotalRows * config.DeflectionAlpha;

        for (int i = 0; i < balls.Length; i++)
        {
            var state = balls[i];
            if (state.Active == 0) continue;

            float position = state.Position;
            float velocity = state.Velocity;
            int lastCol = -1;
            int lastToken = -1;
            int col = geometry.NailColumn(position, row);
            if (col >= 0 && col < rowNailCount)
            {
                int slot = FlatPrmCpuKernels.TokenSlot(config, state.TokenId);
                int tIdx = FlatPrmCpuKernels.TokenIndex(config, state.ContextPosition, slot);
                if (tIdx >= 0 && tIdx < config.TokenKeyCount)
                {
                    int tokenOffsetIndex = FlatPrmCpuKernels.TokenOffsetIndex(config, row, col, tIdx);
                    float blend = Math.Clamp(config.SharedOffsetBlend, 0f, 1f);
                    float offX = tokenOffsetX[tokenOffsetIndex];
                    float offY = tokenOffsetY[tokenOffsetIndex];
                    if (blend > 0f)
                    {
                        int sharedOffsetIndex = FlatPrmCpuKernels.SharedOffsetIndex(config, row, col, slot);
                        offX = offX * (1f - blend) + sharedOffsetX[sharedOffsetIndex] * blend;
                        offY = offY * (1f - blend) + sharedOffsetY[sharedOffsetIndex] * blend;
                    }

                    float idf = MathF.Pow(1f / Math.Max(state.Mass, 0.01f), config.DeflectionIdfPower);
                    float rawStepX = offX * maxStepX * idf;
                    lastCol = col;
                    lastToken = tIdx;
                    contactColumns[row * balls.Length + i] = col;
                    position += Math.Clamp(rawStepX, -maxStepX, maxStepX);
                    velocity += offY * config.DeflectionAlphaY * nailRadii[FlatPrmCpuKernels.NailIndex(config, row, col)] * idf;
                }
            }

            balls[i] = state.With(position, velocity, lastCol, lastToken, state.Active, state.Stuck);
        }
    }

    private static void ApplyBallInteractions(FlatPrmKernelConfig config, Span<FlatPrmGpuBallState> balls)
    {
        if (config.GravityG <= 0f && config.CollisionRadius <= 0f)
            return;

        for (int i = 0; i < balls.Length; i++)
        for (int j = i + 1; j < balls.Length; j++)
        {
            var left = balls[i];
            var right = balls[j];
            if (left.Active == 0 || right.Active == 0)
                continue;

            float d = Math.Abs(left.Position - right.Position);
            if (d > config.ProximityBand)
                continue;

            float leftVelocity = left.Velocity;
            float rightVelocity = right.Velocity;

            if (config.GravityG > 0f)
            {
                float g = config.GravityG * left.Mass * right.Mass / (d * d + 1e-6f);
                float dir = Math.Sign(right.Position - left.Position);
                leftVelocity += g * dir * config.DeltaTime;
                rightVelocity -= g * dir * config.DeltaTime;
            }

            if (config.CollisionRadius > 0f && d < config.CollisionRadius)
            {
                float mi = left.Mass;
                float mj = right.Mass;
                float denom = Math.Max(mi + mj, 1e-6f);
                float vi = leftVelocity;
                float vj = rightVelocity;
                leftVelocity = ((mi - mj) * vi + 2f * mj * vj) / denom;
                rightVelocity = ((mj - mi) * vj + 2f * mi * vi) / denom;
            }

            balls[i] = left.With(left.Position, leftVelocity, left.LastNailColumn, left.LastTokenIndex, left.Active, left.Stuck);
            balls[j] = right.With(right.Position, rightVelocity, right.LastNailColumn, right.LastTokenIndex, right.Active, right.Stuck);
        }
    }

    private static void ApplyTrainingUpdatesRow(
        FlatPrmKernelConfig config,
        FlatPrmRowGeometry geometry,
        int row,
        ReadOnlySpan<FlatPrmGpuBallState> balls,
        Span<float> tokenOffsetX,
        Span<float> tokenOffsetY,
        Span<float> sharedOffsetX,
        Span<float> sharedOffsetY,
        ReadOnlySpan<float> nailRadii,
        ReadOnlySpan<float> nailResistances,
        ReadOnlySpan<float> nailDensities,
        float targetCentre,
        float learningRate)
    {
        if (learningRate <= 0f) return;

        for (int i = 0; i < balls.Length; i++)
        {
            var sample = balls[i];
            if (sample.Active == 0) continue;

            int tokenId = sample.TokenId;
            bool isPredictionProbe = tokenId == -1;
            if (tokenId < 0 && !isPredictionProbe) continue;

            float probeTrainingWeight = Math.Max(config.PredictionProbeTrainingWeight, 0f);
            if (isPredictionProbe && probeTrainingWeight <= 0f) continue;

            int col = sample.LastNailColumn;
            if (col < 0 || col >= geometry.RowNailCounts[row]) continue;

            int tIdx = sample.LastTokenIndex;
            if (tIdx < 0 || tIdx >= config.TokenKeyCount) continue;

            int slot = FlatPrmCpuKernels.TokenSlot(config, tokenId);
            if (slot < 0 || slot >= config.TokenSlotCount) continue;

            float forceX = FlatPrmCpuKernels.MagnetForce(config, row, sample.Position, targetCentre);
            float rowWidth = Math.Max(geometry.RowWidths[row], 1f);
            float idealX = forceX * config.TotalRows / rowWidth;
            float idealY = idealX * 0.25f;
            float massFactor = isPredictionProbe
                ? probeTrainingWeight
                : MathF.Sqrt(Math.Max(sample.Mass, 0.01f)) * Math.Clamp(sample.RelevanceWeight, 0f, 1f);
            if (massFactor <= 0f) continue;

            int nailIndex = FlatPrmCpuKernels.NailIndex(config, row, col);
            float density = Math.Max(nailDensities[nailIndex], 0.1f);
            float inertia = Math.Max(nailResistances[nailIndex] * density * (1f + nailRadii[nailIndex]), 0.05f);
            float scale = Math.Clamp(learningRate * massFactor / inertia, 0f, 1f);
            ApplyOffsetUpdate(config, row, col, tIdx, slot, idealX, idealY, scale, tokenOffsetX, tokenOffsetY, sharedOffsetX, sharedOffsetY);
        }
    }

    private static void IntegrateAndResolveBoundsRow(
        FlatPrmKernelConfig config,
        FlatPrmRowGeometry geometry,
        int row,
        Span<FlatPrmGpuBallState> balls)
    {
        float left = geometry.LeftBorders[row];
        float right = geometry.RightBorders[row];
        float fallbackPosition = (left + right) / 2f;

        for (int i = 0; i < balls.Length; i++)
        {
            var state = balls[i];
            if (state.Active == 0) continue;

            float position = state.Position;
            float velocity = state.Velocity;
            if (float.IsNaN(velocity) || float.IsInfinity(velocity)) velocity = 0f;

            velocity = Math.Clamp(velocity, -config.MaxVelocity, config.MaxVelocity);
            position += velocity * config.DeltaTime;
            if (float.IsNaN(position) || float.IsInfinity(position)) position = fallbackPosition;

            int active = state.Active;
            int stuck = state.Stuck;
            if (position < left)
            {
                if (row <= config.WideningRows)
                {
                    active = 0;
                }
                else
                {
                    position = left + (left - position) * 0.35f;
                    velocity = Math.Abs(velocity) * 0.55f;
                }
            }
            else if (position > right)
            {
                if (row <= config.WideningRows)
                {
                    active = 0;
                }
                else
                {
                    position = right - (position - right) * 0.35f;
                    velocity = -Math.Abs(velocity) * 0.55f;
                }
            }

            if (active != 0 && IsStuck(config, geometry, row, position, velocity))
            {
                stuck = 1;
                active = 0;
            }

            balls[i] = state.With(position, velocity, state.LastNailColumn, state.LastTokenIndex, active, stuck);
        }
    }

    private static bool IsStuck(FlatPrmKernelConfig config, FlatPrmRowGeometry geometry, int row, float position, float velocity)
    {
        if (row <= config.WideningRows) return false;
        if (Math.Abs(velocity) > 0.04f) return false;

        int col = geometry.NailColumn(position, row);
        if (col < 0 || col >= geometry.RowNailCounts[row]) return false;

        float dist = Math.Abs(position - geometry.NailBaseX(row, col));
        return dist <= Math.Max(config.NailSpacing * 0.18f, 0.15f);
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

        int offsetIndex = FlatPrmCpuKernels.TokenOffsetIndex(config, row, col, tokenIndex);
        int sharedIndex = FlatPrmCpuKernels.SharedOffsetIndex(config, row, col, tokenSlot);
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

    private static void Validate(
        FlatPrmKernelConfig config,
        FlatPrmRowGeometry geometry,
        ReadOnlySpan<FlatPrmGpuBallState> balls,
        ReadOnlySpan<float> tokenOffsetX,
        ReadOnlySpan<float> tokenOffsetY,
        ReadOnlySpan<float> sharedOffsetX,
        ReadOnlySpan<float> sharedOffsetY,
        ReadOnlySpan<float> nailRadii,
        ReadOnlySpan<float> nailResistances,
        ReadOnlySpan<float> nailDensities,
        ReadOnlySpan<int> contactColumns)
    {
        if (geometry.TotalRows < config.TotalRows) throw new ArgumentException("Geometry has fewer rows than config.", nameof(geometry));
        if (geometry.MaxColumns < config.MaxColumns) throw new ArgumentException("Geometry has fewer columns than config.", nameof(geometry));

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
        EnsureLength(contactColumns.Length, config.TotalRows * balls.Length, nameof(contactColumns));
    }

    private static void EnsureLength(int actual, int required, string name)
    {
        if (actual < required)
            throw new ArgumentException($"{name} length {actual} is smaller than required length {required}.", name);
    }
}
