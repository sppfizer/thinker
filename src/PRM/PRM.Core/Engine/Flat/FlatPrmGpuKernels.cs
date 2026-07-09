using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;

namespace PRM.Core.Engine.Flat;

internal readonly struct FlatPrmGpuKernelConfig
{
    public readonly int TotalRows;
    public readonly int WideningRows;
    public readonly int VocabSize;
    public readonly int WindowSize;
    public readonly int MaxColumns;
    public readonly int TokenSlotCount;
    public readonly int TokenKeyCount;
    public readonly int RowStart;
    public readonly int RowCount;
    public readonly float NailSpacing;
    public readonly float DeflectionAlpha;
    public readonly float DeflectionAlphaY;
    public readonly float DeflectionIdfPower;
    public readonly float SharedOffsetBlend;
    public readonly float PredictionProbeTrainingWeight;
    public readonly float GravityG;
    public readonly float ProximityBand;
    public readonly float CollisionRadius;
    public readonly float DeltaTime;
    public readonly float MaxVelocity;

    public FlatPrmGpuKernelConfig(FlatPrmKernelConfig config, int rowStart, int rowCount)
    {
        TotalRows = config.TotalRows;
        WideningRows = config.WideningRows;
        VocabSize = config.VocabSize;
        WindowSize = config.WindowSize;
        MaxColumns = config.MaxColumns;
        TokenSlotCount = config.TokenSlotCount;
        TokenKeyCount = config.TokenKeyCount;
        RowStart = rowStart;
        RowCount = rowCount;
        NailSpacing = config.NailSpacing;
        DeflectionAlpha = config.DeflectionAlpha;
        DeflectionAlphaY = config.DeflectionAlphaY;
        DeflectionIdfPower = config.DeflectionIdfPower;
        SharedOffsetBlend = config.SharedOffsetBlend;
        PredictionProbeTrainingWeight = config.PredictionProbeTrainingWeight;
        GravityG = config.GravityG;
        ProximityBand = config.ProximityBand;
        CollisionRadius = config.CollisionRadius;
        DeltaTime = config.DeltaTime;
        MaxVelocity = config.MaxVelocity;
    }
}

internal readonly struct FlatPrmGpuRowGeometry
{
    public readonly float RowWidth;
    public readonly float LeftBorder;
    public readonly float RightBorder;
    public readonly int RowNailCount;

    public FlatPrmGpuRowGeometry(float rowWidth, float leftBorder, float rightBorder, int rowNailCount)
    {
        RowWidth = rowWidth;
        LeftBorder = leftBorder;
        RightBorder = rightBorder;
        RowNailCount = rowNailCount;
    }
}

internal readonly struct FlatPrmGpuTrainingConfig
{
    public readonly int Row;
    public readonly int SampleIndex;
    public readonly float LearningRate;
    public readonly float TargetCentre;

    public FlatPrmGpuTrainingConfig(int row, int sampleIndex, float learningRate, float targetCentre)
    {
        Row = row;
        SampleIndex = sampleIndex;
        LearningRate = learningRate;
        TargetCentre = targetCentre;
    }
}

internal readonly struct FlatPrmGpuTrainingSample
{
    public readonly float Position;
    public readonly float Mass;
    public readonly float RelevanceWeight;
    public readonly int TokenId;
    public readonly int LastNailColumn;
    public readonly int LastTokenIndex;

    public FlatPrmGpuTrainingSample(
        float position,
        float mass,
        float relevanceWeight,
        int tokenId,
        int lastNailColumn,
        int lastTokenIndex)
    {
        Position = position;
        Mass = mass;
        RelevanceWeight = relevanceWeight;
        TokenId = tokenId;
        LastNailColumn = lastNailColumn;
        LastTokenIndex = lastTokenIndex;
    }
}

internal readonly struct FlatPrmGpuNailProperties
{
    public readonly float Radius;
    public readonly float Resistance;
    public readonly float Density;

    public FlatPrmGpuNailProperties(float radius, float resistance, float density)
    {
        Radius = radius;
        Resistance = resistance;
        Density = density;
    }
}

internal readonly struct FlatPrmGpuBallState
{
    public readonly float Position;
    public readonly float Velocity;
    public readonly float Mass;
    public readonly float RelevanceWeight;
    public readonly int TokenId;
    public readonly int ContextPosition;
    public readonly int LastNailColumn;
    public readonly int LastTokenIndex;
    public readonly int Active;
    public readonly int Stuck;

    public FlatPrmGpuBallState(
        float position,
        float velocity,
        float mass,
        float relevanceWeight,
        int tokenId,
        int contextPosition,
        int lastNailColumn = -1,
        int lastTokenIndex = -1,
        int active = 1,
        int stuck = 0)
    {
        Position = position;
        Velocity = velocity;
        Mass = mass;
        RelevanceWeight = relevanceWeight;
        TokenId = tokenId;
        ContextPosition = contextPosition;
        LastNailColumn = lastNailColumn;
        LastTokenIndex = lastTokenIndex;
        Active = active;
        Stuck = stuck;
    }

    public FlatPrmGpuBallState With(
        float position,
        float velocity,
        int lastNailColumn,
        int lastTokenIndex,
        int active,
        int stuck) =>
        new(
            position,
            velocity,
            Mass,
            RelevanceWeight,
            TokenId,
            ContextPosition,
            lastNailColumn,
            lastTokenIndex,
            active,
            stuck);
}

internal static class FlatPrmGpuKernels
{
    public static void RunNailDeflectionIntegrationRows(
        Index1D ball,
        FlatPrmGpuKernelConfig config,
        ArrayView1D<float, Stride1D.Dense> positions,
        ArrayView1D<float, Stride1D.Dense> velocities,
        ArrayView1D<float, Stride1D.Dense> masses,
        ArrayView1D<int, Stride1D.Dense> contextPositions,
        ArrayView1D<int, Stride1D.Dense> tokenIds,
        ArrayView1D<float, Stride1D.Dense> tokenOffsetX,
        ArrayView1D<float, Stride1D.Dense> tokenOffsetY,
        ArrayView1D<float, Stride1D.Dense> sharedOffsetX,
        ArrayView1D<float, Stride1D.Dense> sharedOffsetY,
        ArrayView1D<float, Stride1D.Dense> nailRadii,
        ArrayView1D<int, Stride1D.Dense> lastNailState,
        ArrayView1D<FlatPrmGpuRowGeometry, Stride1D.Dense> rowGeometry)
    {
        int i = ball;
        if (i < 0 || i >= positions.Length)
            return;

        float position = positions[i];
        float velocity = velocities[i];
        float mass = masses[i];
        int contextPosition = contextPositions[i];
        int tokenId = tokenIds[i];
        int endRow = config.RowStart + config.RowCount;

        for (int row = config.RowStart; row < endRow; row++)
        {
            lastNailState[i] = -1;
            lastNailState[positions.Length + i] = -1;

            FlatPrmGpuRowGeometry geometry = rowGeometry[row];
            int col = NailColumn(position, row, config.NailSpacing, geometry.LeftBorder);
            if (col >= 0 && col < geometry.RowNailCount)
            {
                int slot = TokenSlot(config.VocabSize, tokenId);
                int tIdx = TokenIndex(config.WindowSize, config.TokenSlotCount, contextPosition, slot);
                if (tIdx >= 0 && tIdx < config.TokenKeyCount)
                {
                    float blend = XMath.Clamp(config.SharedOffsetBlend, 0f, 1f);
                    int tokenOffsetIndex = ((row * config.MaxColumns + col) * config.TokenKeyCount) + tIdx;
                    float offX = tokenOffsetX[tokenOffsetIndex];
                    float offY = tokenOffsetY[tokenOffsetIndex];
                    if (blend > 0f)
                    {
                        int sharedOffsetIndex = ((row * config.MaxColumns + col) * config.TokenSlotCount) + slot;
                        offX = offX * (1f - blend) + sharedOffsetX[sharedOffsetIndex] * blend;
                        offY = offY * (1f - blend) + sharedOffsetY[sharedOffsetIndex] * blend;
                    }

                    float radius = nailRadii[row * config.MaxColumns + col];
                    float idf = XMath.Pow(1f / XMath.Max(mass, 0.01f), config.DeflectionIdfPower);
                    float rowWidth = XMath.Max(geometry.RowWidth, 1f);
                    float maxStepX = rowWidth / config.TotalRows * config.DeflectionAlpha;
                    float rawStepX = offX * maxStepX * idf;

                    lastNailState[i] = col;
                    lastNailState[positions.Length + i] = tIdx;
                    position += XMath.Clamp(rawStepX, -maxStepX, maxStepX);
                    velocity += offY * config.DeflectionAlphaY * radius * idf;
                }
            }

            float fallbackPosition = (geometry.LeftBorder + geometry.RightBorder) * 0.5f;
            if (XMath.IsNaN(velocity) || XMath.IsInfinity(velocity))
                velocity = 0f;

            velocity = XMath.Clamp(velocity, -config.MaxVelocity, config.MaxVelocity);
            position += velocity * config.DeltaTime;

            if (XMath.IsNaN(position) || XMath.IsInfinity(position))
                position = fallbackPosition;
        }

        positions[i] = position;
        velocities[i] = velocity;
    }

    public static void ApplyTrainingUpdateRowSample(
        Index1D index,
        FlatPrmGpuKernelConfig config,
        FlatPrmGpuTrainingConfig training,
        ArrayView1D<FlatPrmGpuTrainingSample, Stride1D.Dense> samples,
        ArrayView1D<float, Stride1D.Dense> tokenOffsetX,
        ArrayView1D<float, Stride1D.Dense> tokenOffsetY,
        ArrayView1D<float, Stride1D.Dense> sharedOffsetX,
        ArrayView1D<float, Stride1D.Dense> sharedOffsetY,
        ArrayView1D<FlatPrmGpuNailProperties, Stride1D.Dense> nailProperties,
        ArrayView1D<FlatPrmGpuRowGeometry, Stride1D.Dense> rowGeometry)
    {
        if (index != 0 || training.Row < 0 || training.Row >= config.TotalRows ||
            training.SampleIndex < 0 || training.SampleIndex >= samples.Length ||
            training.LearningRate <= 0f)
        {
            return;
        }

        FlatPrmGpuTrainingSample sample = samples[training.SampleIndex];
        int tokenId = sample.TokenId;
        bool isPredictionProbe = tokenId == -1;
        if (tokenId < 0 && !isPredictionProbe)
            return;

        float probeTrainingWeight = XMath.Max(config.PredictionProbeTrainingWeight, 0f);
        if (isPredictionProbe && probeTrainingWeight <= 0f)
            return;

        FlatPrmGpuRowGeometry geometry = rowGeometry[training.Row];
        int col = sample.LastNailColumn;
        if (col < 0 || col >= geometry.RowNailCount)
            return;

        int tIdx = sample.LastTokenIndex;
        if (tIdx < 0 || tIdx >= config.TokenKeyCount)
            return;

        int slot = TokenSlot(config.VocabSize, tokenId);
        if (slot < 0 || slot >= config.TokenSlotCount)
            return;

        float forceX = MagnetForce(config, training.Row, sample.Position, training.TargetCentre);
        float rowWidth = XMath.Max(geometry.RowWidth, 1f);
        float idealX = forceX * config.TotalRows / rowWidth;
        float idealY = idealX * 0.25f;

        float massFactor;
        if (isPredictionProbe)
        {
            massFactor = probeTrainingWeight;
        }
        else
        {
            float relevance = XMath.Clamp(sample.RelevanceWeight, 0f, 1f);
            massFactor = XMath.Sqrt(XMath.Max(sample.Mass, 0.01f)) * relevance;
        }

        if (massFactor <= 0f)
            return;

        int nailIndex = training.Row * config.MaxColumns + col;
        FlatPrmGpuNailProperties nail = nailProperties[nailIndex];
        float density = XMath.Max(nail.Density, 0.1f);
        float inertia = XMath.Max(nail.Resistance * density * (1f + nail.Radius), 0.05f);
        float scale = XMath.Clamp(training.LearningRate * massFactor / inertia, 0f, 1f);
        if (scale <= 0f)
            return;

        ApplyOffsetUpdate(
            config,
            training.Row,
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

    public static void ApplyTrainingDeflectionRow(
        Index1D ball,
        FlatPrmGpuKernelConfig config,
        int row,
        ArrayView1D<FlatPrmGpuBallState, Stride1D.Dense> balls,
        ArrayView1D<float, Stride1D.Dense> tokenOffsetX,
        ArrayView1D<float, Stride1D.Dense> tokenOffsetY,
        ArrayView1D<float, Stride1D.Dense> sharedOffsetX,
        ArrayView1D<float, Stride1D.Dense> sharedOffsetY,
        ArrayView1D<FlatPrmGpuNailProperties, Stride1D.Dense> nailProperties,
        ArrayView1D<int, Stride1D.Dense> contactColumns,
        ArrayView1D<FlatPrmGpuRowGeometry, Stride1D.Dense> rowGeometry)
    {
        int i = ball;
        if (i < 0 || i >= balls.Length || row < 0 || row >= config.TotalRows)
            return;

        FlatPrmGpuBallState state = balls[i];
        if (state.Active == 0)
            return;

        float position = state.Position;
        float velocity = state.Velocity;
        int lastCol = -1;
        int lastToken = -1;

        FlatPrmGpuRowGeometry geometry = rowGeometry[row];
        int col = NailColumn(position, row, config.NailSpacing, geometry.LeftBorder);
        if (col >= 0 && col < geometry.RowNailCount)
        {
            int slot = TokenSlot(config.VocabSize, state.TokenId);
            int tIdx = TokenIndex(config.WindowSize, config.TokenSlotCount, state.ContextPosition, slot);
            if (tIdx >= 0 && tIdx < config.TokenKeyCount)
            {
                float blend = XMath.Clamp(config.SharedOffsetBlend, 0f, 1f);
                int tokenOffsetIndex = ((row * config.MaxColumns + col) * config.TokenKeyCount) + tIdx;
                float offX = tokenOffsetX[tokenOffsetIndex];
                float offY = tokenOffsetY[tokenOffsetIndex];
                if (blend > 0f)
                {
                    int sharedOffsetIndex = ((row * config.MaxColumns + col) * config.TokenSlotCount) + slot;
                    offX = offX * (1f - blend) + sharedOffsetX[sharedOffsetIndex] * blend;
                    offY = offY * (1f - blend) + sharedOffsetY[sharedOffsetIndex] * blend;
                }

                FlatPrmGpuNailProperties nail = nailProperties[row * config.MaxColumns + col];
                float idf = XMath.Pow(1f / XMath.Max(state.Mass, 0.01f), config.DeflectionIdfPower);
                float rowWidth = XMath.Max(geometry.RowWidth, 1f);
                float maxStepX = rowWidth / config.TotalRows * config.DeflectionAlpha;
                float rawStepX = offX * maxStepX * idf;

                lastCol = col;
                lastToken = tIdx;
                contactColumns[row * balls.Length + i] = col;
                position += XMath.Clamp(rawStepX, -maxStepX, maxStepX);
                velocity += offY * config.DeflectionAlphaY * nail.Radius * idf;
            }
        }

        balls[i] = state.With(position, velocity, lastCol, lastToken, state.Active, state.Stuck);
    }

    public static void ApplyBallInteractions(
        Index1D index,
        FlatPrmGpuKernelConfig config,
        ArrayView1D<FlatPrmGpuBallState, Stride1D.Dense> balls)
    {
        if (index != 0 || (config.GravityG <= 0f && config.CollisionRadius <= 0f))
            return;

        for (int i = 0; i < balls.Length; i++)
        for (int j = i + 1; j < balls.Length; j++)
        {
            FlatPrmGpuBallState left = balls[i];
            FlatPrmGpuBallState right = balls[j];
            if (left.Active == 0 || right.Active == 0)
                continue;

            float d = XMath.Abs(left.Position - right.Position);
            if (d > config.ProximityBand)
                continue;

            float leftVelocity = left.Velocity;
            float rightVelocity = right.Velocity;

            if (config.GravityG > 0f)
            {
                float g = config.GravityG * left.Mass * right.Mass / (d * d + 1e-6f);
                float dir = Sign(right.Position - left.Position);
                leftVelocity += g * dir * config.DeltaTime;
                rightVelocity -= g * dir * config.DeltaTime;
            }

            if (config.CollisionRadius > 0f && d < config.CollisionRadius)
            {
                float mi = left.Mass;
                float mj = right.Mass;
                float denom = XMath.Max(mi + mj, 1e-6f);
                float vi = leftVelocity;
                float vj = rightVelocity;
                leftVelocity = ((mi - mj) * vi + 2f * mj * vj) / denom;
                rightVelocity = ((mj - mi) * vj + 2f * mi * vi) / denom;
            }

            balls[i] = left.With(left.Position, leftVelocity, left.LastNailColumn, left.LastTokenIndex, left.Active, left.Stuck);
            balls[j] = right.With(right.Position, rightVelocity, right.LastNailColumn, right.LastTokenIndex, right.Active, right.Stuck);
        }
    }

    public static void ApplyTrainingUpdatesRow(
        Index1D index,
        FlatPrmGpuKernelConfig config,
        int row,
        float learningRate,
        float targetCentre,
        ArrayView1D<FlatPrmGpuBallState, Stride1D.Dense> balls,
        ArrayView1D<float, Stride1D.Dense> tokenOffsetX,
        ArrayView1D<float, Stride1D.Dense> tokenOffsetY,
        ArrayView1D<float, Stride1D.Dense> sharedOffsetX,
        ArrayView1D<float, Stride1D.Dense> sharedOffsetY,
        ArrayView1D<FlatPrmGpuNailProperties, Stride1D.Dense> nailProperties,
        ArrayView1D<FlatPrmGpuRowGeometry, Stride1D.Dense> rowGeometry)
    {
        if (index != 0 || row < 0 || row >= config.TotalRows || learningRate <= 0f)
            return;

        FlatPrmGpuRowGeometry geometry = rowGeometry[row];
        for (int i = 0; i < balls.Length; i++)
        {
            FlatPrmGpuBallState sample = balls[i];
            if (sample.Active == 0)
                continue;

            int tokenId = sample.TokenId;
            bool isPredictionProbe = tokenId == -1;
            if (tokenId < 0 && !isPredictionProbe)
                continue;

            float probeTrainingWeight = XMath.Max(config.PredictionProbeTrainingWeight, 0f);
            if (isPredictionProbe && probeTrainingWeight <= 0f)
                continue;

            int col = sample.LastNailColumn;
            if (col < 0 || col >= geometry.RowNailCount)
                continue;

            int tIdx = sample.LastTokenIndex;
            if (tIdx < 0 || tIdx >= config.TokenKeyCount)
                continue;

            int slot = TokenSlot(config.VocabSize, tokenId);
            if (slot < 0 || slot >= config.TokenSlotCount)
                continue;

            float forceX = MagnetForce(config, row, sample.Position, targetCentre);
            float rowWidth = XMath.Max(geometry.RowWidth, 1f);
            float idealX = forceX * config.TotalRows / rowWidth;
            float idealY = idealX * 0.25f;

            float massFactor;
            if (isPredictionProbe)
            {
                massFactor = probeTrainingWeight;
            }
            else
            {
                float relevance = XMath.Clamp(sample.RelevanceWeight, 0f, 1f);
                massFactor = XMath.Sqrt(XMath.Max(sample.Mass, 0.01f)) * relevance;
            }

            if (massFactor <= 0f)
                continue;

            int nailIndex = row * config.MaxColumns + col;
            FlatPrmGpuNailProperties nail = nailProperties[nailIndex];
            float density = XMath.Max(nail.Density, 0.1f);
            float inertia = XMath.Max(nail.Resistance * density * (1f + nail.Radius), 0.05f);
            float scale = XMath.Clamp(learningRate * massFactor / inertia, 0f, 1f);
            if (scale <= 0f)
                continue;

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
    }

    public static void IntegrateAndResolveBoundsRow(
        Index1D ball,
        FlatPrmGpuKernelConfig config,
        int row,
        ArrayView1D<FlatPrmGpuBallState, Stride1D.Dense> balls,
        ArrayView1D<FlatPrmGpuRowGeometry, Stride1D.Dense> rowGeometry)
    {
        int i = ball;
        if (i < 0 || i >= balls.Length || row < 0 || row >= config.TotalRows)
            return;

        FlatPrmGpuBallState state = balls[i];
        if (state.Active == 0)
            return;

        FlatPrmGpuRowGeometry geometry = rowGeometry[row];
        float position = state.Position;
        float velocity = state.Velocity;
        float left = geometry.LeftBorder;
        float right = geometry.RightBorder;
        float fallbackPosition = (left + right) * 0.5f;

        if (XMath.IsNaN(velocity) || XMath.IsInfinity(velocity))
            velocity = 0f;

        velocity = XMath.Clamp(velocity, -config.MaxVelocity, config.MaxVelocity);
        position += velocity * config.DeltaTime;

        if (XMath.IsNaN(position) || XMath.IsInfinity(position))
            position = fallbackPosition;

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
                velocity = XMath.Abs(velocity) * 0.55f;
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
                velocity = -XMath.Abs(velocity) * 0.55f;
            }
        }

        if (active != 0 && row > config.WideningRows)
        {
            float jamThreshold = XMath.Max(config.NailSpacing * 0.18f, 0.15f);
            if (XMath.Abs(velocity) <= 0.04f)
            {
                int col = NailColumn(position, row, config.NailSpacing, left);
                if (col >= 0 && col < geometry.RowNailCount)
                {
                    float stagger = (row & 1) == 1 ? config.NailSpacing * 0.5f : 0f;
                    float nailX = left + stagger + col * config.NailSpacing;
                    if (XMath.Abs(position - nailX) <= jamThreshold)
                    {
                        stuck = 1;
                        active = 0;
                    }
                }
            }
        }

        balls[i] = state.With(position, velocity, state.LastNailColumn, state.LastTokenIndex, active, stuck);
    }

    public static void RunTrainingSampleRows(
        Index1D index,
        FlatPrmGpuKernelConfig config,
        int ballCount,
        float learningRate,
        float targetCentre,
        ArrayView1D<FlatPrmGpuBallState, Stride1D.Dense> balls,
        ArrayView1D<float, Stride1D.Dense> tokenOffsetX,
        ArrayView1D<float, Stride1D.Dense> tokenOffsetY,
        ArrayView1D<float, Stride1D.Dense> sharedOffsetX,
        ArrayView1D<float, Stride1D.Dense> sharedOffsetY,
        ArrayView1D<FlatPrmGpuNailProperties, Stride1D.Dense> nailProperties,
        ArrayView1D<int, Stride1D.Dense> contactColumns,
        ArrayView1D<FlatPrmGpuRowGeometry, Stride1D.Dense> rowGeometry)
    {
        if (index != 0)
            return;

        int activeBallCount = ballCount;
        if (activeBallCount < 0) activeBallCount = 0;
        if (activeBallCount > balls.IntLength) activeBallCount = balls.IntLength;

        for (int row = 0; row < config.TotalRows; row++)
        {
            FlatPrmGpuRowGeometry geometry = rowGeometry[row];
            for (int i = 0; i < activeBallCount; i++)
            {
                FlatPrmGpuBallState state = balls[i];
                if (state.Active == 0)
                    continue;

                float position = state.Position;
                float velocity = state.Velocity;
                int lastCol = -1;
                int lastToken = -1;

                int col = NailColumn(position, row, config.NailSpacing, geometry.LeftBorder);
                if (col >= 0 && col < geometry.RowNailCount)
                {
                    int slot = TokenSlot(config.VocabSize, state.TokenId);
                    int tIdx = TokenIndex(config.WindowSize, config.TokenSlotCount, state.ContextPosition, slot);
                    if (tIdx >= 0 && tIdx < config.TokenKeyCount)
                    {
                        float blend = XMath.Clamp(config.SharedOffsetBlend, 0f, 1f);
                        int tokenOffsetIndex = ((row * config.MaxColumns + col) * config.TokenKeyCount) + tIdx;
                        float offX = tokenOffsetX[tokenOffsetIndex];
                        float offY = tokenOffsetY[tokenOffsetIndex];
                        if (blend > 0f)
                        {
                            int sharedOffsetIndex = ((row * config.MaxColumns + col) * config.TokenSlotCount) + slot;
                            offX = offX * (1f - blend) + sharedOffsetX[sharedOffsetIndex] * blend;
                            offY = offY * (1f - blend) + sharedOffsetY[sharedOffsetIndex] * blend;
                        }

                        FlatPrmGpuNailProperties nail = nailProperties[row * config.MaxColumns + col];
                        float idf = XMath.Pow(1f / XMath.Max(state.Mass, 0.01f), config.DeflectionIdfPower);
                        float rowWidth = XMath.Max(geometry.RowWidth, 1f);
                        float maxStepX = rowWidth / config.TotalRows * config.DeflectionAlpha;
                        float rawStepX = offX * maxStepX * idf;

                        lastCol = col;
                        lastToken = tIdx;
                        contactColumns[row * balls.Length + i] = col;
                        position += XMath.Clamp(rawStepX, -maxStepX, maxStepX);
                        velocity += offY * config.DeflectionAlphaY * nail.Radius * idf;
                    }
                }

                balls[i] = state.With(position, velocity, lastCol, lastToken, state.Active, state.Stuck);
            }

            if (config.GravityG > 0f || config.CollisionRadius > 0f)
            {
                for (int i = 0; i < activeBallCount; i++)
                for (int j = i + 1; j < activeBallCount; j++)
                {
                    FlatPrmGpuBallState left = balls[i];
                    FlatPrmGpuBallState right = balls[j];
                    if (left.Active == 0 || right.Active == 0)
                        continue;

                    float d = XMath.Abs(left.Position - right.Position);
                    if (d > config.ProximityBand)
                        continue;

                    float leftVelocity = left.Velocity;
                    float rightVelocity = right.Velocity;

                    if (config.GravityG > 0f)
                    {
                        float g = config.GravityG * left.Mass * right.Mass / (d * d + 1e-6f);
                        float dir = Sign(right.Position - left.Position);
                        leftVelocity += g * dir * config.DeltaTime;
                        rightVelocity -= g * dir * config.DeltaTime;
                    }

                    if (config.CollisionRadius > 0f && d < config.CollisionRadius)
                    {
                        float mi = left.Mass;
                        float mj = right.Mass;
                        float denom = XMath.Max(mi + mj, 1e-6f);
                        float vi = leftVelocity;
                        float vj = rightVelocity;
                        leftVelocity = ((mi - mj) * vi + 2f * mj * vj) / denom;
                        rightVelocity = ((mj - mi) * vj + 2f * mi * vi) / denom;
                    }

                    balls[i] = left.With(left.Position, leftVelocity, left.LastNailColumn, left.LastTokenIndex, left.Active, left.Stuck);
                    balls[j] = right.With(right.Position, rightVelocity, right.LastNailColumn, right.LastTokenIndex, right.Active, right.Stuck);
                }
            }

            if (learningRate > 0f)
            {
                for (int i = 0; i < activeBallCount; i++)
                {
                    FlatPrmGpuBallState sample = balls[i];
                    if (sample.Active == 0)
                        continue;

                    int tokenId = sample.TokenId;
                    bool isPredictionProbe = tokenId == -1;
                    if (tokenId < 0 && !isPredictionProbe)
                        continue;

                    float probeTrainingWeight = XMath.Max(config.PredictionProbeTrainingWeight, 0f);
                    if (isPredictionProbe && probeTrainingWeight <= 0f)
                        continue;

                    int col = sample.LastNailColumn;
                    if (col < 0 || col >= geometry.RowNailCount)
                        continue;

                    int tIdx = sample.LastTokenIndex;
                    if (tIdx < 0 || tIdx >= config.TokenKeyCount)
                        continue;

                    int slot = TokenSlot(config.VocabSize, tokenId);
                    if (slot < 0 || slot >= config.TokenSlotCount)
                        continue;

                    float forceX = MagnetForce(config, row, sample.Position, targetCentre);
                    float rowWidth = XMath.Max(geometry.RowWidth, 1f);
                    float idealX = forceX * config.TotalRows / rowWidth;
                    float idealY = idealX * 0.25f;
                    float massFactor = isPredictionProbe
                        ? probeTrainingWeight
                        : XMath.Sqrt(XMath.Max(sample.Mass, 0.01f)) * XMath.Clamp(sample.RelevanceWeight, 0f, 1f);
                    if (massFactor <= 0f)
                        continue;

                    int nailIndex = row * config.MaxColumns + col;
                    FlatPrmGpuNailProperties nail = nailProperties[nailIndex];
                    float density = XMath.Max(nail.Density, 0.1f);
                    float inertia = XMath.Max(nail.Resistance * density * (1f + nail.Radius), 0.05f);
                    float scale = XMath.Clamp(learningRate * massFactor / inertia, 0f, 1f);
                    if (scale <= 0f)
                        continue;

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
            }

            for (int i = 0; i < activeBallCount; i++)
            {
                FlatPrmGpuBallState state = balls[i];
                if (state.Active == 0)
                    continue;

                float position = state.Position;
                float velocity = state.Velocity;
                float left = geometry.LeftBorder;
                float right = geometry.RightBorder;
                float fallbackPosition = (left + right) * 0.5f;

                if (XMath.IsNaN(velocity) || XMath.IsInfinity(velocity))
                    velocity = 0f;

                velocity = XMath.Clamp(velocity, -config.MaxVelocity, config.MaxVelocity);
                position += velocity * config.DeltaTime;

                if (XMath.IsNaN(position) || XMath.IsInfinity(position))
                    position = fallbackPosition;

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
                        velocity = XMath.Abs(velocity) * 0.55f;
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
                        velocity = -XMath.Abs(velocity) * 0.55f;
                    }
                }

                if (active != 0 && row > config.WideningRows && XMath.Abs(velocity) <= 0.04f)
                {
                    int col = NailColumn(position, row, config.NailSpacing, left);
                    if (col >= 0 && col < geometry.RowNailCount)
                    {
                        float stagger = (row & 1) == 1 ? config.NailSpacing * 0.5f : 0f;
                        float nailX = left + stagger + col * config.NailSpacing;
                        if (XMath.Abs(position - nailX) <= XMath.Max(config.NailSpacing * 0.18f, 0.15f))
                        {
                            stuck = 1;
                            active = 0;
                        }
                    }
                }

                balls[i] = state.With(position, velocity, state.LastNailColumn, state.LastTokenIndex, active, stuck);
            }
        }
    }

    private static int TokenSlot(int vocabSize, int tokenId) =>
        tokenId >= 0 && tokenId < vocabSize ? tokenId : vocabSize;

    private static int TokenIndex(int windowSize, int tokenSlotCount, int contextPosition, int tokenSlot)
    {
        int pos = XMath.Clamp(contextPosition, 0, windowSize - 1);
        return pos * tokenSlotCount + tokenSlot;
    }

    private static int NailColumn(float x, int row, float nailSpacing, float leftBorder)
    {
        if (XMath.IsNaN(x) || XMath.IsInfinity(x))
            return -1;

        float stagger = (row & 1) == 1 ? nailSpacing * 0.5f : 0f;
        return (int)((x - leftBorder - stagger) / nailSpacing);
    }

    private static float MagnetForce(FlatPrmGpuKernelConfig config, int row, float x, float targetCentre)
    {
        float delta = targetCentre - x;
        if (row <= config.WideningRows)
            return delta * 0.4f;

        int narrowingRows = XMath.Max(config.TotalRows - config.WideningRows, 1);
        float depthFrac = (float)(row - config.WideningRows) / narrowingRows;
        return delta * (0.4f + 0.6f * depthFrac);
    }

    private static float Sign(float value) =>
        value > 0f ? 1f : value < 0f ? -1f : 0f;

    private static void ApplyOffsetUpdate(
        FlatPrmGpuKernelConfig config,
        int row,
        int col,
        int tIdx,
        int slot,
        float idealX,
        float idealY,
        float scale,
        ArrayView1D<float, Stride1D.Dense> tokenOffsetX,
        ArrayView1D<float, Stride1D.Dense> tokenOffsetY,
        ArrayView1D<float, Stride1D.Dense> sharedOffsetX,
        ArrayView1D<float, Stride1D.Dense> sharedOffsetY)
    {
        float blend = XMath.Clamp(config.SharedOffsetBlend, 0f, 1f);
        float posScale = scale * (1f - blend);
        float sharedScale = scale * blend;
        int tokenOffsetIndex = ((row * config.MaxColumns + col) * config.TokenKeyCount) + tIdx;
        int sharedOffsetIndex = ((row * config.MaxColumns + col) * config.TokenSlotCount) + slot;

        float newX = tokenOffsetX[tokenOffsetIndex] + posScale * (idealX - tokenOffsetX[tokenOffsetIndex]);
        float newY = tokenOffsetY[tokenOffsetIndex] + posScale * (idealY - tokenOffsetY[tokenOffsetIndex]);
        float newSharedX = sharedOffsetX[sharedOffsetIndex] + sharedScale * (idealX - sharedOffsetX[sharedOffsetIndex]);
        float newSharedY = sharedOffsetY[sharedOffsetIndex] + sharedScale * (idealY - sharedOffsetY[sharedOffsetIndex]);

        ProjectUnitCircle(ref newX, ref newY);
        ProjectUnitCircle(ref newSharedX, ref newSharedY);

        tokenOffsetX[tokenOffsetIndex] = newX;
        tokenOffsetY[tokenOffsetIndex] = newY;
        sharedOffsetX[sharedOffsetIndex] = newSharedX;
        sharedOffsetY[sharedOffsetIndex] = newSharedY;
    }

    private static void ProjectUnitCircle(ref float x, ref float y)
    {
        float mag = XMath.Sqrt(x * x + y * y);
        if (mag > 1f)
        {
            x /= mag;
            y /= mag;
        }
    }
}
