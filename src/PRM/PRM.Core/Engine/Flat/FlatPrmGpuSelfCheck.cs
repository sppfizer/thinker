using PRM.Core.Models;

namespace PRM.Core.Engine.Flat;

public static class FlatPrmGpuSelfCheck
{
    public static FlatPrmGpuParityResult RunCpuGpuParity(float tolerance = 1e-5f, int? openClDeviceIndex = null)
    {
        var (config, geometry, inputs) = CreateDeterministicInputs();

        float[] cpuPositions = inputs.Positions.ToArray();
        float[] cpuVelocities = inputs.Velocities.ToArray();
        int[] cpuLastColumns = new int[cpuPositions.Length];
        int[] cpuLastTokenIndices = new int[cpuPositions.Length];

        FlatPrmCpuKernels.RunNailDeflectionIntegrationRows(
            config,
            geometry,
            0,
            config.TotalRows,
            cpuPositions,
            cpuVelocities,
            inputs.Masses,
            inputs.ContextPositions,
            inputs.TokenIds,
            inputs.TokenOffsetX,
            inputs.TokenOffsetY,
            inputs.SharedOffsetX,
            inputs.SharedOffsetY,
            inputs.NailRadii,
            cpuLastColumns,
            cpuLastTokenIndices);

        float[] gpuPositions = inputs.Positions.ToArray();
        float[] gpuVelocities = inputs.Velocities.ToArray();
        int[] gpuLastColumns = new int[gpuPositions.Length];
        int[] gpuLastTokenIndices = new int[gpuPositions.Length];

        FlatPrmGpuRunResult run = FlatPrmGpuBackend.RunNailDeflectionIntegrationRows(
            config,
            geometry,
            0,
            config.TotalRows,
            gpuPositions,
            gpuVelocities,
            inputs.Masses,
            inputs.ContextPositions,
            inputs.TokenIds,
            inputs.TokenOffsetX,
            inputs.TokenOffsetY,
            inputs.SharedOffsetX,
            inputs.SharedOffsetY,
            inputs.NailRadii,
            gpuLastColumns,
            gpuLastTokenIndices,
            openClDeviceIndex,
            allowCpuFallback: false);

        var comparison = FlatPrmSelfCheck.CompareFinalState(
            gpuPositions,
            gpuVelocities,
            cpuPositions,
            cpuVelocities);
        bool lastStateMatches = cpuLastColumns.SequenceEqual(gpuLastColumns) &&
            cpuLastTokenIndices.SequenceEqual(gpuLastTokenIndices);
        bool passed = comparison.IsWithin(tolerance) && lastStateMatches;

        return new FlatPrmGpuParityResult(
            GpuAvailable: run.UsedGpu,
            Passed: passed,
            Device: run.Device,
            Comparison: comparison,
            Message: passed
                ? $"CPU-flat vs GPU-flat parity passed within {tolerance}."
                : $"CPU-flat vs GPU-flat parity failed within {tolerance}; lastStateMatches={lastStateMatches}.");
    }

    public static bool TryRunCpuGpuParity(
        out FlatPrmGpuParityResult result,
        float tolerance = 1e-5f,
        int? openClDeviceIndex = null)
    {
        try
        {
            result = RunCpuGpuParity(tolerance, openClDeviceIndex);
            return result.Passed;
        }
        catch (Exception ex)
        {
            result = new FlatPrmGpuParityResult(
                GpuAvailable: false,
                Passed: false,
                Device: null,
                Comparison: null,
                Message: $"OpenCL GPU parity check was not run ({ex.GetType().Name}: {ex.Message}).");
            return false;
        }
    }

    public static FlatPrmGpuTrainingParityResult RunCpuGpuTrainingUpdateParity(
        float tolerance = 1e-5f,
        int? openClDeviceIndex = null)
    {
        var (config, geometry, inputs) = CreateDeterministicInputs();
        const int row = 2;
        const int sampleIndex = 2;
        const float learningRate = 0.08f;
        const float targetCentre = 2.25f;

        int[] lastNailColumns = CreateLastNailColumns(config, geometry, row, inputs.Positions);
        int[] lastTokenIndices = CreateLastTokenIndices(config, inputs.ContextPositions, inputs.TokenIds);

        float[] cpuTokenOffsetX = inputs.TokenOffsetX.ToArray();
        float[] cpuTokenOffsetY = inputs.TokenOffsetY.ToArray();
        float[] cpuSharedOffsetX = inputs.SharedOffsetX.ToArray();
        float[] cpuSharedOffsetY = inputs.SharedOffsetY.ToArray();

        FlatPrmCpuKernels.ApplyTrainingUpdateRowSample(
            config,
            geometry,
            row,
            sampleIndex,
            learningRate,
            targetCentre,
            inputs.Positions,
            inputs.Masses,
            inputs.RelevanceWeights,
            inputs.TokenIds,
            lastNailColumns,
            lastTokenIndices,
            cpuTokenOffsetX,
            cpuTokenOffsetY,
            cpuSharedOffsetX,
            cpuSharedOffsetY,
            inputs.NailRadii,
            inputs.NailResistances,
            inputs.NailDensities);

        float[] gpuTokenOffsetX = inputs.TokenOffsetX.ToArray();
        float[] gpuTokenOffsetY = inputs.TokenOffsetY.ToArray();
        float[] gpuSharedOffsetX = inputs.SharedOffsetX.ToArray();
        float[] gpuSharedOffsetY = inputs.SharedOffsetY.ToArray();

        FlatPrmGpuRunResult run = FlatPrmGpuBackend.ApplyTrainingUpdateRowSample(
            config,
            geometry,
            row,
            sampleIndex,
            learningRate,
            targetCentre,
            inputs.Positions,
            inputs.Masses,
            inputs.RelevanceWeights,
            inputs.TokenIds,
            lastNailColumns,
            lastTokenIndices,
            gpuTokenOffsetX,
            gpuTokenOffsetY,
            gpuSharedOffsetX,
            gpuSharedOffsetY,
            inputs.NailRadii,
            inputs.NailResistances,
            inputs.NailDensities,
            openClDeviceIndex,
            allowCpuFallback: false);

        var tokenX = CompareArrays(gpuTokenOffsetX, cpuTokenOffsetX);
        var tokenY = CompareArrays(gpuTokenOffsetY, cpuTokenOffsetY);
        var sharedX = CompareArrays(gpuSharedOffsetX, cpuSharedOffsetX);
        var sharedY = CompareArrays(gpuSharedOffsetY, cpuSharedOffsetY);
        bool passed = tokenX.IsWithin(tolerance) &&
            tokenY.IsWithin(tolerance) &&
            sharedX.IsWithin(tolerance) &&
            sharedY.IsWithin(tolerance);

        return new FlatPrmGpuTrainingParityResult(
            GpuAvailable: run.UsedGpu,
            Passed: passed,
            Device: run.Device,
            TokenOffsetXComparison: tokenX,
            TokenOffsetYComparison: tokenY,
            SharedOffsetXComparison: sharedX,
            SharedOffsetYComparison: sharedY,
            Message: passed
                ? $"CPU-flat vs GPU-flat training update parity passed within {tolerance}."
                : $"CPU-flat vs GPU-flat training update parity failed within {tolerance}.");
    }

    public static bool TryRunCpuGpuTrainingUpdateParity(
        out FlatPrmGpuTrainingParityResult result,
        float tolerance = 1e-5f,
        int? openClDeviceIndex = null)
    {
        try
        {
            result = RunCpuGpuTrainingUpdateParity(tolerance, openClDeviceIndex);
            return result.Passed;
        }
        catch (Exception ex)
        {
            result = new FlatPrmGpuTrainingParityResult(
                GpuAvailable: false,
                Passed: false,
                Device: null,
                TokenOffsetXComparison: null,
                TokenOffsetYComparison: null,
                SharedOffsetXComparison: null,
                SharedOffsetYComparison: null,
                Message: $"OpenCL GPU training update parity check was not run ({ex.GetType().Name}: {ex.Message}).");
            return false;
        }
    }

    public static FlatPrmGpuFullTrainingParityResult RunCpuGpuFullTrainingParity(
        float tolerance = 1e-5f,
        int? openClDeviceIndex = null)
    {
        var (config, geometry, inputs) = CreateDeterministicInputs();
        const float learningRate = 0.08f;
        float[] targets = [2.25f, 4.75f];

        float[] cpuTokenOffsetX = inputs.TokenOffsetX.ToArray();
        float[] cpuTokenOffsetY = inputs.TokenOffsetY.ToArray();
        float[] cpuSharedOffsetX = inputs.SharedOffsetX.ToArray();
        float[] cpuSharedOffsetY = inputs.SharedOffsetY.ToArray();
        float[] gpuTokenOffsetX = inputs.TokenOffsetX.ToArray();
        float[] gpuTokenOffsetY = inputs.TokenOffsetY.ToArray();
        float[] gpuSharedOffsetX = inputs.SharedOffsetX.ToArray();
        float[] gpuSharedOffsetY = inputs.SharedOffsetY.ToArray();

        FlatPrmGpuBallState[] cpuBalls = [];
        FlatPrmGpuBallState[] gpuBalls = [];
        int[] cpuContacts = [];
        int[] gpuContacts = [];
        FlatPrmGpuRunResult? run = null;

        for (int sample = 0; sample < targets.Length; sample++)
        {
            cpuBalls = CreateTrainingStates(inputs, sample);
            gpuBalls = CreateTrainingStates(inputs, sample);
            cpuContacts = new int[config.TotalRows * cpuBalls.Length];
            gpuContacts = new int[config.TotalRows * gpuBalls.Length];

            FlatPrmTrainingKernels.RunTrainingSample(
                config,
                geometry,
                cpuBalls,
                cpuTokenOffsetX,
                cpuTokenOffsetY,
                cpuSharedOffsetX,
                cpuSharedOffsetY,
                inputs.NailRadii,
                inputs.NailResistances,
                inputs.NailDensities,
                cpuContacts,
                targets[sample],
                learningRate);

            run = FlatPrmGpuBackend.RunTrainingSample(
                config,
                geometry,
                gpuBalls,
                gpuTokenOffsetX,
                gpuTokenOffsetY,
                gpuSharedOffsetX,
                gpuSharedOffsetY,
                inputs.NailRadii,
                inputs.NailResistances,
                inputs.NailDensities,
                gpuContacts,
                targets[sample],
                learningRate,
                openClDeviceIndex,
                allowCpuFallback: false);
        }

        float[] cpuPositions = cpuBalls.Select(b => b.Position).ToArray();
        float[] cpuVelocities = cpuBalls.Select(b => b.Velocity).ToArray();
        float[] gpuPositions = gpuBalls.Select(b => b.Position).ToArray();
        float[] gpuVelocities = gpuBalls.Select(b => b.Velocity).ToArray();
        var ballComparison = FlatPrmSelfCheck.CompareFinalState(gpuPositions, gpuVelocities, cpuPositions, cpuVelocities);
        var tokenX = CompareArrays(gpuTokenOffsetX, cpuTokenOffsetX);
        var tokenY = CompareArrays(gpuTokenOffsetY, cpuTokenOffsetY);
        var sharedX = CompareArrays(gpuSharedOffsetX, cpuSharedOffsetX);
        var sharedY = CompareArrays(gpuSharedOffsetY, cpuSharedOffsetY);
        bool activeStateMatches = cpuBalls.Select(b => (b.Active, b.Stuck)).SequenceEqual(gpuBalls.Select(b => (b.Active, b.Stuck)));
        bool contactStateMatches = cpuContacts.SequenceEqual(gpuContacts);
        bool passed = ballComparison.IsWithin(tolerance) &&
            tokenX.IsWithin(tolerance) &&
            tokenY.IsWithin(tolerance) &&
            sharedX.IsWithin(tolerance) &&
            sharedY.IsWithin(tolerance) &&
            activeStateMatches &&
            contactStateMatches;

        return new FlatPrmGpuFullTrainingParityResult(
            GpuAvailable: run?.UsedGpu == true,
            Passed: passed,
            Device: run?.Device,
            BallComparison: ballComparison,
            TokenOffsetXComparison: tokenX,
            TokenOffsetYComparison: tokenY,
            SharedOffsetXComparison: sharedX,
            SharedOffsetYComparison: sharedY,
            ActiveStateMatches: activeStateMatches,
            ContactStateMatches: contactStateMatches,
            Message: passed
                ? $"CPU-flat vs GPU-flat full training sample parity passed within {tolerance}."
                : $"CPU-flat vs GPU-flat full training sample parity failed within {tolerance}.");
    }

    public static bool TryRunCpuGpuFullTrainingParity(
        out FlatPrmGpuFullTrainingParityResult result,
        float tolerance = 1e-5f,
        int? openClDeviceIndex = null)
    {
        try
        {
            result = RunCpuGpuFullTrainingParity(tolerance, openClDeviceIndex);
            return result.Passed;
        }
        catch (Exception ex)
        {
            result = new FlatPrmGpuFullTrainingParityResult(
                GpuAvailable: false,
                Passed: false,
                Device: null,
                BallComparison: null,
                TokenOffsetXComparison: null,
                TokenOffsetYComparison: null,
                SharedOffsetXComparison: null,
                SharedOffsetYComparison: null,
                ActiveStateMatches: false,
                ContactStateMatches: false,
                Message: $"OpenCL GPU full training parity check was not run ({ex.GetType().Name}: {ex.Message}).");
            return false;
        }
    }

    private static (
        FlatPrmKernelConfig Config,
        FlatPrmRowGeometry Geometry,
        FlatPrmParityInputs Inputs) CreateDeterministicInputs()
    {
        var diamondConfig = new DiamondConfig
        {
            EntryWidth = 4f,
            MaxWidth = 8f,
            WideningRows = 1,
            NarrowingRows = 2,
            NailSpacing = 2f,
            DeflectionAlpha = 0.8f,
            DeflectionAlphaY = 0.15f,
            DeflectionIdfPower = 0.5f,
            SharedOffsetBlend = 0.25f,
            PredictionProbeTrainingWeight = 0.7f,
            DeltaTime = 0.1f,
            InputWindowSize = 2
        };

        const int vocabSize = 2;
        var geometry = FlatPrmRowGeometry.FromConfig(diamondConfig);
        var config = FlatPrmKernelConfig.FromConfig(diamondConfig, vocabSize, geometry.MaxColumns);

        var positions = new[] { 2.25f, 3.75f, 5.25f };
        var velocities = new[] { 0.05f, -0.04f, 0.02f };
        var masses = new[] { 1.0f, 1.8f, 0.6f };
        var relevanceWeights = new[] { 1.0f, 0.65f, 0.5f };
        var contextPositions = new[] { 0, 1, -1 };
        var tokenIds = new[] { 0, 1, -1 };

        int tokenOffsetLength = config.TotalRows * config.MaxColumns * config.TokenKeyCount;
        int sharedOffsetLength = config.TotalRows * config.MaxColumns * config.TokenSlotCount;
        int nailLength = config.TotalRows * config.MaxColumns;
        var tokenOffsetX = new float[tokenOffsetLength];
        var tokenOffsetY = new float[tokenOffsetLength];
        var sharedOffsetX = new float[sharedOffsetLength];
        var sharedOffsetY = new float[sharedOffsetLength];
        var nailRadii = new float[nailLength];
        var nailResistances = new float[nailLength];
        var nailDensities = new float[nailLength];

        for (int i = 0; i < tokenOffsetLength; i++)
        {
            tokenOffsetX[i] = ((i % 7) - 3) * 0.03125f;
            tokenOffsetY[i] = ((i % 5) - 2) * 0.0275f;
        }

        for (int i = 0; i < sharedOffsetLength; i++)
        {
            sharedOffsetX[i] = ((i % 3) - 1) * 0.01875f;
            sharedOffsetY[i] = ((i % 4) - 1.5f) * 0.01625f;
        }

        for (int i = 0; i < nailLength; i++)
        {
            nailRadii[i] = 0.45f + (i % 4) * 0.05f;
            nailResistances[i] = 0.35f + (i % 5) * 0.07f;
            nailDensities[i] = 0.9f + (i % 6) * 0.11f;
        }

        return (config, geometry, new FlatPrmParityInputs(
            positions,
            velocities,
            masses,
            relevanceWeights,
            contextPositions,
            tokenIds,
            tokenOffsetX,
            tokenOffsetY,
            sharedOffsetX,
            sharedOffsetY,
            nailRadii,
            nailResistances,
            nailDensities));
    }

    private static int[] CreateLastNailColumns(
        FlatPrmKernelConfig config,
        FlatPrmRowGeometry geometry,
        int row,
        ReadOnlySpan<float> positions)
    {
        var columns = new int[positions.Length];
        for (int i = 0; i < columns.Length; i++)
        {
            int col = geometry.NailColumn(positions[i], row);
            columns[i] = col >= 0 && col < geometry.RowNailCounts[row] ? col : -1;
        }

        return columns;
    }

    private static int[] CreateLastTokenIndices(
        FlatPrmKernelConfig config,
        ReadOnlySpan<int> contextPositions,
        ReadOnlySpan<int> tokenIds)
    {
        var tokenIndices = new int[tokenIds.Length];
        for (int i = 0; i < tokenIndices.Length; i++)
        {
            int slot = FlatPrmCpuKernels.TokenSlot(config, tokenIds[i]);
            tokenIndices[i] = FlatPrmCpuKernels.TokenIndex(config, contextPositions[i], slot);
        }

        return tokenIndices;
    }

    private static FlatPrmGpuBallState[] CreateTrainingStates(FlatPrmParityInputs inputs, int sample)
    {
        var states = new FlatPrmGpuBallState[inputs.Positions.Length];
        for (int i = 0; i < states.Length; i++)
        {
            states[i] = new FlatPrmGpuBallState(
                inputs.Positions[i] + sample * 0.125f,
                inputs.Velocities[i] - sample * 0.015f,
                inputs.Masses[i],
                inputs.RelevanceWeights[i],
                inputs.TokenIds[i],
                inputs.ContextPositions[i]);
        }

        return states;
    }

    private static FlatPrmArrayComparison CompareArrays(
        ReadOnlySpan<float> actual,
        ReadOnlySpan<float> expected)
    {
        int compared = Math.Min(actual.Length, expected.Length);
        float maxDelta = 0f;
        for (int i = 0; i < compared; i++)
            maxDelta = Math.Max(maxDelta, Math.Abs(actual[i] - expected[i]));

        return new FlatPrmArrayComparison(
            ComparedCount: compared,
            CountDelta: actual.Length - expected.Length,
            MaxDelta: maxDelta);
    }

    private sealed record FlatPrmParityInputs(
        float[] Positions,
        float[] Velocities,
        float[] Masses,
        float[] RelevanceWeights,
        int[] ContextPositions,
        int[] TokenIds,
        float[] TokenOffsetX,
        float[] TokenOffsetY,
        float[] SharedOffsetX,
        float[] SharedOffsetY,
        float[] NailRadii,
        float[] NailResistances,
        float[] NailDensities);
}
