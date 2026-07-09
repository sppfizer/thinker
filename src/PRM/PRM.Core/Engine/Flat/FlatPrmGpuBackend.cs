using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.OpenCL;

namespace PRM.Core.Engine.Flat;

public static class FlatPrmGpuBackend
{
    public static IReadOnlyList<FlatPrmGpuDeviceInfo> DiscoverOpenClDevices()
    {
        try
        {
            using var context = CreateOpenClContext();
            var devices = GetOpenClDevices(context);
            var infos = new FlatPrmGpuDeviceInfo[devices.Length];
            for (int i = 0; i < devices.Length; i++)
                infos[i] = FlatPrmGpuDeviceInfo.FromDevice(i, devices[i]);
            return infos;
        }
        catch
        {
            return [];
        }
    }

    public static FlatPrmGpuRunResult RunNailDeflectionIntegrationRows(
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
        Span<int> lastTokenIndices,
        int? openClDeviceIndex = null,
        bool allowCpuFallback = true)
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

        try
        {
            using var context = CreateOpenClContext();
            var devices = GetOpenClDevices(context);
            var selected = SelectDevice(devices, openClDeviceIndex);
            if (selected is null)
            {
                string message = devices.Length == 0
                    ? "No OpenCL devices found."
                    : "No OpenCL GPU devices found.";
                return FallBackToCpu(
                    message,
                    allowCpuFallback,
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
            }

            using var accelerator = selected.CreateCLAccelerator(context);
            RunOnAccelerator(
                accelerator,
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

            var info = FlatPrmGpuDeviceInfo.FromDevice(Array.IndexOf(devices, selected), selected);
            return new FlatPrmGpuRunResult(
                UsedGpu: info.IsGpu,
                UsedCpuFallback: false,
                Device: info,
                Message: $"Ran flat nail deflection + velocity integration on OpenCL device '{info.Name}' (gpu={info.IsGpu}).");
        }
        catch (Exception ex) when (allowCpuFallback)
        {
            return FallBackToCpu(
                $"OpenCL GPU execution failed ({ex.GetType().Name}: {ex.Message}).",
                allowCpuFallback,
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
        }
    }

    public static FlatPrmGpuRunResult ApplyTrainingUpdateRowSample(
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
        ReadOnlySpan<float> nailDensities,
        int? openClDeviceIndex = null,
        bool allowCpuFallback = true)
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

        try
        {
            using var context = CreateOpenClContext();
            var devices = GetOpenClDevices(context);
            var selected = SelectDevice(devices, openClDeviceIndex);
            if (selected is null)
            {
                string message = devices.Length == 0
                    ? "No OpenCL devices found."
                    : "No OpenCL GPU devices found.";
                return FallBackTrainingToCpu(
                    message,
                    allowCpuFallback,
                    config,
                    geometry,
                    row,
                    sampleIndex,
                    learningRate,
                    targetCentre,
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
            }

            using var accelerator = selected.CreateCLAccelerator(context);
            RunTrainingOnAccelerator(
                accelerator,
                config,
                geometry,
                row,
                sampleIndex,
                learningRate,
                targetCentre,
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

            var info = FlatPrmGpuDeviceInfo.FromDevice(Array.IndexOf(devices, selected), selected);
            return new FlatPrmGpuRunResult(
                UsedGpu: info.IsGpu,
                UsedCpuFallback: false,
                Device: info,
                Message: $"Ran flat training update row/sample on OpenCL device '{info.Name}' (gpu={info.IsGpu}).");
        }
        catch (Exception ex) when (allowCpuFallback)
        {
            return FallBackTrainingToCpu(
                $"OpenCL GPU training update failed ({ex.GetType().Name}: {ex.Message}).",
                allowCpuFallback,
                config,
                geometry,
                row,
                sampleIndex,
                learningRate,
                targetCentre,
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
        }
    }

    internal static FlatPrmGpuRunResult RunTrainingSample(
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
        float learningRate,
        int? openClDeviceIndex = null,
        bool allowCpuFallback = true)
    {
        try
        {
            using var context = CreateOpenClContext();
            var devices = GetOpenClDevices(context);
            var selected = SelectDevice(devices, openClDeviceIndex);
            if (selected is null)
            {
                string message = devices.Length == 0
                    ? "No OpenCL devices found."
                    : "No OpenCL GPU devices found.";
                return FallBackTrainingSampleToCpu(
                    message,
                    allowCpuFallback,
                    config,
                    geometry,
                    balls,
                    tokenOffsetX,
                    tokenOffsetY,
                    sharedOffsetX,
                    sharedOffsetY,
                    nailRadii,
                    nailResistances,
                    nailDensities,
                    contactColumns,
                    targetCentre,
                    learningRate);
            }

            using var accelerator = selected.CreateCLAccelerator(context);
            RunTrainingSampleOnAccelerator(
                accelerator,
                config,
                geometry,
                balls,
                tokenOffsetX,
                tokenOffsetY,
                sharedOffsetX,
                sharedOffsetY,
                nailRadii,
                nailResistances,
                nailDensities,
                contactColumns,
                targetCentre,
                learningRate);

            var info = FlatPrmGpuDeviceInfo.FromDevice(Array.IndexOf(devices, selected), selected);
            return new FlatPrmGpuRunResult(
                UsedGpu: info.IsGpu,
                UsedCpuFallback: false,
                Device: info,
                Message: $"Ran flat full training sample on OpenCL device '{info.Name}' (gpu={info.IsGpu}).");
        }
        catch (Exception ex) when (allowCpuFallback)
        {
            return FallBackTrainingSampleToCpu(
                $"OpenCL GPU training sample failed ({ex.GetType().Name}: {ex.Message}).",
                allowCpuFallback,
                config,
                geometry,
                balls,
                tokenOffsetX,
                tokenOffsetY,
                sharedOffsetX,
                sharedOffsetY,
                nailRadii,
                nailResistances,
                nailDensities,
                contactColumns,
                targetCentre,
                learningRate);
        }
    }

    private static Context CreateOpenClContext() =>
        Context.Create(builder => builder.OpenCL());

    private static CLDevice[] GetOpenClDevices(Context context)
    {
        var collection = context.GetCLDevices();
        var devices = new CLDevice[collection.Count];
        for (int i = 0; i < collection.Count; i++)
            devices[i] = collection[i];
        return devices;
    }

    private static CLDevice? SelectDevice(IReadOnlyList<CLDevice> devices, int? openClDeviceIndex)
    {
        if (openClDeviceIndex is int index)
            return index >= 0 && index < devices.Count ? devices[index] : null;

        return devices.FirstOrDefault(device =>
            (device.DeviceType & CLDeviceType.CL_DEVICE_TYPE_GPU) != 0);
    }

    private static FlatPrmGpuRunResult FallBackToCpu(
        string reason,
        bool allowCpuFallback,
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
        if (!allowCpuFallback)
            throw new InvalidOperationException(reason);

        FlatPrmCpuKernels.RunNailDeflectionIntegrationRows(
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

        return new FlatPrmGpuRunResult(
            UsedGpu: false,
            UsedCpuFallback: true,
            Device: null,
            Message: $"{reason} Used CPU flat fallback.");
    }

    private static FlatPrmGpuRunResult FallBackTrainingToCpu(
        string reason,
        bool allowCpuFallback,
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
        if (!allowCpuFallback)
            throw new InvalidOperationException(reason);

        FlatPrmCpuKernels.ApplyTrainingUpdateRowSample(
            config,
            geometry,
            row,
            sampleIndex,
            learningRate,
            targetCentre,
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

        return new FlatPrmGpuRunResult(
            UsedGpu: false,
            UsedCpuFallback: true,
            Device: null,
            Message: $"{reason} Used CPU flat fallback.");
    }

    private static FlatPrmGpuRunResult FallBackTrainingSampleToCpu(
        string reason,
        bool allowCpuFallback,
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
        if (!allowCpuFallback)
            throw new InvalidOperationException(reason);

        FlatPrmTrainingKernels.RunTrainingSample(
            config,
            geometry,
            balls,
            tokenOffsetX,
            tokenOffsetY,
            sharedOffsetX,
            sharedOffsetY,
            nailRadii,
            nailResistances,
            nailDensities,
            contactColumns,
            targetCentre,
            learningRate);

        return new FlatPrmGpuRunResult(
            UsedGpu: false,
            UsedCpuFallback: true,
            Device: null,
            Message: $"{reason} Used CPU flat fallback.");
    }

    private static void RunOnAccelerator(
        Accelerator accelerator,
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
        float[] positionArray = positions.ToArray();
        float[] velocityArray = velocities.ToArray();
        int[] lastState = new int[positions.Length * 2];

        using var positionsBuffer = accelerator.Allocate1D<float>(positionArray.Length);
        using var velocitiesBuffer = accelerator.Allocate1D<float>(velocityArray.Length);
        using var massesBuffer = accelerator.Allocate1D<float>(masses.Length);
        using var contextPositionsBuffer = accelerator.Allocate1D<int>(contextPositions.Length);
        using var tokenIdsBuffer = accelerator.Allocate1D<int>(tokenIds.Length);
        using var tokenOffsetXBuffer = accelerator.Allocate1D<float>(tokenOffsetX.Length);
        using var tokenOffsetYBuffer = accelerator.Allocate1D<float>(tokenOffsetY.Length);
        using var sharedOffsetXBuffer = accelerator.Allocate1D<float>(sharedOffsetX.Length);
        using var sharedOffsetYBuffer = accelerator.Allocate1D<float>(sharedOffsetY.Length);
        using var nailRadiiBuffer = accelerator.Allocate1D<float>(nailRadii.Length);
        using var lastStateBuffer = accelerator.Allocate1D<int>(lastState.Length);
        using var rowGeometryBuffer = accelerator.Allocate1D<FlatPrmGpuRowGeometry>(geometry.TotalRows);

        positionsBuffer.CopyFromCPU(positionArray);
        velocitiesBuffer.CopyFromCPU(velocityArray);
        massesBuffer.CopyFromCPU(masses.ToArray());
        contextPositionsBuffer.CopyFromCPU(contextPositions.ToArray());
        tokenIdsBuffer.CopyFromCPU(tokenIds.ToArray());
        tokenOffsetXBuffer.CopyFromCPU(tokenOffsetX.ToArray());
        tokenOffsetYBuffer.CopyFromCPU(tokenOffsetY.ToArray());
        sharedOffsetXBuffer.CopyFromCPU(sharedOffsetX.ToArray());
        sharedOffsetYBuffer.CopyFromCPU(sharedOffsetY.ToArray());
        nailRadiiBuffer.CopyFromCPU(nailRadii.ToArray());
        lastStateBuffer.CopyFromCPU(lastState);
        rowGeometryBuffer.CopyFromCPU(PackGeometry(geometry));

        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            FlatPrmGpuKernelConfig,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<FlatPrmGpuRowGeometry, Stride1D.Dense>>(
                FlatPrmGpuKernels.RunNailDeflectionIntegrationRows);

        kernel(
            positionArray.Length,
            new FlatPrmGpuKernelConfig(config, rowStart, rowCount),
            positionsBuffer.View,
            velocitiesBuffer.View,
            massesBuffer.View,
            contextPositionsBuffer.View,
            tokenIdsBuffer.View,
            tokenOffsetXBuffer.View,
            tokenOffsetYBuffer.View,
            sharedOffsetXBuffer.View,
            sharedOffsetYBuffer.View,
            nailRadiiBuffer.View,
            lastStateBuffer.View,
            rowGeometryBuffer.View);

        accelerator.Synchronize();

        positionsBuffer.CopyToCPU(positionArray);
        velocitiesBuffer.CopyToCPU(velocityArray);
        lastStateBuffer.CopyToCPU(lastState);

        positionArray.CopyTo(positions);
        velocityArray.CopyTo(velocities);
        lastState.AsSpan(0, positions.Length).CopyTo(lastNailColumns);
        lastState.AsSpan(positions.Length, positions.Length).CopyTo(lastTokenIndices);
    }

    private static void RunTrainingOnAccelerator(
        Accelerator accelerator,
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
        float[] tokenOffsetXArray = tokenOffsetX.ToArray();
        float[] tokenOffsetYArray = tokenOffsetY.ToArray();
        float[] sharedOffsetXArray = sharedOffsetX.ToArray();
        float[] sharedOffsetYArray = sharedOffsetY.ToArray();

        using var samplesBuffer = accelerator.Allocate1D<FlatPrmGpuTrainingSample>(positions.Length);
        using var tokenOffsetXBuffer = accelerator.Allocate1D<float>(tokenOffsetXArray.Length);
        using var tokenOffsetYBuffer = accelerator.Allocate1D<float>(tokenOffsetYArray.Length);
        using var sharedOffsetXBuffer = accelerator.Allocate1D<float>(sharedOffsetXArray.Length);
        using var sharedOffsetYBuffer = accelerator.Allocate1D<float>(sharedOffsetYArray.Length);
        using var nailPropertiesBuffer = accelerator.Allocate1D<FlatPrmGpuNailProperties>(nailRadii.Length);
        using var rowGeometryBuffer = accelerator.Allocate1D<FlatPrmGpuRowGeometry>(geometry.TotalRows);

        samplesBuffer.CopyFromCPU(PackTrainingSamples(
            positions,
            masses,
            relevanceWeights,
            tokenIds,
            lastNailColumns,
            lastTokenIndices));
        tokenOffsetXBuffer.CopyFromCPU(tokenOffsetXArray);
        tokenOffsetYBuffer.CopyFromCPU(tokenOffsetYArray);
        sharedOffsetXBuffer.CopyFromCPU(sharedOffsetXArray);
        sharedOffsetYBuffer.CopyFromCPU(sharedOffsetYArray);
        nailPropertiesBuffer.CopyFromCPU(PackNailProperties(nailRadii, nailResistances, nailDensities));
        rowGeometryBuffer.CopyFromCPU(PackGeometry(geometry));

        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            FlatPrmGpuKernelConfig,
            FlatPrmGpuTrainingConfig,
            ArrayView1D<FlatPrmGpuTrainingSample, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<FlatPrmGpuNailProperties, Stride1D.Dense>,
            ArrayView1D<FlatPrmGpuRowGeometry, Stride1D.Dense>>(
                FlatPrmGpuKernels.ApplyTrainingUpdateRowSample);

        kernel(
            1,
            new FlatPrmGpuKernelConfig(config, row, 1),
            new FlatPrmGpuTrainingConfig(row, sampleIndex, learningRate, targetCentre),
            samplesBuffer.View,
            tokenOffsetXBuffer.View,
            tokenOffsetYBuffer.View,
            sharedOffsetXBuffer.View,
            sharedOffsetYBuffer.View,
            nailPropertiesBuffer.View,
            rowGeometryBuffer.View);

        accelerator.Synchronize();

        tokenOffsetXBuffer.CopyToCPU(tokenOffsetXArray);
        tokenOffsetYBuffer.CopyToCPU(tokenOffsetYArray);
        sharedOffsetXBuffer.CopyToCPU(sharedOffsetXArray);
        sharedOffsetYBuffer.CopyToCPU(sharedOffsetYArray);

        tokenOffsetXArray.CopyTo(tokenOffsetX);
        tokenOffsetYArray.CopyTo(tokenOffsetY);
        sharedOffsetXArray.CopyTo(sharedOffsetX);
        sharedOffsetYArray.CopyTo(sharedOffsetY);
    }

    private static void RunTrainingSampleOnAccelerator(
        Accelerator accelerator,
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
        FlatPrmGpuBallState[] ballArray = balls.ToArray();
        float[] tokenOffsetXArray = tokenOffsetX.ToArray();
        float[] tokenOffsetYArray = tokenOffsetY.ToArray();
        float[] sharedOffsetXArray = sharedOffsetX.ToArray();
        float[] sharedOffsetYArray = sharedOffsetY.ToArray();
        int[] contactArray = new int[config.TotalRows * ballArray.Length];
        Array.Fill(contactArray, -1);

        using var ballsBuffer = accelerator.Allocate1D<FlatPrmGpuBallState>(ballArray.Length);
        using var tokenOffsetXBuffer = accelerator.Allocate1D<float>(tokenOffsetXArray.Length);
        using var tokenOffsetYBuffer = accelerator.Allocate1D<float>(tokenOffsetYArray.Length);
        using var sharedOffsetXBuffer = accelerator.Allocate1D<float>(sharedOffsetXArray.Length);
        using var sharedOffsetYBuffer = accelerator.Allocate1D<float>(sharedOffsetYArray.Length);
        using var nailPropertiesBuffer = accelerator.Allocate1D<FlatPrmGpuNailProperties>(nailRadii.Length);
        using var contactsBuffer = accelerator.Allocate1D<int>(contactArray.Length);
        using var rowGeometryBuffer = accelerator.Allocate1D<FlatPrmGpuRowGeometry>(geometry.TotalRows);

        ballsBuffer.CopyFromCPU(ballArray);
        tokenOffsetXBuffer.CopyFromCPU(tokenOffsetXArray);
        tokenOffsetYBuffer.CopyFromCPU(tokenOffsetYArray);
        sharedOffsetXBuffer.CopyFromCPU(sharedOffsetXArray);
        sharedOffsetYBuffer.CopyFromCPU(sharedOffsetYArray);
        nailPropertiesBuffer.CopyFromCPU(PackNailProperties(nailRadii, nailResistances, nailDensities));
        contactsBuffer.CopyFromCPU(contactArray);
        rowGeometryBuffer.CopyFromCPU(PackGeometry(geometry));

        var deflectKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            FlatPrmGpuKernelConfig,
            int,
            ArrayView1D<FlatPrmGpuBallState, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<FlatPrmGpuNailProperties, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<FlatPrmGpuRowGeometry, Stride1D.Dense>>(
                FlatPrmGpuKernels.ApplyTrainingDeflectionRow);

        var updateKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            FlatPrmGpuKernelConfig,
            int,
            float,
            float,
            ArrayView1D<FlatPrmGpuBallState, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<FlatPrmGpuNailProperties, Stride1D.Dense>,
            ArrayView1D<FlatPrmGpuRowGeometry, Stride1D.Dense>>(
                FlatPrmGpuKernels.ApplyTrainingUpdatesRow);

        var interactionKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            FlatPrmGpuKernelConfig,
            ArrayView1D<FlatPrmGpuBallState, Stride1D.Dense>>(
                FlatPrmGpuKernels.ApplyBallInteractions);

        var integrateKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            FlatPrmGpuKernelConfig,
            int,
            ArrayView1D<FlatPrmGpuBallState, Stride1D.Dense>,
            ArrayView1D<FlatPrmGpuRowGeometry, Stride1D.Dense>>(
                FlatPrmGpuKernels.IntegrateAndResolveBoundsRow);

        var gpuConfig = new FlatPrmGpuKernelConfig(config, 0, config.TotalRows);
        for (int row = 0; row < config.TotalRows; row++)
        {
            deflectKernel(
                ballArray.Length,
                gpuConfig,
                row,
                ballsBuffer.View,
                tokenOffsetXBuffer.View,
                tokenOffsetYBuffer.View,
                sharedOffsetXBuffer.View,
                sharedOffsetYBuffer.View,
                nailPropertiesBuffer.View,
                contactsBuffer.View,
                rowGeometryBuffer.View);

            interactionKernel(
                1,
                gpuConfig,
                ballsBuffer.View);

            updateKernel(
                1,
                gpuConfig,
                row,
                learningRate,
                targetCentre,
                ballsBuffer.View,
                tokenOffsetXBuffer.View,
                tokenOffsetYBuffer.View,
                sharedOffsetXBuffer.View,
                sharedOffsetYBuffer.View,
                nailPropertiesBuffer.View,
                rowGeometryBuffer.View);

            integrateKernel(
                ballArray.Length,
                gpuConfig,
                row,
                ballsBuffer.View,
                rowGeometryBuffer.View);
        }

        accelerator.Synchronize();

        ballsBuffer.CopyToCPU(ballArray);
        tokenOffsetXBuffer.CopyToCPU(tokenOffsetXArray);
        tokenOffsetYBuffer.CopyToCPU(tokenOffsetYArray);
        sharedOffsetXBuffer.CopyToCPU(sharedOffsetXArray);
        sharedOffsetYBuffer.CopyToCPU(sharedOffsetYArray);
        contactsBuffer.CopyToCPU(contactArray);

        ballArray.CopyTo(balls);
        tokenOffsetXArray.CopyTo(tokenOffsetX);
        tokenOffsetYArray.CopyTo(tokenOffsetY);
        sharedOffsetXArray.CopyTo(sharedOffsetX);
        sharedOffsetYArray.CopyTo(sharedOffsetY);
        contactArray.CopyTo(contactColumns);
    }

    private static FlatPrmGpuRowGeometry[] PackGeometry(FlatPrmRowGeometry geometry)
    {
        var packed = new FlatPrmGpuRowGeometry[geometry.TotalRows];
        for (int row = 0; row < packed.Length; row++)
        {
            packed[row] = new FlatPrmGpuRowGeometry(
                geometry.RowWidths[row],
                geometry.LeftBorders[row],
                geometry.RightBorders[row],
                geometry.RowNailCounts[row]);
        }

        return packed;
    }

    private static FlatPrmGpuTrainingSample[] PackTrainingSamples(
        ReadOnlySpan<float> positions,
        ReadOnlySpan<float> masses,
        ReadOnlySpan<float> relevanceWeights,
        ReadOnlySpan<int> tokenIds,
        ReadOnlySpan<int> lastNailColumns,
        ReadOnlySpan<int> lastTokenIndices)
    {
        var packed = new FlatPrmGpuTrainingSample[positions.Length];
        for (int i = 0; i < packed.Length; i++)
        {
            packed[i] = new FlatPrmGpuTrainingSample(
                positions[i],
                masses[i],
                relevanceWeights[i],
                tokenIds[i],
                lastNailColumns[i],
                lastTokenIndices[i]);
        }

        return packed;
    }

    private static FlatPrmGpuNailProperties[] PackNailProperties(
        ReadOnlySpan<float> radii,
        ReadOnlySpan<float> resistances,
        ReadOnlySpan<float> densities)
    {
        var packed = new FlatPrmGpuNailProperties[radii.Length];
        for (int i = 0; i < packed.Length; i++)
            packed[i] = new FlatPrmGpuNailProperties(radii[i], resistances[i], densities[i]);

        return packed;
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
