using ILGPU.Runtime.OpenCL;

namespace PRM.Core.Engine.Flat;

public readonly record struct FlatPrmGpuDeviceInfo(
    int Index,
    string Name,
    string Vendor,
    string Platform,
    string DeviceType,
    long MemorySizeBytes,
    int MaxThreadsPerGroup,
    bool IsGpu)
{
    internal static FlatPrmGpuDeviceInfo FromDevice(int index, CLDevice device)
    {
        bool isGpu = (device.DeviceType & CLDeviceType.CL_DEVICE_TYPE_GPU) != 0;
        return new FlatPrmGpuDeviceInfo(
            index,
            device.Name,
            device.VendorName,
            device.PlatformName,
            device.DeviceType.ToString(),
            device.MemorySize,
            device.MaxNumThreadsPerGroup,
            isGpu);
    }
}

public sealed record FlatPrmGpuRunResult(
    bool UsedGpu,
    bool UsedCpuFallback,
    FlatPrmGpuDeviceInfo? Device,
    string Message);

public sealed record FlatPrmGpuParityResult(
    bool GpuAvailable,
    bool Passed,
    FlatPrmGpuDeviceInfo? Device,
    FlatPrmComparison? Comparison,
    string Message);

public readonly record struct FlatPrmArrayComparison(
    int ComparedCount,
    int CountDelta,
    float MaxDelta)
{
    public bool IsWithin(float tolerance) =>
        CountDelta == 0 &&
        MaxDelta <= tolerance;
}

public sealed record FlatPrmGpuTrainingParityResult(
    bool GpuAvailable,
    bool Passed,
    FlatPrmGpuDeviceInfo? Device,
    FlatPrmArrayComparison? TokenOffsetXComparison,
    FlatPrmArrayComparison? TokenOffsetYComparison,
    FlatPrmArrayComparison? SharedOffsetXComparison,
    FlatPrmArrayComparison? SharedOffsetYComparison,
    string Message);

public sealed record FlatPrmGpuFullTrainingParityResult(
    bool GpuAvailable,
    bool Passed,
    FlatPrmGpuDeviceInfo? Device,
    FlatPrmComparison? BallComparison,
    FlatPrmArrayComparison? TokenOffsetXComparison,
    FlatPrmArrayComparison? TokenOffsetYComparison,
    FlatPrmArrayComparison? SharedOffsetXComparison,
    FlatPrmArrayComparison? SharedOffsetYComparison,
    bool ActiveStateMatches,
    bool ContactStateMatches,
    string Message);
