using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.OpenCL;
using PRM.Core.Models;

namespace PRM.Core.Engine.Flat;

internal sealed class FlatPrmGpuTrainingSession : IDisposable
{
    private readonly DiamondGrid _grid;
    private readonly Context _context;
    private readonly Accelerator _accelerator;
    private readonly FlatPrmKernelConfig _config;
    private readonly FlatPrmGpuKernelConfig _gpuConfig;
    private readonly FlatPrmRowGeometry _geometry;
    private readonly FlatPrmGpuRunResult _run;
    private readonly int _maxBalls;
    private readonly int _tokenRows;
    private readonly int _tokenColumns;
    private readonly int _tokenDepth;
    private readonly int _sharedRows;
    private readonly int _sharedColumns;
    private readonly int _sharedDepth;
    private readonly FlatPrmGpuBallState[] _ballArray;
    private readonly int[] _contactArray;
    private readonly FlatPrmGpuNailProperties[] _nailPropertiesArray;
    private readonly MemoryBuffer1D<FlatPrmGpuBallState, Stride1D.Dense> _ballsBuffer;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _tokenOffsetXBuffer;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _tokenOffsetYBuffer;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _sharedOffsetXBuffer;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _sharedOffsetYBuffer;
    private readonly MemoryBuffer1D<FlatPrmGpuNailProperties, Stride1D.Dense> _nailPropertiesBuffer;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _contactsBuffer;
    private readonly MemoryBuffer1D<FlatPrmGpuRowGeometry, Stride1D.Dense> _rowGeometryBuffer;
    private readonly Action<
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
        ArrayView1D<FlatPrmGpuRowGeometry, Stride1D.Dense>> _deflectKernel;
    private readonly Action<
        Index1D,
        FlatPrmGpuKernelConfig,
        ArrayView1D<FlatPrmGpuBallState, Stride1D.Dense>> _interactionKernel;
    private readonly Action<
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
        ArrayView1D<FlatPrmGpuRowGeometry, Stride1D.Dense>> _updateKernel;
    private readonly Action<
        Index1D,
        FlatPrmGpuKernelConfig,
        int,
        ArrayView1D<FlatPrmGpuBallState, Stride1D.Dense>,
        ArrayView1D<FlatPrmGpuRowGeometry, Stride1D.Dense>> _integrateKernel;
    private bool _offsetsDirty;

    public string RunMessage => _run.Message;

    public FlatPrmGpuTrainingSession(DiamondGrid grid, FlatPrmGpuTrainingOptions options)
    {
        if (!FlatPrmGpuTrainingRunner.IsSupported(grid.Config, out var reason))
            throw new NotSupportedException(reason);

        _grid = grid;
        _geometry = FlatPrmRowGeometry.FromConfig(grid.Config, grid.Nails.GetLength(1));
        _config = FlatPrmKernelConfig.FromConfig(grid.Config, grid.Vocab.Length, _geometry.MaxColumns);
        _gpuConfig = new FlatPrmGpuKernelConfig(_config, 0, _config.TotalRows);
        _maxBalls = Math.Max(grid.Config.InputWindowSize + 1, 1);
        _ballArray = new FlatPrmGpuBallState[_maxBalls];
        _contactArray = new int[_config.TotalRows * _maxBalls];
        _nailPropertiesArray = new FlatPrmGpuNailProperties[_config.TotalRows * _config.MaxColumns];

        var tokenOffX = grid.Simulator.GetTokenOffX();
        var tokenOffY = grid.Simulator.GetTokenOffY();
        var sharedOffX = grid.Simulator.GetSharedOffX();
        var sharedOffY = grid.Simulator.GetSharedOffY();
        _tokenRows = tokenOffX.GetLength(0);
        _tokenColumns = tokenOffX.GetLength(1);
        _tokenDepth = tokenOffX.GetLength(2);
        _sharedRows = sharedOffX.GetLength(0);
        _sharedColumns = sharedOffX.GetLength(1);
        _sharedDepth = sharedOffX.GetLength(2);

        _context = Context.Create(builder => builder.OpenCL());
        var devices = GetOpenClDevices(_context);
        var selected = SelectDevice(devices, options.OpenClDeviceIndex);
        if (selected is null)
        {
            _context.Dispose();
            string message = devices.Length == 0
                ? "No OpenCL devices found."
                : "No OpenCL GPU devices found.";
            throw new InvalidOperationException(message);
        }

        _accelerator = selected.CreateCLAccelerator(_context);
        var deviceInfo = FlatPrmGpuDeviceInfo.FromDevice(Array.IndexOf(devices, selected), selected);
        _run = new FlatPrmGpuRunResult(
            UsedGpu: deviceInfo.IsGpu,
            UsedCpuFallback: false,
            Device: deviceInfo,
            Message: $"Ran persistent flat GPU training on OpenCL device '{deviceInfo.Name}' (gpu={deviceInfo.IsGpu}).");

        _ballsBuffer = _accelerator.Allocate1D<FlatPrmGpuBallState>(_ballArray.Length);
        _tokenOffsetXBuffer = _accelerator.Allocate1D<float>(_tokenRows * _tokenColumns * _tokenDepth);
        _tokenOffsetYBuffer = _accelerator.Allocate1D<float>(_tokenRows * _tokenColumns * _tokenDepth);
        _sharedOffsetXBuffer = _accelerator.Allocate1D<float>(_sharedRows * _sharedColumns * _sharedDepth);
        _sharedOffsetYBuffer = _accelerator.Allocate1D<float>(_sharedRows * _sharedColumns * _sharedDepth);
        _nailPropertiesBuffer = _accelerator.Allocate1D<FlatPrmGpuNailProperties>(_nailPropertiesArray.Length);
        _contactsBuffer = _accelerator.Allocate1D<int>(_contactArray.Length);
        _rowGeometryBuffer = _accelerator.Allocate1D<FlatPrmGpuRowGeometry>(_geometry.TotalRows);

        _tokenOffsetXBuffer.CopyFromCPU(FlatPrmArrayPacking.Flatten(tokenOffX));
        _tokenOffsetYBuffer.CopyFromCPU(FlatPrmArrayPacking.Flatten(tokenOffY));
        _sharedOffsetXBuffer.CopyFromCPU(FlatPrmArrayPacking.Flatten(sharedOffX));
        _sharedOffsetYBuffer.CopyFromCPU(FlatPrmArrayPacking.Flatten(sharedOffY));
        _rowGeometryBuffer.CopyFromCPU(PackGeometry(_geometry));

        _deflectKernel = _accelerator.LoadAutoGroupedStreamKernel<
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

        _interactionKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            FlatPrmGpuKernelConfig,
            ArrayView1D<FlatPrmGpuBallState, Stride1D.Dense>>(
                FlatPrmGpuKernels.ApplyBallInteractions);

        _updateKernel = _accelerator.LoadAutoGroupedStreamKernel<
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

        _integrateKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            FlatPrmGpuKernelConfig,
            int,
            ArrayView1D<FlatPrmGpuBallState, Stride1D.Dense>,
            ArrayView1D<FlatPrmGpuRowGeometry, Stride1D.Dense>>(
                FlatPrmGpuKernels.IntegrateAndResolveBoundsRow);
    }

    public FlatPrmGpuTrainingSampleResult TrainSample(
        int[] inputTokenIds,
        int targetTokenId,
        float learningRate)
    {
        var allBalls = _grid.CreateBallsForFlatTraining(inputTokenIds);
        if (allBalls.Count > _maxBalls)
            throw new InvalidOperationException($"GPU training sample has {allBalls.Count} balls but session capacity is {_maxBalls}.");

        PackBallStates(allBalls);
        Array.Fill(_contactArray, -1);
        RefreshNailProperties();

        _ballsBuffer.CopyFromCPU(_ballArray);
        _contactsBuffer.CopyFromCPU(_contactArray);
        _nailPropertiesBuffer.CopyFromCPU(_nailPropertiesArray);

        float targetCentre = _grid.SlotCentreForFlatTraining(targetTokenId);
        for (int row = 0; row < _config.TotalRows; row++)
        {
            _deflectKernel(
                _ballArray.Length,
                _gpuConfig,
                row,
                _ballsBuffer.View,
                _tokenOffsetXBuffer.View,
                _tokenOffsetYBuffer.View,
                _sharedOffsetXBuffer.View,
                _sharedOffsetYBuffer.View,
                _nailPropertiesBuffer.View,
                _contactsBuffer.View,
                _rowGeometryBuffer.View);

            _interactionKernel(
                1,
                _gpuConfig,
                _ballsBuffer.View);

            _updateKernel(
                1,
                _gpuConfig,
                row,
                learningRate,
                targetCentre,
                _ballsBuffer.View,
                _tokenOffsetXBuffer.View,
                _tokenOffsetYBuffer.View,
                _sharedOffsetXBuffer.View,
                _sharedOffsetYBuffer.View,
                _nailPropertiesBuffer.View,
                _rowGeometryBuffer.View);

            _integrateKernel(
                _ballArray.Length,
                _gpuConfig,
                row,
                _ballsBuffer.View,
                _rowGeometryBuffer.View);
        }

        _accelerator.Synchronize();

        _ballsBuffer.CopyToCPU(_ballArray);
        _contactsBuffer.CopyToCPU(_contactArray);
        _offsetsDirty = true;

        var survivors = BuildSurvivorBalls(_ballArray.AsSpan(0, allBalls.Count), _contactArray, _maxBalls, _config.TotalRows);
        var (predicted, confidence) = _grid.ScoreFlatTraining(survivors, allBalls);
        bool correct = predicted == targetTokenId;
        _grid.ApplyFlatPostTrainingAdjustment(survivors, targetCentre, learningRate, correct);

        return new FlatPrmGpuTrainingSampleResult(predicted, correct, confidence, _run);
    }

    public void FlushOffsetsToGrid()
    {
        if (!_offsetsDirty)
            return;

        var tokenOffX = new float[_tokenRows * _tokenColumns * _tokenDepth];
        var tokenOffY = new float[_tokenRows * _tokenColumns * _tokenDepth];
        var sharedOffX = new float[_sharedRows * _sharedColumns * _sharedDepth];
        var sharedOffY = new float[_sharedRows * _sharedColumns * _sharedDepth];

        _tokenOffsetXBuffer.CopyToCPU(tokenOffX);
        _tokenOffsetYBuffer.CopyToCPU(tokenOffY);
        _sharedOffsetXBuffer.CopyToCPU(sharedOffX);
        _sharedOffsetYBuffer.CopyToCPU(sharedOffY);

        _grid.Simulator.SetTokenOffsets(
            FlatPrmArrayPacking.Unflatten(tokenOffX, _tokenRows, _tokenColumns, _tokenDepth),
            FlatPrmArrayPacking.Unflatten(tokenOffY, _tokenRows, _tokenColumns, _tokenDepth));
        _grid.Simulator.SetSharedOffsets(
            FlatPrmArrayPacking.Unflatten(sharedOffX, _sharedRows, _sharedColumns, _sharedDepth),
            FlatPrmArrayPacking.Unflatten(sharedOffY, _sharedRows, _sharedColumns, _sharedDepth));

        _offsetsDirty = false;
    }

    public void Dispose()
    {
        FlushOffsetsToGrid();
        _rowGeometryBuffer.Dispose();
        _contactsBuffer.Dispose();
        _nailPropertiesBuffer.Dispose();
        _sharedOffsetYBuffer.Dispose();
        _sharedOffsetXBuffer.Dispose();
        _tokenOffsetYBuffer.Dispose();
        _tokenOffsetXBuffer.Dispose();
        _ballsBuffer.Dispose();
        _accelerator.Dispose();
        _context.Dispose();
    }

    private void PackBallStates(IReadOnlyList<Ball> balls)
    {
        Array.Fill(_ballArray, new FlatPrmGpuBallState(0f, 0f, 0f, 0f, 0, 0, active: 0));
        for (int i = 0; i < balls.Count; i++)
        {
            var ball = balls[i];
            _ballArray[i] = new FlatPrmGpuBallState(
                ball.Position,
                ball.Velocity,
                ball.Mass,
                ball.RelevanceWeight,
                ball.TokenId,
                ball.ContextPosition,
                active: ball.Active ? 1 : 0,
                stuck: ball.Stuck ? 1 : 0);
        }
    }

    private void RefreshNailProperties()
    {
        for (int row = 0; row < _config.TotalRows; row++)
        for (int col = 0; col < _config.MaxColumns; col++)
        {
            var nail = _grid.Nails[row, col];
            _nailPropertiesArray[row * _config.MaxColumns + col] =
                new FlatPrmGpuNailProperties(nail.Radius, nail.Resistance, nail.Density);
        }
    }

    private static List<Ball> BuildSurvivorBalls(
        ReadOnlySpan<FlatPrmGpuBallState> states,
        ReadOnlySpan<int> contactColumns,
        int contactStride,
        int totalRows)
    {
        var survivors = new List<Ball>();
        for (int i = 0; i < states.Length; i++)
        {
            var state = states[i];
            if (state.Active == 0)
                continue;

            var ball = new Ball(
                state.TokenId,
                state.Position,
                state.Mass,
                state.ContextPosition,
                state.RelevanceWeight)
            {
                Velocity = state.Velocity,
                Stuck = state.Stuck != 0
            };

            for (int row = 0; row < totalRows; row++)
            {
                int col = contactColumns[row * contactStride + i];
                if (col >= 0)
                {
                    int nailId = row * 10_000 + col;
                    if (ball.ContactNailIds.Count == 0 || ball.ContactNailIds[^1] != nailId)
                        ball.ContactNailIds.Add(nailId);
                }
            }

            survivors.Add(ball);
        }

        return survivors;
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
}
