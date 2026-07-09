using PRM.Core.Engine;
using PRM.Core.Engine.Flat;

namespace PRM.Core.Modes;

public sealed class GpuTrainingMode
{
    private readonly SpecialistRouter _router;
    private readonly FlatPrmGpuTrainingOptions _options;
    private bool _useCpuObjectFallback;
    private string _fallbackReason = "";

    public float LearningRate { get; set; } = 0.01f;
    public int EpochCount { get; set; } = 1;
    public float LrDecayPerEpoch { get; set; } = 1.0f;
    public ReplayCurriculumOptions ReplayCurriculum { get; } = new();
    public bool ReplayMisses { get => ReplayCurriculum.Enabled; set => ReplayCurriculum.Enabled = value; }
    public string LastExecutionMessage { get; private set; } = "";

    public GpuTrainingMode(SpecialistRouter router, FlatPrmGpuTrainingOptions? options = null)
    {
        _router = router;
        _options = options ?? new FlatPrmGpuTrainingOptions();
        PrepareExecutionPlan();
    }

    public IEnumerable<EpochMetrics> Run(
        IEnumerable<(int[] inputIds, int targetId)> dataset,
        Action<int, TrainStepResult>? onStep = null)
    {
        if (_useCpuObjectFallback)
        {
            LastExecutionMessage = _fallbackReason;
            var cpu = new TrainingMode(_router)
            {
                LearningRate = LearningRate,
                EpochCount = EpochCount,
                LrDecayPerEpoch = LrDecayPerEpoch
            };
            cpu.ReplayCurriculum.Enabled = ReplayCurriculum.Enabled;
            cpu.ReplayCurriculum.MaxReplayPasses = ReplayCurriculum.MaxReplayPasses;
            cpu.ReplayCurriculum.MaxReplayStepsPerEpoch = ReplayCurriculum.MaxReplayStepsPerEpoch;
            cpu.ReplayCurriculum.ReplayLearningRateScale = ReplayCurriculum.ReplayLearningRateScale;
            cpu.ReplayCurriculum.ReplayLearningRateDecayPerPass = ReplayCurriculum.ReplayLearningRateDecayPerPass;
            foreach (var metrics in cpu.Run(dataset, onStep))
                yield return metrics;
            yield break;
        }

        var sessions = _router.Specialists.ToDictionary(
            specialist => specialist,
            specialist => new FlatPrmGpuTrainingSession(specialist, _options));

        try
        {
            LastExecutionMessage = sessions.Count > 0
                ? sessions.First().Value.RunMessage
                : "No GPU specialists available.";

            float lr = LearningRate;
            for (int epoch = 0; epoch < EpochCount; epoch++)
            {
                yield return RunGpuEpoch(dataset, epoch, lr, onStep, sessions);
                lr *= LrDecayPerEpoch;
                _router.DecayNailStiffness(0.02f);
            }
        }
        finally
        {
            foreach (var session in sessions.Values)
                session.Dispose();
        }
    }

    private EpochMetrics RunGpuEpoch(
        IEnumerable<(int[] inputIds, int targetId)> dataset,
        int epoch,
        float learningRate,
        Action<int, TrainStepResult>? onStep,
        IReadOnlyDictionary<DiamondGrid, FlatPrmGpuTrainingSession> sessions)
    {
        var results = new List<TrainStepResult>();
        var misses = new List<ReplaySample>();

        foreach (var (inputIds, targetId) in dataset)
        {
            var (predicted, isCorrect, role, conf, message) = TrainGpu(inputIds, targetId, learningRate, sessions);
            LastExecutionMessage = message;
            int resultIndex = results.Count;
            results.Add(new TrainStepResult(predicted, targetId, isCorrect, role, conf, Retries: 0, Misses: isCorrect ? 0 : 1));
            if (!isCorrect)
                misses.Add(new ReplaySample(inputIds, targetId, resultIndex));
        }

        ReplayMissSamples(misses, results, learningRate, sessions);

        foreach (var result in results)
            onStep?.Invoke(epoch, result);

        return BuildMetrics(results);
    }

    private (int predicted, bool correct, string role, float confidence, string message) TrainGpu(
        int[] inputTokenIds,
        int targetTokenId,
        float learningRate,
        IReadOnlyDictionary<DiamondGrid, FlatPrmGpuTrainingSession> sessions)
    {
        DiamondGrid? winner = null;
        float bestConf = -1f;
        int bestPred = -1;
        string message = "";

        foreach (var spec in _router.Specialists)
        {
            var result = sessions.TryGetValue(spec, out var session)
                ? session.TrainSample(inputTokenIds, targetTokenId, learningRate)
                : FlatPrmGpuTrainingRunner.TrainSample(spec, inputTokenIds, targetTokenId, learningRate, _options);
            message = result.Run.Message;
            if (result.Confidence > bestConf)
            {
                bestConf = result.Confidence;
                bestPred = result.Predicted;
                winner = spec;
            }
        }

        return (bestPred, bestPred == targetTokenId, winner?.Role ?? "?", bestConf, message);
    }

    private void ReplayMissSamples(
        List<ReplaySample> misses,
        List<TrainStepResult> results,
        float learningRate,
        IReadOnlyDictionary<DiamondGrid, FlatPrmGpuTrainingSession> sessions)
    {
        if (!ReplayCurriculum.Enabled || misses.Count == 0 || ReplayCurriculum.MaxReplayPasses <= 0)
            return;

        float replayLearningRate = learningRate * MathF.Max(0f, ReplayCurriculum.ReplayLearningRateScale);
        int remainingBudget = ResolveReplayBudget(misses.Count);
        var currentMisses = misses;
        float replayDecay = MathF.Max(0f, ReplayCurriculum.ReplayLearningRateDecayPerPass);

        for (int pass = 0; pass < ReplayCurriculum.MaxReplayPasses && currentMisses.Count > 0 && remainingBudget > 0 && replayLearningRate > 0f; pass++)
        {
            var stillMissing = new List<ReplaySample>();
            foreach (var sample in currentMisses)
            {
                if (remainingBudget <= 0)
                {
                    stillMissing.Add(sample);
                    continue;
                }

                var (predicted, isCorrect, role, conf, message) = TrainGpu(sample.InputIds, sample.TargetId, replayLearningRate, sessions);
                LastExecutionMessage = message;
                remainingBudget--;

                var previous = results[sample.ResultIndex];
                results[sample.ResultIndex] = new TrainStepResult(
                    predicted,
                    sample.TargetId,
                    isCorrect,
                    role,
                    conf,
                    previous.Retries + 1,
                    isCorrect ? 0 : 1);

                if (!isCorrect)
                    stillMissing.Add(sample);
            }

            currentMisses = stillMissing;
            replayLearningRate *= replayDecay;
        }
    }

    private int ResolveReplayBudget(int missCount)
    {
        var passBudget = (long)missCount * Math.Max(0, ReplayCurriculum.MaxReplayPasses);
        if (passBudget <= 0) return 0;
        if (ReplayCurriculum.MaxReplayStepsPerEpoch <= 0)
            return passBudget > int.MaxValue ? int.MaxValue : (int)passBudget;

        return (int)Math.Min(passBudget, ReplayCurriculum.MaxReplayStepsPerEpoch);
    }

    private static EpochMetrics BuildMetrics(List<TrainStepResult> results)
    {
        int correct = 0, totalRetries = 0, totalMisses = 0;
        float totalConf = 0f;
        var roleCounts = new Dictionary<string, int>();

        foreach (var result in results)
        {
            if (result.Correct) correct++;
            totalConf += result.Confidence;
            totalRetries += result.Retries;
            totalMisses += result.Misses;
            roleCounts[result.Role] = roleCounts.GetValueOrDefault(result.Role) + 1;
        }

        int total = results.Count;
        return new EpochMetrics(
            total,
            correct,
            total > 0 ? (float)correct / total : 0f,
            total > 0 ? totalConf / total : 0f,
            roleCounts,
            totalMisses,
            totalRetries);
    }

    private void PrepareExecutionPlan()
    {
        foreach (var specialist in _router.Specialists)
        {
            if (!FlatPrmGpuTrainingRunner.IsSupported(specialist.Config, out var reason))
            {
                _useCpuObjectFallback = true;
                _fallbackReason = $"GPU training disabled for '{specialist.Role}': {reason} Falling back to CPU TrainingMode.";
                return;
            }
        }

        var devices = FlatPrmGpuBackend.DiscoverOpenClDevices();
        if (_options.OpenClDeviceIndex is int requestedIndex)
        {
            if (devices.Any(d => d.Index == requestedIndex))
                return;

            _useCpuObjectFallback = true;
            _fallbackReason = $"GPU training disabled: OpenCL device index {requestedIndex} was not found. Falling back to CPU TrainingMode.";
            return;
        }

        if (devices.Any(d => d.IsGpu))
            return;

        _useCpuObjectFallback = true;
        _fallbackReason = "GPU training disabled: no OpenCL GPU device found. Falling back to CPU TrainingMode.";
    }

    private sealed record ReplaySample(int[] InputIds, int TargetId, int ResultIndex);
}
