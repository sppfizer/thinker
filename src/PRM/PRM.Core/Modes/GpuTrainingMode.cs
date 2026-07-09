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
                foreach (var session in sessions.Values)
                    session.FlushOffsetsToGrid();
                _router.DecayNailStiffness(0.02f);
                foreach (var session in sessions.Values)
                    session.RefreshNailPropertiesFromGrid();
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
        int batchSize = Math.Max(1, _options.MiniBatchSize);
        var pending = new List<(int[] inputIds, int targetId)>(batchSize);

        foreach (var (inputIds, targetId) in dataset)
        {
            pending.Add((inputIds, targetId));
            if (pending.Count >= batchSize)
                FlushPending();
        }

        FlushPending();

        ReplayMissSamples(misses, results, learningRate, sessions);

        foreach (var result in results)
            onStep?.Invoke(epoch, result);

        return BuildMetrics(results);

        void FlushPending()
        {
            if (pending.Count == 0)
                return;

            var trained = TrainGpuBatch(pending, learningRate, sessions);
            for (int i = 0; i < trained.Count; i++)
            {
                var sample = pending[i];
                var (predicted, isCorrect, role, conf, message) = trained[i];
                LastExecutionMessage = message;
                int resultIndex = results.Count;
                results.Add(new TrainStepResult(predicted, sample.targetId, isCorrect, role, conf, Retries: 0, Misses: isCorrect ? 0 : 1));
                if (!isCorrect)
                    misses.Add(new ReplaySample(sample.inputIds, sample.targetId, resultIndex));
            }

            pending.Clear();
        }
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

    private List<(int predicted, bool correct, string role, float confidence, string message)> TrainGpuBatch(
        IReadOnlyList<(int[] inputIds, int targetId)> samples,
        float learningRate,
        IReadOnlyDictionary<DiamondGrid, FlatPrmGpuTrainingSession> sessions)
    {
        var bestPredictions = Enumerable.Repeat(-1, samples.Count).ToArray();
        var bestConfidences = Enumerable.Repeat(-1f, samples.Count).ToArray();
        var bestRoles = Enumerable.Repeat("?", samples.Count).ToArray();
        string message = "";

        foreach (var spec in _router.Specialists)
        {
            FlatPrmGpuTrainingSampleResult[] specResults;
            if (sessions.TryGetValue(spec, out var session))
            {
                specResults = session.TrainBatch(samples, learningRate);
            }
            else
            {
                specResults = new FlatPrmGpuTrainingSampleResult[samples.Count];
                for (int i = 0; i < samples.Count; i++)
                    specResults[i] = FlatPrmGpuTrainingRunner.TrainSample(spec, samples[i].inputIds, samples[i].targetId, learningRate, _options);
            }

            for (int i = 0; i < specResults.Length; i++)
            {
                var result = specResults[i];
                message = result.Run.Message;
                if (result.Confidence > bestConfidences[i])
                {
                    bestConfidences[i] = result.Confidence;
                    bestPredictions[i] = result.Predicted;
                    bestRoles[i] = spec.Role;
                }
            }
        }

        var trained = new List<(int predicted, bool correct, string role, float confidence, string message)>(samples.Count);
        for (int i = 0; i < samples.Count; i++)
        {
            int targetId = samples[i].targetId;
            int predicted = bestPredictions[i];
            trained.Add((predicted, predicted == targetId, bestRoles[i], bestConfidences[i], message));
        }

        return trained;
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
            int batchSize = Math.Max(1, _options.MiniBatchSize);
            for (int start = 0; start < currentMisses.Count && remainingBudget > 0; start += batchSize)
            {
                int count = Math.Min(Math.Min(batchSize, currentMisses.Count - start), remainingBudget);
                var replayBatch = currentMisses
                    .Skip(start)
                    .Take(count)
                    .Select(sample => (sample.InputIds, sample.TargetId))
                    .ToList();
                var trained = TrainGpuBatch(replayBatch, replayLearningRate, sessions);

                for (int i = 0; i < trained.Count; i++)
                {
                    var sample = currentMisses[start + i];
                    var (predicted, isCorrect, role, conf, message) = trained[i];
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
