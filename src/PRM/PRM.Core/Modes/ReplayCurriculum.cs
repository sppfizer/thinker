using PRM.Core.Engine;

namespace PRM.Core.Modes;

/// <summary>Configures delayed hard-example replay after the full training pass.</summary>
public sealed class ReplayCurriculumOptions
{
    public bool Enabled { get; set; } = true;
    public int MaxReplayPasses { get; set; } = 5;
    /// <summary>0 means use the per-miss pass budget: initial misses × MaxReplayPasses.</summary>
    public int MaxReplayStepsPerEpoch { get; set; } = 0;
    public float ReplayLearningRateScale { get; set; } = 0.5f;
    public float ReplayLearningRateDecayPerPass { get; set; } = 0.75f;
}

internal static class ReplayCurriculumRunner
{
    public static EpochMetrics RunEpoch(
        SpecialistRouter router,
        IEnumerable<(int[] inputIds, int targetId)> dataset,
        int epoch,
        float learningRate,
        ReplayCurriculumOptions replay,
        Action<int, TrainStepResult>? onStep)
    {
        var results = new List<TrainStepResult>();
        var misses = new List<ReplaySample>();

        foreach (var (inputIds, targetId) in dataset)
        {
            var (predicted, isCorrect, role, conf) = router.Train(inputIds, targetId, learningRate);
            var resultIndex = results.Count;

            results.Add(new TrainStepResult(
                predicted,
                targetId,
                isCorrect,
                role,
                conf,
                Retries: 0,
                Misses: isCorrect ? 0 : 1));

            if (!isCorrect)
                misses.Add(new ReplaySample(inputIds, targetId, resultIndex));
        }

        ReplayMisses(router, misses, results, learningRate, replay);

        foreach (var result in results)
            onStep?.Invoke(epoch, result);

        return BuildMetrics(results);
    }

    private static void ReplayMisses(
        SpecialistRouter router,
        List<ReplaySample> misses,
        List<TrainStepResult> results,
        float learningRate,
        ReplayCurriculumOptions replay)
    {
        if (!replay.Enabled || misses.Count == 0 || replay.MaxReplayPasses <= 0)
            return;

        var replayLearningRate = learningRate * MathF.Max(0f, replay.ReplayLearningRateScale);
        if (replayLearningRate <= 0f)
            return;

        var remainingBudget = ResolveReplayBudget(misses.Count, replay);
        if (remainingBudget <= 0)
            return;

        var currentMisses = misses;
        var replayDecay = MathF.Max(0f, replay.ReplayLearningRateDecayPerPass);

        for (var pass = 0; pass < replay.MaxReplayPasses && currentMisses.Count > 0 && remainingBudget > 0 && replayLearningRate > 0f; pass++)
        {
            var stillMissing = new List<ReplaySample>();

            foreach (var sample in currentMisses)
            {
                if (remainingBudget <= 0)
                {
                    stillMissing.Add(sample);
                    continue;
                }

                var (predicted, isCorrect, role, conf) = router.Train(sample.InputIds, sample.TargetId, replayLearningRate);
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

    private static int ResolveReplayBudget(int missCount, ReplayCurriculumOptions replay)
    {
        var passBudget = (long)missCount * Math.Max(0, replay.MaxReplayPasses);
        if (passBudget <= 0)
            return 0;

        if (replay.MaxReplayStepsPerEpoch <= 0)
            return passBudget > int.MaxValue ? int.MaxValue : (int)passBudget;

        return (int)Math.Min(passBudget, replay.MaxReplayStepsPerEpoch);
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

        var total = results.Count;
        return new EpochMetrics(
            total,
            correct,
            total > 0 ? (float)correct / total : 0f,
            total > 0 ? totalConf / total : 0f,
            roleCounts,
            totalMisses,
            totalRetries);
    }

    private sealed record ReplaySample(int[] InputIds, int TargetId, int ResultIndex);
}
