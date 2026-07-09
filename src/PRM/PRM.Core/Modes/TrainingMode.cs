using PRM.Core.Engine;
using PRM.Core.Models;

namespace PRM.Core.Modes;

/// <summary>
/// TRAINING MODE — forward pass with live nail updates via magnetic force.
/// Balls fall through the diamond; nails nudge toward the correct output as each ball passes.
/// No backward pass. No stored activations. One phase.
///
/// LrDecayPerEpoch: multiply LearningRate by this factor at the end of each epoch.
/// Set to 1.0 to disable decay.  0.97 gives smooth convergence over ~100 epochs.
/// </summary>
public class TrainingMode
{
    private readonly SpecialistRouter _router;

    public float LearningRate    { get; set; } = 0.01f;
    public int   EpochCount      { get; set; } = 1;
    public float LrDecayPerEpoch { get; set; } = 1.0f;  // multiply LR each epoch

    public TrainingMode(SpecialistRouter router) => _router = router;

    public IEnumerable<EpochMetrics> Run(
        IEnumerable<(int[] inputIds, int targetId)> dataset,
        Action<int, TrainStepResult>? onStep = null)
    {
        float lr = LearningRate;

        for (int epoch = 0; epoch < EpochCount; epoch++)
        {
            int total = 0, correct = 0;
            float totalConf = 0f;
            int totalRetries = 0;
            int totalMisses = 0;
            var roleCounts = new Dictionary<string, int>();

            foreach (var (inputIds, targetId) in dataset)
            {
                var (predicted, isCorrect, role, conf) = _router.Train(inputIds, targetId, lr);
                int retries = 0;
                int misses = isCorrect ? 0 : 1;

                while (!isCorrect && retries < 5)
                {
                    retries++;
                    var retried = _router.Train(inputIds, targetId, lr);
                    predicted = retried.predicted;
                    isCorrect = retried.correct;
                    role = retried.role;
                    conf = retried.confidence;
                    if (isCorrect) misses = 0;
                }

                total++;
                if (isCorrect) correct++;
                totalConf += conf;
                totalRetries += retries;
                totalMisses += misses;
                roleCounts[role] = roleCounts.GetValueOrDefault(role) + 1;

                var result = new TrainStepResult(predicted, targetId, isCorrect, role, conf, retries, misses);
                onStep?.Invoke(epoch, result);
            }

            yield return new EpochMetrics(
                total, correct,
                total > 0 ? (float)correct / total : 0f,
                total > 0 ? totalConf / total        : 0f,
                roleCounts,
                totalMisses,
                totalRetries
            );

            // Apply learning rate decay after each epoch
            lr *= LrDecayPerEpoch;
        }
    }
}
