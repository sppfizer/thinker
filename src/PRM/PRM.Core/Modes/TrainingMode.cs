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
            var roleCounts = new Dictionary<string, int>();

            foreach (var (inputIds, targetId) in dataset)
            {
                var (predicted, isCorrect, role, conf) =
                    _router.Train(inputIds, targetId, lr);

                total++;
                if (isCorrect) correct++;
                totalConf += conf;
                roleCounts[role] = roleCounts.GetValueOrDefault(role) + 1;

                var result = new TrainStepResult(predicted, targetId, isCorrect, role, conf);
                onStep?.Invoke(epoch, result);
            }

            yield return new EpochMetrics(
                total, correct,
                total > 0 ? (float)correct / total : 0f,
                total > 0 ? totalConf / total        : 0f,
                roleCounts
            );

            // Apply learning rate decay after each epoch
            lr *= LrDecayPerEpoch;
        }
    }
}

