using PRM.Core.Engine;
using PRM.Core.Models;
using PRM.Core.Modes;

namespace PRM.Core.Modes;

/// <summary>
/// TRAINING MODE — forward pass with live nail updates via magnetic force.
/// Balls fall through the diamond; nails nudge toward the correct output as each ball passes.
/// No backward pass. No stored activations. One phase.
/// </summary>
public class TrainingMode
{
    private readonly SpecialistRouter _router;

    public float LearningRate { get; set; } = 0.01f;
    public int   EpochCount   { get; set; } = 1;

    public TrainingMode(SpecialistRouter router) => _router = router;

    public IEnumerable<EpochMetrics> Run(
        IEnumerable<(int[] inputIds, int targetId)> dataset,
        Action<int, TrainStepResult>? onStep = null)
    {
        for (int epoch = 0; epoch < EpochCount; epoch++)
        {
            int total = 0, correct = 0;
            float totalConf = 0f;
            var roleCounts = new Dictionary<string, int>();

            foreach (var (inputIds, targetId) in dataset)
            {
                var (predicted, isCorrect, role, conf) =
                    _router.Train(inputIds, targetId, LearningRate);

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
        }
    }
}
