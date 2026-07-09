using PRM.Core.Engine;
using PRM.Core.Modes;

namespace PRM.Core.Modes;

/// <summary>
/// TUNE MODE — like training but with a lower learning rate and smaller dataset.
/// Used for fine-tuning a pre-trained specialist on a specific corpus subset.
/// Thick-nail (high diameter) routes are preserved; only thin nails adjust freely.
/// </summary>
public class TuneMode
{
    private readonly SpecialistRouter _router;

    public float LearningRate { get; set; } = 0.001f;   // 10× lower than training default
    public int   EpochCount   { get; set; } = 1;

    public TuneMode(SpecialistRouter router) => _router = router;

    public IEnumerable<EpochMetrics> Run(
        IEnumerable<(int[] inputIds, int targetId)> dataset,
        Action<int, TrainStepResult>? onStep = null)
    {
        for (int epoch = 0; epoch < EpochCount; epoch++)
        {
            int total = 0, correct = 0;
            float totalConf = 0f;
            int totalRetries = 0;
            int totalMisses = 0;
            var roleCounts = new Dictionary<string, int>();

            foreach (var (inputIds, targetId) in dataset)
            {
                // Same as training but with reduced LR — thick nails barely move
                var (predicted, isCorrect, role, conf) = _router.Train(inputIds, targetId, LearningRate);
                int retries = 0;
                int misses = isCorrect ? 0 : 1;

                while (!isCorrect && retries < 5)
                {
                    retries++;
                    var retried = _router.Train(inputIds, targetId, LearningRate);
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
        }
    }
}
