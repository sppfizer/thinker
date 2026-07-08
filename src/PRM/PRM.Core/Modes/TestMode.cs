using PRM.Core.Engine;
using PRM.Core.Modes;

namespace PRM.Core.Modes;

/// <summary>
/// TEST MODE — forward pass only, no nail updates, no magnet.
/// Measures next-token accuracy and ball-mass retention confidence.
/// </summary>
public class TestMode
{
    private readonly SpecialistRouter _router;
    public TestMode(SpecialistRouter router) => _router = router;

    public EpochMetrics Run(
        IEnumerable<(int[] inputIds, int targetId)> dataset,
        Action<EvalStepResult>? onStep = null)
    {
        int total = 0, correct = 0;
        float totalConf = 0f;
        var roleCounts = new Dictionary<string, int>();

        foreach (var (inputIds, targetId) in dataset)
        {
            var (predicted, role, conf) = _router.Predict(inputIds);
            bool isCorrect = predicted == targetId;

            total++;
            if (isCorrect) correct++;
            totalConf += conf;
            roleCounts[role] = roleCounts.GetValueOrDefault(role) + 1;

            onStep?.Invoke(new EvalStepResult(predicted, targetId, isCorrect, role, conf));
        }

        return new EpochMetrics(
            total, correct,
            total > 0 ? (float)correct / total : 0f,
            total > 0 ? totalConf / total        : 0f,
            roleCounts
        );
    }
}
