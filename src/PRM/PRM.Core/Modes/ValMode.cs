using PRM.Core.Engine;
using PRM.Core.Modes;

namespace PRM.Core.Modes;

/// <summary>
/// VAL MODE — validation pass on a held-out dataset.
/// No nail updates. Reports accuracy, average confidence, per-role activation,
/// and a simple confusion summary (top mismatches).
/// </summary>
public class ValMode
{
    private readonly SpecialistRouter _router;
    public ValMode(SpecialistRouter router) => _router = router;

    public (EpochMetrics metrics, List<(int predicted, int target, string role)> mismatches) Run(
        IEnumerable<(int[] inputIds, int targetId)> dataset,
        int maxMismatches = 50)
    {
        int total = 0, correct = 0;
        float totalConf = 0f;
        var roleCounts  = new Dictionary<string, int>();
        var mismatches  = new List<(int, int, string)>();

        foreach (var (inputIds, targetId) in dataset)
        {
            var (predicted, role, conf) = _router.Predict(inputIds);
            bool isCorrect = predicted == targetId;

            total++;
            if (isCorrect) correct++;
            totalConf += conf;
            roleCounts[role] = roleCounts.GetValueOrDefault(role) + 1;

            if (!isCorrect && mismatches.Count < maxMismatches)
                mismatches.Add((predicted, targetId, role));
        }

        var metrics = new EpochMetrics(
            total, correct,
            total > 0 ? (float)correct / total : 0f,
            total > 0 ? totalConf / total        : 0f,
            roleCounts
        );

        return (metrics, mismatches);
    }
}
