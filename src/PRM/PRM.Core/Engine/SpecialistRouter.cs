using PRM.Core.Models;

namespace PRM.Core.Engine;

/// <summary>
/// Routes input tokens to the most appropriate specialist diamond.
/// For ≤3 specialists: parallel mode — run all, pick highest confidence.
/// For >3 specialists: use a separate routing diamond (set UseRouter = true).
/// </summary>
public class SpecialistRouter
{
    private readonly List<DiamondGrid> _specialists;
    private readonly bool              _parallel;

    public IReadOnlyList<DiamondGrid> Specialists => _specialists;

    public SpecialistRouter(IEnumerable<DiamondGrid> specialists)
    {
        _specialists = specialists.ToList();
        _parallel    = _specialists.Count <= 3;
    }

    /// <summary>Predict using best-fit specialist (inference mode).</summary>
    public (int tokenId, string role, float confidence) Predict(int[] inputTokenIds)
    {
        if (_parallel)
            return RunParallel(inputTokenIds, train: false, target: -1, lr: 0f);

        // With many specialists: use first as router placeholder (extend later)
        return RunParallel(inputTokenIds, train: false, target: -1, lr: 0f);
    }

    /// <summary>Train using best-fit specialist.</summary>
    public (int predicted, bool correct, string role, float confidence) Train(
        int[] inputTokenIds, int targetTokenId, float learningRate)
    {
        // All specialists train in parallel; each updates its own nails
        DiamondGrid? winner = null;
        float         bestConf = -1f;
        int           bestPred = -1;

        foreach (var spec in _specialists)
        {
            var (pred, correct, conf) = spec.Train(inputTokenIds, targetTokenId, learningRate);
            if (conf > bestConf)
            {
                bestConf = conf;
                bestPred = pred;
                winner   = spec;
            }
        }

        bool isCorrect = bestPred == targetTokenId;
        return (bestPred, isCorrect, winner?.Role ?? "?", bestConf);
    }

    // ── Private ───────────────────────────────────────────────────────────────

    private (int tokenId, string role, float confidence) RunParallel(
        int[] inputTokenIds, bool train, int target, float lr)
    {
        DiamondGrid? winner = null;
        float         bestConf = -1f;
        int           bestToken = -1;

        foreach (var spec in _specialists)
        {
            var (tokenId, conf) = spec.Predict(inputTokenIds);
            if (conf > bestConf) { bestConf = conf; bestToken = tokenId; winner = spec; }
        }

        return (bestToken, winner?.Role ?? "?", bestConf);
    }
}
