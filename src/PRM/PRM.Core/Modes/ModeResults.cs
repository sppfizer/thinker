namespace PRM.Core.Modes;

/// <summary>Result of a single training step.</summary>
public record TrainStepResult(
    int    Predicted,
    int    Target,
    bool   Correct,
    string Role,
    float  Confidence
);

/// <summary>Result of a single test/val prediction.</summary>
public record EvalStepResult(
    int    Predicted,
    int    Target,
    bool   Correct,
    string Role,
    float  Confidence
);

/// <summary>Aggregated metrics over an epoch or dataset pass.</summary>
public record EpochMetrics(
    int   Total,
    int   Correct,
    float Accuracy,
    float AvgConfidence,
    Dictionary<string, int> RoleActivationCounts
)
{
    public override string ToString() =>
        $"Acc={Accuracy:P1}  AvgConf={AvgConfidence:F3}  Correct={Correct}/{Total}";
}
