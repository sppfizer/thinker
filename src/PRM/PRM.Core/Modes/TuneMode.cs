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
    public ReplayCurriculumOptions ReplayCurriculum { get; } = new();
    public bool  ReplayMisses { get => ReplayCurriculum.Enabled; set => ReplayCurriculum.Enabled = value; }
    public int   MaxReplayPasses { get => ReplayCurriculum.MaxReplayPasses; set => ReplayCurriculum.MaxReplayPasses = value; }
    public int   MaxReplayStepsPerEpoch { get => ReplayCurriculum.MaxReplayStepsPerEpoch; set => ReplayCurriculum.MaxReplayStepsPerEpoch = value; }
    public float ReplayLearningRateScale { get => ReplayCurriculum.ReplayLearningRateScale; set => ReplayCurriculum.ReplayLearningRateScale = value; }
    public float ReplayLearningRateDecayPerPass { get => ReplayCurriculum.ReplayLearningRateDecayPerPass; set => ReplayCurriculum.ReplayLearningRateDecayPerPass = value; }

    public TuneMode(SpecialistRouter router) => _router = router;

    public IEnumerable<EpochMetrics> Run(
        IEnumerable<(int[] inputIds, int targetId)> dataset,
        Action<int, TrainStepResult>? onStep = null)
    {
        for (int epoch = 0; epoch < EpochCount; epoch++)
        {
            // Same as training but with reduced LR — thick nails barely move
            yield return ReplayCurriculumRunner.RunEpoch(_router, dataset, epoch, LearningRate, ReplayCurriculum, onStep);
        }
    }
}
