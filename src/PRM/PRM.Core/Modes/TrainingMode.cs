using PRM.Core.Engine;
using PRM.Core.Models;

namespace PRM.Core.Modes;

/// <summary>
/// TRAINING MODE — forward pass with live nail updates via magnetic force.
/// Balls fall through the diamond; nails nudge toward the correct output as each ball passes.
/// No backward pass. First pass trains every sample once; misses replay afterward at a reduced LR.
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
    public ReplayCurriculumOptions ReplayCurriculum { get; } = new();
    public bool  ReplayMisses { get => ReplayCurriculum.Enabled; set => ReplayCurriculum.Enabled = value; }
    public int   MaxReplayPasses { get => ReplayCurriculum.MaxReplayPasses; set => ReplayCurriculum.MaxReplayPasses = value; }
    public int   MaxReplayStepsPerEpoch { get => ReplayCurriculum.MaxReplayStepsPerEpoch; set => ReplayCurriculum.MaxReplayStepsPerEpoch = value; }
    public float ReplayLearningRateScale { get => ReplayCurriculum.ReplayLearningRateScale; set => ReplayCurriculum.ReplayLearningRateScale = value; }
    public float ReplayLearningRateDecayPerPass { get => ReplayCurriculum.ReplayLearningRateDecayPerPass; set => ReplayCurriculum.ReplayLearningRateDecayPerPass = value; }

    public TrainingMode(SpecialistRouter router) => _router = router;

    public IEnumerable<EpochMetrics> Run(
        IEnumerable<(int[] inputIds, int targetId)> dataset,
        Action<int, TrainStepResult>? onStep = null)
    {
        float lr = LearningRate;

        for (int epoch = 0; epoch < EpochCount; epoch++)
        {
            yield return ReplayCurriculumRunner.RunEpoch(_router, dataset, epoch, lr, ReplayCurriculum, onStep);

            // Apply learning rate decay after each epoch
            lr *= LrDecayPerEpoch;
            _router.DecayNailStiffness(0.02f);
        }
    }
}
