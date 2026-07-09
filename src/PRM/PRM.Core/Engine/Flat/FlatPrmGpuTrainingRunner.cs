using PRM.Core.Models;

namespace PRM.Core.Engine.Flat;

public sealed record FlatPrmGpuTrainingOptions(
    int? OpenClDeviceIndex = null,
    bool AllowFlatCpuFallback = true);

public sealed record FlatPrmGpuTrainingSampleResult(
    int Predicted,
    bool Correct,
    float Confidence,
    FlatPrmGpuRunResult Run);

public static class FlatPrmGpuTrainingRunner
{
    public static bool IsSupported(DiamondConfig config, out string reason)
    {
        if (config.ContextSummaryBallCount > 0)
        {
            reason = "Context summary balls are not implemented in the GPU training path.";
            return false;
        }

        if (config.DownstreamNailInfluence > 0f && config.DownstreamNailInfluenceRows > 0)
        {
            reason = "Downstream nail influence is not implemented in the GPU training path.";
            return false;
        }

        reason = "Supported GPU training subset: flat nail deflection, gravity/collision interactions, velocity integration, bounds/stuck resolution, live token/shared offset updates, scoring, replay, and CPU post-pass reinforce/soften.";
        return true;
    }

    internal static FlatPrmGpuTrainingSampleResult TrainSample(
        DiamondGrid grid,
        int[] inputTokenIds,
        int targetTokenId,
        float learningRate,
        FlatPrmGpuTrainingOptions? options = null)
    {
        options ??= new FlatPrmGpuTrainingOptions();
        if (!IsSupported(grid.Config, out var reason))
            throw new NotSupportedException(reason);

        var allBalls = grid.CreateBallsForFlatTraining(inputTokenIds);
        var states = PackBallStates(allBalls);
        var contactColumns = new int[grid.Config.TotalRows * states.Length];
        Array.Fill(contactColumns, -1);

        var simulator = grid.Simulator;
        var tokenOffXShape = simulator.GetTokenOffX();
        var tokenOffYShape = simulator.GetTokenOffY();
        var sharedOffXShape = simulator.GetSharedOffX();
        var sharedOffYShape = simulator.GetSharedOffY();

        var tokenOffX = FlatPrmArrayPacking.Flatten(tokenOffXShape);
        var tokenOffY = FlatPrmArrayPacking.Flatten(tokenOffYShape);
        var sharedOffX = FlatPrmArrayPacking.Flatten(sharedOffXShape);
        var sharedOffY = FlatPrmArrayPacking.Flatten(sharedOffYShape);
        var geometry = FlatPrmRowGeometry.FromConfig(grid.Config, grid.Nails.GetLength(1));
        var config = FlatPrmKernelConfig.FromConfig(grid.Config, grid.Vocab.Length, geometry.MaxColumns);
        var nailRadii = FlatPrmArrayPacking.FlattenNailRadii(grid.Nails, config.TotalRows, config.MaxColumns);
        var nailResistances = FlatPrmArrayPacking.FlattenNailResistances(grid.Nails, config.TotalRows, config.MaxColumns);
        var nailDensities = FlatPrmArrayPacking.FlattenNailDensities(grid.Nails, config.TotalRows, config.MaxColumns);
        float targetCentre = grid.SlotCentreForFlatTraining(targetTokenId);

        var run = FlatPrmGpuBackend.RunTrainingSample(
            config,
            geometry,
            states,
            tokenOffX,
            tokenOffY,
            sharedOffX,
            sharedOffY,
            nailRadii,
            nailResistances,
            nailDensities,
            contactColumns,
            targetCentre,
            learningRate,
            options.OpenClDeviceIndex,
            options.AllowFlatCpuFallback);

        simulator.SetTokenOffsets(
            FlatPrmArrayPacking.Unflatten(
                tokenOffX,
                tokenOffXShape.GetLength(0),
                tokenOffXShape.GetLength(1),
                tokenOffXShape.GetLength(2)),
            FlatPrmArrayPacking.Unflatten(
                tokenOffY,
                tokenOffYShape.GetLength(0),
                tokenOffYShape.GetLength(1),
                tokenOffYShape.GetLength(2)));

        simulator.SetSharedOffsets(
            FlatPrmArrayPacking.Unflatten(
                sharedOffX,
                sharedOffXShape.GetLength(0),
                sharedOffXShape.GetLength(1),
                sharedOffXShape.GetLength(2)),
            FlatPrmArrayPacking.Unflatten(
                sharedOffY,
                sharedOffYShape.GetLength(0),
                sharedOffYShape.GetLength(1),
                sharedOffYShape.GetLength(2)));

        var survivors = BuildSurvivorBalls(states, contactColumns, config.TotalRows);
        var (predicted, confidence) = grid.ScoreFlatTraining(survivors, allBalls);
        bool correct = predicted == targetTokenId;
        grid.ApplyFlatPostTrainingAdjustment(survivors, targetCentre, learningRate, correct);

        return new FlatPrmGpuTrainingSampleResult(predicted, correct, confidence, run);
    }

    private static FlatPrmGpuBallState[] PackBallStates(IReadOnlyList<Ball> balls)
    {
        var states = new FlatPrmGpuBallState[balls.Count];
        for (int i = 0; i < balls.Count; i++)
        {
            var ball = balls[i];
            states[i] = new FlatPrmGpuBallState(
                ball.Position,
                ball.Velocity,
                ball.Mass,
                ball.RelevanceWeight,
                ball.TokenId,
                ball.ContextPosition,
                active: ball.Active ? 1 : 0,
                stuck: ball.Stuck ? 1 : 0);
        }

        return states;
    }

    private static List<Ball> BuildSurvivorBalls(
        IReadOnlyList<FlatPrmGpuBallState> states,
        ReadOnlySpan<int> contactColumns,
        int totalRows)
    {
        var survivors = new List<Ball>();
        for (int i = 0; i < states.Count; i++)
        {
            var state = states[i];
            if (state.Active == 0)
                continue;

            var ball = new Ball(
                state.TokenId,
                state.Position,
                state.Mass,
                state.ContextPosition,
                state.RelevanceWeight)
            {
                Velocity = state.Velocity,
                Stuck = state.Stuck != 0
            };

            for (int row = 0; row < totalRows; row++)
            {
                int col = contactColumns[row * states.Count + i];
                if (col >= 0)
                {
                    int nailId = row * 10_000 + col;
                    if (ball.ContactNailIds.Count == 0 || ball.ContactNailIds[^1] != nailId)
                        ball.ContactNailIds.Add(nailId);
                }
            }

            survivors.Add(ball);
        }

        return survivors;
    }
}
