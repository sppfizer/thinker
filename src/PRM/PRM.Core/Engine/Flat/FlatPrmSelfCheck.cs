using PRM.Core.Models;

namespace PRM.Core.Engine.Flat;

public readonly record struct FlatPrmComparison(
    int ComparedCount,
    int CountDelta,
    float MaxPositionDelta,
    float MaxVelocityDelta)
{
    public bool IsWithin(float tolerance) =>
        CountDelta == 0 &&
        MaxPositionDelta <= tolerance &&
        MaxVelocityDelta <= tolerance;
}

/// <summary>
/// Small parity helper for comparing flat kernel output with BallSimulator output.
/// Use only for the first safe slice: no training, no gravity/collisions, and no
/// boundary drops/bounces/stuck removals if exact count parity is expected.
/// </summary>
public static class FlatPrmSelfCheck
{
    public static FlatPrmComparison CompareFinalState(
        ReadOnlySpan<float> flatPositions,
        ReadOnlySpan<float> flatVelocities,
        IReadOnlyList<Ball> objectBalls)
    {
        var expectedPositions = new float[objectBalls.Count];
        var expectedVelocities = new float[objectBalls.Count];
        for (int i = 0; i < objectBalls.Count; i++)
        {
            expectedPositions[i] = objectBalls[i].Position;
            expectedVelocities[i] = objectBalls[i].Velocity;
        }

        return CompareFinalState(flatPositions, flatVelocities, expectedPositions, expectedVelocities);
    }

    public static FlatPrmComparison CompareFinalState(
        ReadOnlySpan<float> flatPositions,
        ReadOnlySpan<float> flatVelocities,
        ReadOnlySpan<float> expectedPositions,
        ReadOnlySpan<float> expectedVelocities)
    {
        int compared = Math.Min(
            Math.Min(flatPositions.Length, flatVelocities.Length),
            Math.Min(expectedPositions.Length, expectedVelocities.Length));

        float maxPositionDelta = 0f;
        float maxVelocityDelta = 0f;
        for (int i = 0; i < compared; i++)
        {
            maxPositionDelta = Math.Max(maxPositionDelta, Math.Abs(flatPositions[i] - expectedPositions[i]));
            maxVelocityDelta = Math.Max(maxVelocityDelta, Math.Abs(flatVelocities[i] - expectedVelocities[i]));
        }

        int flatCount = Math.Min(flatPositions.Length, flatVelocities.Length);
        int expectedCount = Math.Min(expectedPositions.Length, expectedVelocities.Length);
        return new FlatPrmComparison(
            ComparedCount: compared,
            CountDelta: flatCount - expectedCount,
            MaxPositionDelta: maxPositionDelta,
            MaxVelocityDelta: maxVelocityDelta);
    }

    public static void ThrowIfOutsideTolerance(FlatPrmComparison comparison, float tolerance)
    {
        if (comparison.IsWithin(tolerance)) return;

        throw new InvalidOperationException(
            $"Flat PRM parity failed: compared={comparison.ComparedCount}, " +
            $"countDelta={comparison.CountDelta}, " +
            $"maxPositionDelta={comparison.MaxPositionDelta}, " +
            $"maxVelocityDelta={comparison.MaxVelocityDelta}, " +
            $"tolerance={tolerance}.");
    }
}
