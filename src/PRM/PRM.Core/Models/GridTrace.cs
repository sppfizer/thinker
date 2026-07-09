namespace PRM.Core.Models;

/// <summary>Single snapshot of one ball at a specific row.</summary>
public record BallFrame(int TokenId, float Position, float Velocity, float Mass, int ContextPosition);

/// <summary>
/// Full simulation trace: ball positions at every row, plus grid geometry.
/// Produced by BallSimulator.SimulateWithTrace — used by the PRM.Viz visualiser.
/// </summary>
public class GridTrace
{
    /// <summary>Number of grid rows.</summary>
    public int TotalRows { get; init; }

    /// <summary>Widening rows (thinking phase).</summary>
    public int WideningRows { get; init; }

    /// <summary>Maximum horizontal width (grid units).</summary>
    public float MaxWidth { get; init; }

    /// <summary>Left border of each row (grid units).</summary>
    public float[] GridLefts { get; init; } = [];

    /// <summary>Right border of each row (grid units).</summary>
    public float[] GridRights { get; init; } = [];

    /// <summary>Nail base X positions per row. [row][nailIndex]</summary>
    public float[][] NailBaseXs { get; init; } = [];

    /// <summary>Per-nail X-offset averaged across active balls. [row][nailIndex]</summary>
    public float[][] NailOffXs { get; init; } = [];

    /// <summary>Per-nail Y-offset averaged across active balls. [row][nailIndex]</summary>
    public float[][] NailOffYs { get; init; } = [];

    /// <summary>Per-nail physical radius (influence size). [row][nailIndex]</summary>
    public float[][] NailRadii { get; init; } = [];

    /// <summary>Per-nail resistance (higher = stiffer). [row][nailIndex]</summary>
    public float[][] NailResistances { get; init; } = [];

    /// <summary>Number of active nails per row.</summary>
    public int[] RowNailCounts { get; init; } = [];

    /// <summary>
    /// Ball frames per row, indexed [row][ballIndex].
    /// Length = TotalRows + 1  (row 0 = entry, row TotalRows = final positions after last row).
    /// </summary>
    public BallFrame[][] RowFrames { get; init; } = [];

    /// <summary>Convenience: final positions after all rows.</summary>
    public BallFrame[] FinalBalls => RowFrames[^1];
}
