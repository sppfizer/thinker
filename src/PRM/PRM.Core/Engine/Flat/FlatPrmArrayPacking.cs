using PRM.Core.Models;

namespace PRM.Core.Engine.Flat;

/// <summary>
/// Helpers for converting the existing object-model state into GPU-shaped arrays.
/// Packing order is row-major: ((row * maxColumns + col) * depth) + item.
/// </summary>
public static class FlatPrmArrayPacking
{
    public static float[] Flatten(float[,,] source)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);
        int depth = source.GetLength(2);
        var flat = new float[rows * columns * depth];

        for (int row = 0; row < rows; row++)
        for (int col = 0; col < columns; col++)
        for (int item = 0; item < depth; item++)
            flat[((row * columns + col) * depth) + item] = source[row, col, item];

        return flat;
    }

    public static float[,,] Unflatten(float[] source, int rows, int columns, int depth)
    {
        if (rows < 0) throw new ArgumentOutOfRangeException(nameof(rows));
        if (columns < 0) throw new ArgumentOutOfRangeException(nameof(columns));
        if (depth < 0) throw new ArgumentOutOfRangeException(nameof(depth));
        if (source.Length < rows * columns * depth)
            throw new ArgumentException("Source is shorter than the requested 3D shape.", nameof(source));

        var target = new float[rows, columns, depth];
        for (int row = 0; row < rows; row++)
        for (int col = 0; col < columns; col++)
        for (int item = 0; item < depth; item++)
            target[row, col, item] = source[((row * columns + col) * depth) + item];

        return target;
    }

    public static float[] FlattenNailRadii(Nail[,] nails, int totalRows, int maxColumns)
    {
        if (nails.GetLength(0) < totalRows)
            throw new ArgumentException("Nail row count is smaller than totalRows.", nameof(nails));
        if (nails.GetLength(1) < maxColumns)
            throw new ArgumentException("Nail column count is smaller than maxColumns.", nameof(nails));

        var flat = new float[totalRows * maxColumns];
        for (int row = 0; row < totalRows; row++)
        for (int col = 0; col < maxColumns; col++)
            flat[row * maxColumns + col] = nails[row, col].Radius;

        return flat;
    }

    public static float[] FlattenNailResistances(Nail[,] nails, int totalRows, int maxColumns)
    {
        ValidateNailShape(nails, totalRows, maxColumns);

        var flat = new float[totalRows * maxColumns];
        for (int row = 0; row < totalRows; row++)
        for (int col = 0; col < maxColumns; col++)
            flat[row * maxColumns + col] = nails[row, col].Resistance;

        return flat;
    }

    public static float[] FlattenNailDensities(Nail[,] nails, int totalRows, int maxColumns)
    {
        ValidateNailShape(nails, totalRows, maxColumns);

        var flat = new float[totalRows * maxColumns];
        for (int row = 0; row < totalRows; row++)
        for (int col = 0; col < maxColumns; col++)
            flat[row * maxColumns + col] = nails[row, col].Density;

        return flat;
    }

    public static (
        float[] Positions,
        float[] Velocities,
        float[] Masses,
        int[] ContextPositions,
        int[] TokenIds) CopyBalls(IReadOnlyList<Ball> balls)
    {
        var positions = new float[balls.Count];
        var velocities = new float[balls.Count];
        var masses = new float[balls.Count];
        var contextPositions = new int[balls.Count];
        var tokenIds = new int[balls.Count];

        for (int i = 0; i < balls.Count; i++)
        {
            positions[i] = balls[i].Position;
            velocities[i] = balls[i].Velocity;
            masses[i] = balls[i].Mass;
            contextPositions[i] = balls[i].ContextPosition;
            tokenIds[i] = balls[i].TokenId;
        }

        return (positions, velocities, masses, contextPositions, tokenIds);
    }

    private static void ValidateNailShape(Nail[,] nails, int totalRows, int maxColumns)
    {
        if (nails.GetLength(0) < totalRows)
            throw new ArgumentException("Nail row count is smaller than totalRows.", nameof(nails));
        if (nails.GetLength(1) < maxColumns)
            throw new ArgumentException("Nail column count is smaller than maxColumns.", nameof(nails));
    }
}
