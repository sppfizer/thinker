using PRM.Core.Models;

namespace PRM.Core.Engine.Flat;

/// <summary>
/// Flat per-row geometry matching BallSimulator.GridWidth/LeftBorder/NailBaseX.
/// </summary>
public sealed class FlatPrmRowGeometry
{
    public int MaxColumns { get; }
    public float NailSpacing { get; }
    public float MaxWidth { get; }
    public float[] RowWidths { get; }
    public float[] LeftBorders { get; }
    public float[] RightBorders { get; }
    public int[] RowNailCounts { get; }
    public int TotalRows => RowWidths.Length;

    private FlatPrmRowGeometry(
        int maxColumns,
        float nailSpacing,
        float maxWidth,
        float[] rowWidths,
        float[] leftBorders,
        float[] rightBorders,
        int[] rowNailCounts)
    {
        MaxColumns = maxColumns;
        NailSpacing = nailSpacing;
        MaxWidth = maxWidth;
        RowWidths = rowWidths;
        LeftBorders = leftBorders;
        RightBorders = rightBorders;
        RowNailCounts = rowNailCounts;
    }

    public static FlatPrmRowGeometry FromConfig(DiamondConfig config)
    {
        int maxColumns = (int)(config.MaxWidth / config.NailSpacing) + 2;
        return FromConfig(config, maxColumns);
    }

    public static FlatPrmRowGeometry FromConfig(DiamondConfig config, int maxColumns)
    {
        int totalRows = Math.Max(config.TotalRows, 0);
        var rowWidths = new float[totalRows];
        var leftBorders = new float[totalRows];
        var rightBorders = new float[totalRows];
        var rowNailCounts = new int[totalRows];

        for (int row = 0; row < totalRows; row++)
        {
            float width = GridWidth(config, row);
            float left = (config.MaxWidth - width) / 2f;
            rowWidths[row] = width;
            leftBorders[row] = left;
            rightBorders[row] = left + width;
            rowNailCounts[row] = Math.Min((int)(width / config.NailSpacing) + 2, maxColumns);
        }

        return new FlatPrmRowGeometry(
            maxColumns,
            config.NailSpacing,
            config.MaxWidth,
            rowWidths,
            leftBorders,
            rightBorders,
            rowNailCounts);
    }

    public float NailBaseX(int row, int col)
    {
        float stagger = (row % 2 == 1) ? NailSpacing / 2f : 0f;
        return LeftBorders[row] + stagger + col * NailSpacing;
    }

    public int NailColumn(float x, int row)
    {
        if (float.IsNaN(x) || float.IsInfinity(x)) return -1;
        float stagger = (row % 2 == 1) ? NailSpacing / 2f : 0f;
        return (int)((x - LeftBorders[row] - stagger) / NailSpacing);
    }

    private static float GridWidth(DiamondConfig config, int row)
    {
        if (row <= config.WideningRows)
        {
            if (config.WideningRows == 0) return config.MaxWidth;
            return config.EntryWidth + row * (config.MaxWidth - config.EntryWidth) / config.WideningRows;
        }

        if (config.NarrowingRows == 0) return config.MaxWidth;
        float contractionRate = (config.MaxWidth - config.EntryWidth) / config.NarrowingRows;
        return config.MaxWidth - (row - config.WideningRows) * contractionRate;
    }
}
