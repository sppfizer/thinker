using PRM.Core.Models;

namespace PRM.Core.Engine.Flat;

/// <summary>
/// Scalar constants needed by the flat CPU PRM kernels.
/// Keep this shape GPU-friendly: no object references, only values that can be
/// copied next to flat buffers.
/// </summary>
public readonly record struct FlatPrmKernelConfig
{
    public int TotalRows { get; init; }
    public int WideningRows { get; init; }
    public int VocabSize { get; init; }
    public int WindowSize { get; init; }
    public int MaxColumns { get; init; }
    public float NailSpacing { get; init; }
    public float DeflectionAlpha { get; init; }
    public float DeflectionAlphaY { get; init; }
    public float DeflectionIdfPower { get; init; }
    public float SharedOffsetBlend { get; init; }
    public float PredictionProbeTrainingWeight { get; init; }
    public float GravityG { get; init; }
    public float ProximityBand { get; init; }
    public float CollisionRadius { get; init; }
    public float DeltaTime { get; init; }
    public float MaxVelocity { get; init; }

    public int TokenSlotCount => VocabSize + 1;
    public int TokenKeyCount => WindowSize * TokenSlotCount;

    public FlatPrmKernelConfig(
        int totalRows,
        int wideningRows,
        int vocabSize,
        int windowSize,
        int maxColumns,
        float nailSpacing,
        float deflectionAlpha,
        float deflectionAlphaY,
        float deflectionIdfPower,
        float sharedOffsetBlend,
        float predictionProbeTrainingWeight,
        float gravityG,
        float proximityBand,
        float collisionRadius,
        float deltaTime,
        float maxVelocity = 5f)
    {
        TotalRows = totalRows;
        WideningRows = wideningRows;
        VocabSize = Math.Max(vocabSize, 0);
        WindowSize = Math.Max(windowSize, 1);
        MaxColumns = Math.Max(maxColumns, 0);
        NailSpacing = nailSpacing;
        DeflectionAlpha = deflectionAlpha;
        DeflectionAlphaY = deflectionAlphaY;
        DeflectionIdfPower = deflectionIdfPower;
        SharedOffsetBlend = sharedOffsetBlend;
        PredictionProbeTrainingWeight = predictionProbeTrainingWeight;
        GravityG = gravityG;
        ProximityBand = proximityBand;
        CollisionRadius = collisionRadius;
        DeltaTime = deltaTime;
        MaxVelocity = maxVelocity;
    }

    public static FlatPrmKernelConfig FromConfig(DiamondConfig config, int vocabSize)
    {
        int maxColumns = (int)(config.MaxWidth / config.NailSpacing) + 2;
        return FromConfig(config, vocabSize, maxColumns);
    }

    public static FlatPrmKernelConfig FromConfig(DiamondConfig config, int vocabSize, int maxColumns) =>
        new(
            totalRows: config.TotalRows,
            wideningRows: config.WideningRows,
            vocabSize: vocabSize,
            windowSize: config.InputWindowSize,
            maxColumns: maxColumns,
            nailSpacing: config.NailSpacing,
            deflectionAlpha: config.DeflectionAlpha,
            deflectionAlphaY: config.DeflectionAlphaY,
            deflectionIdfPower: config.DeflectionIdfPower,
            sharedOffsetBlend: config.SharedOffsetBlend,
            predictionProbeTrainingWeight: config.PredictionProbeTrainingWeight,
            gravityG: config.GravityG,
            proximityBand: config.ProximityBand,
            collisionRadius: config.CollisionRadius,
            deltaTime: config.DeltaTime);
}
