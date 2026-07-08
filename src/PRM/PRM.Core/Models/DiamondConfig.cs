namespace PRM.Core.Models;

/// <summary>
/// Configuration for a DiamondGrid role specialist.
/// The shape of the diamond encodes the cognitive role.
/// </summary>
public class DiamondConfig
{
    public string RoleName          { get; init; } = "Default";

    /// <summary>Entry width of the grid (vocabulary units).</summary>
    public float  EntryWidth        { get; init; } = 100f;

    /// <summary>Maximum width at the midpoint.</summary>
    public float  MaxWidth          { get; init; } = 300f;

    /// <summary>Number of widening rows (divergent thinking phase).</summary>
    public int    WideningRows      { get; init; } = 50;

    /// <summary>Number of narrowing rows (convergent summarisation phase).</summary>
    public int    NarrowingRows     { get; init; } = 50;

    /// <summary>Nail spacing — determines column count per row.</summary>
    public float  NailSpacing       { get; init; } = 1.0f;

    /// <summary>Default nail diameter for initialisation.</summary>
    public float  DefaultDiameter   { get; init; } = 0.5f;

    /// <summary>Base deflection constant α.</summary>
    public float  DeflectionAlpha   { get; init; } = 0.8f;

    /// <summary>Gravitational constant G for ball-to-ball attraction.</summary>
    public float  GravityG          { get; init; } = 0.01f;

    /// <summary>Maximum distance for gravity and collision checks.</summary>
    public float  ProximityBand     { get; init; } = 10f;

    /// <summary>Collision radius — balls closer than this interact elastically.</summary>
    public float  CollisionRadius   { get; init; } = 0.5f;

    /// <summary>Time-step for velocity integration.</summary>
    public float  DeltaTime         { get; init; } = 0.1f;

    public int    TotalRows         => WideningRows + NarrowingRows;

    /// <summary>Pre-defined role configs.</summary>
    public static DiamondConfig Analyst         => new() { RoleName="Analyst",         WideningRows=50,  NarrowingRows=50,  DefaultDiameter=0.75f };
    public static DiamondConfig Generator       => new() { RoleName="Generator",       WideningRows=90,  NarrowingRows=30,  DefaultDiameter=0.25f };
    public static DiamondConfig Synthesizer     => new() { RoleName="Synthesizer",     WideningRows=30,  NarrowingRows=90,  DefaultDiameter=0.50f };
    public static DiamondConfig Precisionist    => new() { RoleName="Precisionist",    WideningRows=20,  NarrowingRows=80,  DefaultDiameter=0.90f };
    public static DiamondConfig Narrator        => new() { RoleName="Narrator",        WideningRows=60,  NarrowingRows=60,  DefaultDiameter=0.45f };
    public static DiamondConfig Conversationalist => new() { RoleName="Conversationalist", WideningRows=70, NarrowingRows=35, DefaultDiameter=0.30f };
}
