namespace PRM.Core.Models;

/// <summary>
/// Configuration for a DiamondGrid role specialist.
/// The shape of the diamond encodes the cognitive role.
///
/// Nail grid geometry (staggered / hexagonal):
///   Even layers:  nails at x = 0, 2, 4, 6, …
///   Odd  layers:  nails at x = 1, 3, 5, 7, …  (offset by NailSpacing/2 = 1)
///   A nail at maximum unit-circle offset (+1,0) just touches its nearest neighbour
///   (spacing = 2, max offset radius = 1 → touching distance = 1 + 1 = 2). ✓
/// </summary>
public class DiamondConfig
{
    public string RoleName          { get; init; } = "Default";

    /// <summary>Entry width of the grid (grid units).</summary>
    public float  EntryWidth        { get; init; } = 20f;

    /// <summary>Maximum width at the midpoint.</summary>
    public float  MaxWidth          { get; init; } = 60f;

    /// <summary>Number of widening rows (divergent thinking phase).</summary>
    public int    WideningRows      { get; init; } = 10;

    /// <summary>Number of narrowing rows (convergent summarisation phase).</summary>
    public int    NarrowingRows     { get; init; } = 10;

    /// <summary>
    /// Nail spacing = distance between adjacent nail base positions.
    /// Default 2.0 so that the unit-circle offset (max = 1) just reaches the next nail.
    /// </summary>
    public float  NailSpacing       { get; init; } = 2.0f;

    /// <summary>Default nail radius for initialisation (0-1).</summary>
    public float  DefaultRadius     { get; init; } = 0.5f;

    /// <summary>
    /// Base horizontal deflection scale α.
    /// Per-row position change = token_offX * α / mass  (clamped to ±rowWidth/TotalRows*α).
    /// </summary>
    public float  DeflectionAlpha   { get; init; } = 0.8f;

    /// <summary>
    /// Exponent applied to inverse mass during deflection routing.
    /// 0 = flat, 0.5 = sqrt-IDF, 1 = inverse-mass.
    /// </summary>
    public float  DeflectionIdfPower { get; init; } = 0.0f;

    /// <summary>
    /// Vertical-offset contribution to horizontal velocity (new in 2D nail model).
    /// Per-row velocity change = token_offY * αY * radius / mass.
    /// Small (0.1–0.3) lets the nail give multi-row momentum nudges.
    /// </summary>
    public float  DeflectionAlphaY  { get; init; } = 0.15f;

    /// <summary>Gravitational constant G for ball-to-ball attraction.</summary>
    public float  GravityG          { get; init; } = 0.0f;

    /// <summary>Maximum distance for gravity and collision checks.</summary>
    public float  ProximityBand     { get; init; } = 4f;

    /// <summary>Collision radius — balls closer than this interact elastically.</summary>
    public float  CollisionRadius   { get; init; } = 0.0f;

    /// <summary>Time-step for velocity integration.</summary>
    public float  DeltaTime         { get; init; } = 0.1f;

    /// <summary>
    /// Input context window size (number of tokens per sample).
    /// Used to allocate position-aware routing tables (offsets are stored per position).
    /// </summary>
    public int    InputWindowSize   { get; init; } = 3;

    public int    TotalRows         => WideningRows + NarrowingRows;

    // ── Legacy alias so existing JSON keeps loading ────────────────────────────
    /// <summary>Legacy: maps to DefaultRadius.</summary>
    public float  DefaultDiameter   { get => DefaultRadius; init => DefaultRadius = value; }

    /// <summary>Pre-defined role configs (updated for 2D nail model).</summary>
    public static DiamondConfig Analyst         => new() { RoleName="Analyst",         WideningRows=10, NarrowingRows=10, DefaultRadius=0.5f };
    public static DiamondConfig Generator       => new() { RoleName="Generator",       WideningRows=16, NarrowingRows=6,  DefaultRadius=0.3f };
    public static DiamondConfig Synthesizer     => new() { RoleName="Synthesizer",     WideningRows=6,  NarrowingRows=16, DefaultRadius=0.5f };
    public static DiamondConfig Precisionist    => new() { RoleName="Precisionist",    WideningRows=4,  NarrowingRows=14, DefaultRadius=0.8f };
    public static DiamondConfig Narrator        => new() { RoleName="Narrator",        WideningRows=12, NarrowingRows=12, DefaultRadius=0.45f };
    public static DiamondConfig Conversationalist => new() { RoleName="Conversationalist", WideningRows=14, NarrowingRows=7, DefaultRadius=0.3f };
}
