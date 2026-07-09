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

    /// <summary>
    /// Blend between shared token routing and position-specific residual routing.
    /// 0 = old fully position-specific table, 1 = fully shared token table.
    /// </summary>
    public float  SharedOffsetBlend  { get; init; } = 0.0f;

    /// <summary>Gaussian distance sigma in output-slot units. 0 = hard slot bucket voting.</summary>
    public float  ScoreDistanceSigma { get; init; } = 0.0f;

    /// <summary>Weight of the neutral prediction probe ball in scoring. 0 = disabled.</summary>
    public float  ScoreProbeWeight   { get; init; } = 0.0f;

    /// <summary>
    /// Training strength for the neutral prediction probe (tokenId -1).
    /// 0 = keep the probe observational only; >0 pulls its routing state toward the target.
    /// </summary>
    public float  PredictionProbeTrainingWeight { get; init; } = 0.0f;

    /// <summary>
    /// Recency decay for context-token relevance. 0 = all context tokens equally weighted.
    /// Values near 0.1 make later tokens stronger while retaining older-token signal.
    /// </summary>
    public float  ContextRelevanceDecay { get; init; } = 0.0f;

    /// <summary>
    /// Restores decayed relevance for older tokens that reappear later in the same context.
    /// Only has an effect when ContextRelevanceDecay > 0.
    /// </summary>
    public float  ContextReinforcementStrength { get; init; } = 1.0f;

    /// <summary>
    /// Fraction of a hit nail's training update that is also applied to nearby downstream nails.
    /// 0 = disabled, preserving independent nail updates.
    /// </summary>
    public float  DownstreamNailInfluence { get; init; } = 0.0f;

    /// <summary>Number of lower rows that receive downstream nail influence. 0 = disabled.</summary>
    public int    DownstreamNailInfluenceRows { get; init; } = 0;

    /// <summary>Horizontal downstream influence radius in nail-spacing units.</summary>
    public float  DownstreamNailInfluenceRadius { get; init; } = 1.5f;

    /// <summary>Per-row downstream influence decay. 0 = no row decay, larger = faster decay.</summary>
    public float  DownstreamNailInfluenceDecay { get; init; } = 0.5f;

    /// <summary>
    /// Blend downstream influence from legacy nearby-copy (0) to target-directional pathing (1).
    /// No effect unless DownstreamNailInfluence and DownstreamNailInfluenceRows are enabled.
    /// </summary>
    public float  DownstreamNailTargetDirectionality { get; init; } = 1.0f;

    /// <summary>Number of retained context summary balls to synthesize after interaction. 0 = disabled.</summary>
    public int    ContextSummaryBallCount { get; init; } = 0;

    /// <summary>Row after which summary balls are created. -1 = widening/narrowing boundary.</summary>
    public int    ContextSummaryRow { get; init; } = -1;

    /// <summary>Mass multiplier for retained context summary balls.</summary>
    public float  ContextSummaryMassScale { get; init; } = 1.0f;

    /// <summary>Score weight for retained context summary balls. Only used when summaries are enabled.</summary>
    public float  ContextSummaryScoreWeight { get; init; } = 1.0f;

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
