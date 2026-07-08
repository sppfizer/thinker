namespace PRM.Core.Models;

/// <summary>
/// A single nail (peg) in the staggered routing grid.
///
/// Physical model:
///   - Each nail sits at a fixed grid slot (base position).
///   - The nail's actual centre can be offset within a unit circle: (OffsetX², OffsetY²) ≤ 1.
///     At maximum offset the nail just touches its nearest neighbour (spacing = 2).
///   - Radius ∈ (0, 1] controls the nail's size / influence area.
///     Large radius → hard to miss; small radius → precise but narrow.
///   - Resistance is the nail's inertia against training updates.
///     High resistance (→1) = stable / slow learner; low (→0) = flexible.
///
/// The per-token learned routing is stored externally in BallSimulator._tokenOffX/Y.
/// This struct stores only the shared physical properties (radius, resistance).
/// </summary>
public struct Nail
{
    /// <summary>Nail size / influence radius in (0, 1].  Shared across all token types.</summary>
    public float Radius     { get; set; }

    /// <summary>Resistance to training updates in (0, 1].  Acts like the old Diameter/bias.</summary>
    public float Resistance { get; set; }

    public Nail(float radius = 0.5f, float resistance = 0.5f)
    {
        Radius     = radius;
        Resistance = resistance;
    }
}

