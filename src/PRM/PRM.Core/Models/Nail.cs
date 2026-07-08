namespace PRM.Core.Models;

/// <summary>
/// A single nail in the routing grid.
/// Tilt determines deflection direction; diameter determines resistance to training updates.
/// </summary>
public class Nail
{
    /// <summary>Routing direction in [-1, 1].  -1 = hard left, +1 = hard right.</summary>
    public float Tilt     { get; set; }

    /// <summary>Resistance to update in (0, 1].  1 = immovable, 0.01 = very flexible.</summary>
    public float Diameter { get; set; }

    public Nail(float tilt = 0f, float diameter = 0.5f)
    {
        Tilt     = tilt;
        Diameter = diameter;
    }
}
