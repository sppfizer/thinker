namespace PRM.Core.Models;

/// <summary>
/// A token in the frequency-ranked vocabulary.
/// Ball mass is derived from normalised log-frequency.
/// </summary>
public class VocabToken
{
    public int    Id        { get; init; }
    public string Text      { get; init; } = string.Empty;
    public long   Frequency { get; init; }

    /// <summary>Normalised mass in (0, 1] — higher frequency = larger mass.</summary>
    public float  Mass      { get; init; }

    /// <summary>Width of output slot at grid bottom (proportional to log-frequency).</summary>
    public float  SlotWidth { get; init; }

    /// <summary>Left edge of this token's output slot.</summary>
    public float  SlotLeft  { get; set; }

    /// <summary>Right edge of this token's output slot.</summary>
    public float  SlotRight => SlotLeft + SlotWidth;
}
