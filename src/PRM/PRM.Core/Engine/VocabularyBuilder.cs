using PRM.Core.Models;

namespace PRM.Core.Engine;

/// <summary>
/// Builds and maintains the frequency-ranked vocabulary.
/// Token mass = normalised log-frequency.
/// Output slot widths are proportional to log-frequency.
/// </summary>
public class VocabularyBuilder
{
    private readonly Dictionary<string, long> _counts = new(StringComparer.Ordinal);

    public void Feed(IEnumerable<string> tokens)
    {
        foreach (var t in tokens)
            _counts[t] = _counts.GetValueOrDefault(t) + 1;
    }

    public VocabToken[] Build()
    {
        if (_counts.Count == 0) throw new InvalidOperationException("No tokens fed.");

        // Sort descending by frequency → highest freq = index 0
        var sorted = _counts.OrderByDescending(kv => kv.Value).ToArray();

        double maxLog = Math.Log(sorted[0].Value + 1);
        double totalLog = sorted.Sum(kv => Math.Log(kv.Value + 1));

        var tokens = new VocabToken[sorted.Length];
        float cursor = 0f;

        for (int i = 0; i < sorted.Length; i++)
        {
            double logFreq = Math.Log(sorted[i].Value + 1);
            float mass      = (float)(logFreq / maxLog);
            float slotWidth = (float)(logFreq / totalLog * sorted.Length); // relative width

            tokens[i] = new VocabToken
            {
                Id        = i,
                Text      = sorted[i].Key,
                Frequency = sorted[i].Value,
                Mass      = Math.Max(mass, 0.01f),
                SlotWidth = Math.Max(slotWidth, 0.1f),
                SlotLeft  = cursor,
            };
            cursor += tokens[i].SlotWidth;
        }

        return tokens;
    }

    /// <summary>Simple whitespace tokeniser for quick testing.</summary>
    public static IEnumerable<string> Tokenise(string text)
        => text.Split(' ', '\n', '\r', '\t')
               .Select(t => t.Trim().ToLowerInvariant())
               .Where(t => t.Length > 0);
}
