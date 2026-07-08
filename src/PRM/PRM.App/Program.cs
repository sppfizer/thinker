using PRM.Core.Engine;
using PRM.Core.Models;
using PRM.Core.Modes;

Console.WriteLine("╔══════════════════════════════════════════╗");
Console.WriteLine("║  PRM — Physical Routing Model  v0.1      ║");
Console.WriteLine("╚══════════════════════════════════════════╝");
Console.WriteLine();

// ── 1. Build vocabulary from a tiny corpus ────────────────────────────────
var corpus = """
    the cat sat on the mat
    the dog ran to the park
    the cat ran on the mat
    the dog sat on the floor
    a cat sat near the mat
    the mat was on the floor
    a dog ran near the park
    """;

var builder = new VocabularyBuilder();
builder.Feed(VocabularyBuilder.Tokenise(corpus));
VocabToken[] vocab = builder.Build();

Console.WriteLine($"Vocabulary: {vocab.Length} tokens");
foreach (var t in vocab.Take(8))
    Console.WriteLine($"  [{t.Id,2}] '{t.Text,-15}' freq={t.Frequency,3}  mass={t.Mass:F3}  slotW={t.SlotWidth:F2}");
Console.WriteLine();

// ── 2. Create a specialist (Analyst role for this simple test) ────────────
var config = new DiamondConfig
{
    RoleName = "Analyst",
    WideningRows = 8,
    NarrowingRows = 8,
    MaxWidth = 50f,
    EntryWidth = 20f,
    DefaultDiameter = 0.75f
};
var grid   = new DiamondGrid(config, vocab);
var router = new SpecialistRouter(new[] { grid });

// ── 3. Build dataset: sliding window next-token prediction ────────────────
var tokenIds = VocabularyBuilder.Tokenise(corpus)
    .Select(t => vocab.FirstOrDefault(v => v.Text == t)?.Id ?? -1)
    .Where(id => id >= 0)
    .ToArray();

var dataset = new List<(int[] input, int target)>();
int window = 3;
for (int i = 0; i < tokenIds.Length - window; i++)
    dataset.Add((tokenIds[i..(i + window)], tokenIds[i + window]));

Console.WriteLine($"Dataset: {dataset.Count} samples  (window={window})");
Console.WriteLine();

// ── 4. Mode selection ─────────────────────────────────────────────────────
var mode = args.Length > 0 ? args[0].ToLower() : "train";
Console.WriteLine($"MODE: {mode.ToUpper()}");
Console.WriteLine(new string('─', 50));

switch (mode)
{
    // ── TRAINING ─────────────────────────────────────────────────────────
    case "train":
    {
        var trainer = new TrainingMode(router) { LearningRate = 0.02f, EpochCount = 5 };
        int epoch = 0;
        foreach (var metrics in trainer.Run(dataset))
        {
            Console.WriteLine($"Epoch {++epoch:D2}  {metrics}");
        }
        grid.SaveNails("prm_nails.bin");
        Console.WriteLine("\nNails saved → prm_nails.bin");
        break;
    }

    // ── TEST ──────────────────────────────────────────────────────────────
    case "test":
    {
        if (File.Exists("prm_nails.bin")) { grid.LoadNails("prm_nails.bin"); Console.WriteLine("Nails loaded."); }
        var tester  = new TestMode(router);
        var metrics = tester.Run(dataset, r =>
        {
            if (!r.Correct)
                Console.WriteLine($"  MISS  pred={vocab[r.Predicted].Text,-10} target={vocab[r.Target].Text,-10} conf={r.Confidence:F3}");
        });
        Console.WriteLine($"\nResult: {metrics}");
        break;
    }

    // ── TUNE ──────────────────────────────────────────────────────────────
    case "tune":
    {
        if (File.Exists("prm_nails.bin")) { grid.LoadNails("prm_nails.bin"); Console.WriteLine("Nails loaded."); }
        var tuner = new TuneMode(router) { LearningRate = 0.002f, EpochCount = 3 };
        int epoch = 0;
        foreach (var metrics in tuner.Run(dataset.Take(dataset.Count / 2)))
            Console.WriteLine($"Tune epoch {++epoch:D2}  {metrics}");
        grid.SaveNails("prm_nails_tuned.bin");
        Console.WriteLine("\nTuned nails saved → prm_nails_tuned.bin");
        break;
    }

    // ── VAL ───────────────────────────────────────────────────────────────
    case "val":
    {
        if (File.Exists("prm_nails.bin")) { grid.LoadNails("prm_nails.bin"); Console.WriteLine("Nails loaded."); }
        var val = new ValMode(router);
        var (metrics, mismatches) = val.Run(dataset);
        Console.WriteLine($"\nVal result: {metrics}");
        if (mismatches.Count > 0)
        {
            Console.WriteLine($"\nTop mismatches ({mismatches.Count}):");
            foreach (var (pred, tgt, role) in mismatches.Take(10))
                Console.WriteLine($"  pred={vocab[pred].Text,-10} target={vocab[tgt].Text,-10} role={role}");
        }
        break;
    }

    default:
        Console.WriteLine($"Unknown mode '{mode}'. Use: train | test | tune | val");
        break;
}

Console.WriteLine("\nDone.");
