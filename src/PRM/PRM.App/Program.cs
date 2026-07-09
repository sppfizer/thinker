using System.Diagnostics;
using System.Text;
using System.Text.Json;
using PRM.Core.Engine;
using PRM.Core.Models;
using PRM.Core.Modes;
using PRM.Core.Viz;

// Ensure UTF-8 output on all platforms (fixes mojibake in Windows PowerShell console)
Console.OutputEncoding = Encoding.UTF8;

Console.WriteLine("╔══════════════════════════════════════════╗");
Console.WriteLine("║  PRM — Physical Routing Model  v0.1      ║");
Console.WriteLine("╚══════════════════════════════════════════╝");
Console.WriteLine();

// ── 1. Build vocabulary from corpus ──────────────────────────────────────
// Optional: pass --corpus <filename> as first two args to pick a different file
string? corpusFile = null;
string[] modeArgs  = args;
if (args.Length >= 2 && args[0] == "--corpus")
{
    corpusFile = args[1];
    modeArgs   = args[2..];
}
var corpus = LoadCorpus(corpusFile);

var builder = new VocabularyBuilder();
builder.Feed(VocabularyBuilder.Tokenise(corpus));
VocabToken[] vocab = builder.Build();

Console.WriteLine($"Vocabulary: {vocab.Length} tokens");
foreach (var t in vocab.Take(8))
    Console.WriteLine($"  [{t.Id,2}] '{t.Text,-15}' freq={t.Frequency,3}  mass={t.Mass:F3}  slotW={t.SlotWidth:F2}");
Console.WriteLine();

// ── 2. Load the active configuration ───────────────────────────────────────
var config = LoadActiveConfig();
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

// Shuffle so all sentences mix across train/val/test — without this, rare words
// in the last sentences are never seen during training (sequential split problem).
var shuffleRng = new Random(42);
for (int i = dataset.Count - 1; i > 0; i--)
{
    int j = shuffleRng.Next(i + 1);
    (dataset[i], dataset[j]) = (dataset[j], dataset[i]);
}

int trainCount = Math.Max(1, (int)(dataset.Count * 0.60));
int testCount  = Math.Max(1, (int)(dataset.Count * 0.20));
int valCount   = Math.Max(1, dataset.Count - trainCount - testCount);

var trainSet = dataset.Take(trainCount).ToList();
var testSet  = dataset.Skip(trainCount).Take(testCount).ToList();
var valSet   = dataset.Skip(trainCount + testCount).Take(valCount).ToList();
var tuneSet  = trainSet.Take(Math.Max(1, trainSet.Count / 2)).ToList();

Console.WriteLine($"Dataset: {dataset.Count} samples  (window={window})");
Console.WriteLine($"Split: train={trainSet.Count}, test={testSet.Count}, tune={tuneSet.Count}, val={valSet.Count}");
Console.WriteLine();

// ── 4. Mode selection ─────────────────────────────────────────────────────
var mode = modeArgs.Length > 0 ? modeArgs[0].ToLower() : "train";
Console.WriteLine($"MODE: {mode.ToUpper()}");
Console.WriteLine(new string('─', 50));

switch (mode)
{
    // ── TRAINING ─────────────────────────────────────────────────────────
    case "train":
    {
        // Allow: dotnet run -- train [epochs] [lr] [decay]
        int   epochs = modeArgs.Length > 1 ? int.Parse(modeArgs[1])    : 50;
        float lr     = modeArgs.Length > 2 ? float.Parse(modeArgs[2], System.Globalization.CultureInfo.InvariantCulture) : 0.08f;
        float decay  = modeArgs.Length > 3 ? float.Parse(modeArgs[3], System.Globalization.CultureInfo.InvariantCulture) : 0.97f;
        if (File.Exists("prm_nails.bin")) { grid.LoadNails("prm_nails.bin"); Console.WriteLine("Nails loaded — resuming."); }
        var trainer = new TrainingMode(router) { LearningRate = lr, EpochCount = epochs, LrDecayPerEpoch = decay };
        int   epoch    = 0;
        float bestAcc  = -1f;
        float curLR    = lr;
        foreach (var metrics in trainer.Run(trainSet))
        {
            Console.WriteLine($"Epoch {++epoch:D3}  {metrics}  LR={curLR:F5}");
            curLR *= decay;
            if (metrics.Accuracy > bestAcc)
            {
                bestAcc = metrics.Accuracy;
                grid.SaveNails("prm_nails_best.bin");
            }
        }
        // Keep best epoch as active
        if (File.Exists("prm_nails_best.bin"))
            File.Copy("prm_nails_best.bin", "prm_nails.bin", overwrite: true);
        SaveActiveConfig(config);
        Console.WriteLine($"\nBest train acc={bestAcc:P1} → prm_nails.bin");
        break;
    }

    // ── TEST ──────────────────────────────────────────────────────────────
    case "test":
    {
        if (File.Exists("prm_nails.bin")) { grid.LoadNails("prm_nails.bin"); Console.WriteLine("Nails loaded."); }
        var tester  = new TestMode(router);
        var metrics = tester.Run(testSet, r =>
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
        foreach (var metrics in tuner.Run(tuneSet))
            Console.WriteLine($"Tune epoch {++epoch:D2}  {metrics}");
        grid.SaveNails("prm_nails_tuned.bin");
        SaveActiveConfig(config);
        Console.WriteLine("\nTuned nails saved → prm_nails_tuned.bin");
        break;
    }

    // ── VAL ───────────────────────────────────────────────────────────────
    case "val":
    {
        if (File.Exists("prm_nails.bin")) { grid.LoadNails("prm_nails.bin"); Console.WriteLine("Nails loaded."); }
        var val = new ValMode(router);
        var (metrics, mismatches) = val.Run(valSet);
        Console.WriteLine($"\nVal result: {metrics}");
        if (mismatches.Count > 0)
        {
            Console.WriteLine($"\nTop mismatches ({mismatches.Count}):");
            foreach (var (pred, tgt, role) in mismatches.Take(10))
                Console.WriteLine($"  pred={vocab[pred].Text,-10} target={vocab[tgt].Text,-10} role={role}");
        }
        break;
    }

    // ── BENCHMARK ─────────────────────────────────────────────────────────
    case "benchmark":
    {
        if (File.Exists("prm_nails.bin")) { grid.LoadNails("prm_nails.bin"); Console.WriteLine("Nails loaded."); }

        var sw = Stopwatch.StartNew();
        var trainer = new TrainingMode(router) { LearningRate = 0.01f, EpochCount = 1 };
        var trainMetrics = trainer.Run(trainSet).Last();
        sw.Stop();
        Console.WriteLine($"Train benchmark: {trainMetrics}  time={sw.ElapsedMilliseconds}ms");

        sw.Restart();
        var testMetrics = new TestMode(router).Run(testSet);
        sw.Stop();
        Console.WriteLine($"Test  benchmark: {testMetrics}  time={sw.ElapsedMilliseconds}ms");

        sw.Restart();
        var (valMetrics, _) = new ValMode(router).Run(valSet);
        sw.Stop();
        Console.WriteLine($"Val   benchmark: {valMetrics}  time={sw.ElapsedMilliseconds}ms");
        break;
    }

    // ── OPTIMIZE (quick 5-candidate sweep, legacy) ────────────────────────────
    case "optimize":
    {
        RunOptimization(vocab, dataset, trainSet, testSet, tuneSet, valSet);
        break;
    }

    // ── AUTOOPTIMIZE (real hill-climbing convergence loop) ────────────────
    case "autooptimize":
    {
        int   maxIter = 500;
        float target  = 0.60f;
        AutoOptimizer.Run(vocab, trainSet, tuneSet, valSet, testSet, maxIter, target);
        break;
    }

    default:
        Console.WriteLine($"Unknown mode '{mode}'. Use: train | test | tune | val | benchmark | optimize | autooptimize | viz");
        break;

    // ── VIZ ──────────────────────────────────────────────────────────────
    case "viz":
    {
        if (File.Exists("prm_nails.bin")) { grid.LoadNails("prm_nails.bin"); Console.WriteLine("Nails loaded."); }
        int port = 5050;
        { var pi = Array.IndexOf(modeArgs, "--port"); if (pi >= 0 && pi + 1 < modeArgs.Length) int.TryParse(modeArgs[pi + 1], out port); }
        bool noBrowser = modeArgs.Contains("--no-browser");

        // Words to visualise: everything after "viz" on the command line
        int[] vizIds;
        if (modeArgs.Length > 1)
        {
            vizIds = modeArgs[1..]
                .Select(t => vocab.FirstOrDefault(v => v.Text.Trim().Equals(t.Trim(), StringComparison.OrdinalIgnoreCase))?.Id ?? -1)
                .Where(id => id >= 0)
                .Take(4)
                .ToArray();
            if (vizIds.Length < 2)
            {
                Console.WriteLine("Need ≥ 2 known words. Using random sample from corpus.");
                vizIds = GetRandomVizSample(VocabularyBuilder.Tokenise(corpus).Select(t => vocab.FirstOrDefault(v => v.Text == t)?.Id ?? -1).Where(id => id >= 0).ToArray());
            }
        }
        else
        {
            vizIds = GetRandomVizSample(VocabularyBuilder.Tokenise(corpus).Select(t => vocab.FirstOrDefault(v => v.Text == t)?.Id ?? -1).Where(id => id >= 0).ToArray());
        }

        await using var server = new VizServer(vocab, port);
        Console.WriteLine($"Visualizer at http://localhost:{server.Port}/");
        if (!noBrowser)
        {
            Console.WriteLine("Opening browser…");
            try { Process.Start(new ProcessStartInfo($"http://localhost:{server.Port}/") { UseShellExecute = true }); }
            catch { Console.WriteLine("(Could not open browser — open the URL manually)"); }
        }

        using var exitCts = new CancellationTokenSource();
        Console.CancelKeyPress += (_, e) => { e.Cancel = true; exitCts.Cancel(); };
        Console.WriteLine("Press Ctrl+C to quit, or Enter to replay / type new words.\n");

        while (!exitCts.IsCancellationRequested)
        {
            using var connCts = CancellationTokenSource.CreateLinkedTokenSource(exitCts.Token);
            connCts.CancelAfter(TimeSpan.FromMinutes(10));
            try { await server.WaitForClientAsync(connCts.Token); }
            catch (OperationCanceledException) { break; }

            var gridInfo = new DiamondGridInfo(grid.Config.TotalRows, grid.Config.WideningRows, grid.Config.EntryWidth, grid.Config.MaxWidth, grid.Config.NailSpacing);
            await server.SendConfigAsync(gridInfo);
            await Task.Delay(200);
            await VizStreamAsync(server, grid, vocab, vizIds);

            // REPL
            while (!exitCts.IsCancellationRequested)
            {
                Console.Write("tokens (or Enter to replay)> ");
                string? line;
                try { line = await Task.Run(() => Console.ReadLine(), exitCts.Token); }
                catch (OperationCanceledException) { goto nextClient; }
                if (line == null) break;
                line = line.Trim();
                if (line != "")
                {
                    var newIds = VocabularyBuilder.Tokenise(line)
                        .Select(t => vocab.FirstOrDefault(v => v.Text.Trim().Equals(t.Trim(), StringComparison.OrdinalIgnoreCase))?.Id ?? -1)
                        .Where(id => id >= 0).Take(4).ToArray();
                    if (newIds.Length < 2) { Console.WriteLine("  Need ≥ 2 known words."); continue; }
                    vizIds = newIds;
                }
                await VizStreamAsync(server, grid, vocab, vizIds);
            }
            nextClient:;
        }
        break;
    }
}

Console.WriteLine("\nDone.");

// ── Viz helpers ───────────────────────────────────────────────────────────────

static async Task VizStreamAsync(VizServer server, DiamondGrid grid, VocabToken[] vocab, int[] tokenIds)
{
    string[] labels = tokenIds.Select(id => vocab[id].Text.Trim()).ToArray();
    await server.SendClearAsync(labels);
    await Task.Delay(80);

    var trace = grid.SimulateWithTrace(tokenIds);

    // Stream all frames instantly — the browser clock controls playback speed
    for (int r = 0; r < trace.RowFrames.Length - 1; r++)   // skip last (post-grid) frame
    {
        var balls     = trace.RowFrames[r];
        var nailXs    = r < trace.NailBaseXs.Length      ? trace.NailBaseXs[r]      : [];
        var offXs     = r < trace.NailOffXs.Length       ? trace.NailOffXs[r]       : [];
        var nailRadii = r < trace.NailRadii.Length       ? trace.NailRadii[r]       : [];
        var nailRes   = r < trace.NailResistances.Length ? trace.NailResistances[r] : [];
        int rowNum    = r < trace.TotalRows              ? r                        : trace.TotalRows;
        await server.SendFrameAsync(balls, nailXs, offXs, nailRadii, nailRes, rowNum);
    }

    var (predicted, _) = grid.Predict(tokenIds);
    string predLabel   = predicted >= 0 && predicted < vocab.Length ? vocab[predicted].Text.Trim() : "?";
    await server.SendResultAsync(predLabel, target: null, correct: false);
    Console.WriteLine($"[viz] [{string.Join(", ", labels)}] → \"{predLabel}\"");
}

static int[] GetRandomVizSample(int[] tokenIds)
{
    if (tokenIds.Length < 3) return tokenIds;
    var rng = new Random();
    int start = rng.Next(0, tokenIds.Length - 3);
    return tokenIds[start..(start + 3)];
}

static string LoadCorpus(string? filename = null)
{
    filename ??= "tiny_corpus.txt";
    var candidates = new[]
    {
        Path.Combine(Directory.GetCurrentDirectory(), "data", filename),
        Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "data", filename),
        // Fallback: full corpus
        Path.Combine(Directory.GetCurrentDirectory(), "data", "simple_corpus.txt"),
        Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "data", "simple_corpus.txt"),
    };

    foreach (var path in candidates)
    {
        var full = Path.GetFullPath(path);
        if (File.Exists(full))
            return File.ReadAllText(full);
    }

    return """
        the cat sat on the mat
        the dog ran to the park
        the cat ran on the mat
        the dog sat on the floor
        a cat sat near the mat
        the mat was on the floor
        a dog ran near the park
        """;
}

static void RunOptimization(
    VocabToken[] vocab,
    List<(int[] input, int target)> dataset,
    List<(int[] input, int target)> trainSet,
    List<(int[] input, int target)> testSet,
    List<(int[] input, int target)> tuneSet,
    List<(int[] input, int target)> valSet)
{
    var candidates = new[]
    {
        new Candidate("baseline", 0.020f, 0.002f, 8, 8, 0.80f, 0.010f, 10f, 0.50f),
        new Candidate("more-magnet", 0.035f, 0.004f, 10, 10, 0.90f, 0.012f, 12f, 0.45f),
        new Candidate("tight-routes", 0.015f, 0.0015f, 12, 6, 0.75f, 0.008f, 8f, 0.70f),
        new Candidate("broad-think", 0.012f, 0.0012f, 14, 8, 0.70f, 0.006f, 7f, 0.85f),
        new Candidate("sharp-narrow", 0.025f, 0.003f, 6, 14, 0.95f, 0.014f, 14f, 0.35f),
    };

    Candidate? bestCandidate = null;
    EpochMetrics? bestVal = null;

    Console.WriteLine($"Optimization sweep: {candidates.Length} candidates");
    Console.WriteLine("Rule: keep best validation accuracy; rollback worse candidates automatically.");
    Console.WriteLine();

    foreach (var cand in candidates)
    {
        var config = new DiamondConfig
        {
            RoleName = "Analyst",
            WideningRows = cand.WideningRows,
            NarrowingRows = cand.NarrowingRows,
            MaxWidth = 50f,
            EntryWidth = 20f,
            DefaultDiameter = cand.DefaultDiameter,
            DeflectionAlpha = cand.DeflectionAlpha,
            GravityG = cand.GravityG,
            ProximityBand = cand.ProximityBand,
            CollisionRadius = cand.CollisionRadius,
        };

        var grid = new DiamondGrid(config, vocab, new Random(42));
        var router = new SpecialistRouter(new[] { grid });

        var trainer = new TrainingMode(router) { LearningRate = cand.LearningRate, EpochCount = 4 };
        foreach (var _ in trainer.Run(trainSet)) { }

        var tuner = new TuneMode(router) { LearningRate = cand.TuneLearningRate, EpochCount = 2 };
        foreach (var _ in tuner.Run(tuneSet)) { }

        var valResult = new ValMode(router).Run(valSet);
        var testResult = new TestMode(router).Run(testSet);

        var score = valResult.metrics.Accuracy;
        var prevBest = bestVal?.Accuracy ?? -1f;

        Console.WriteLine(
            $"{cand.Name,-12} | trainLR={cand.LearningRate,6:F3} tuneLR={cand.TuneLearningRate,6:F4} " +
            $"rows={cand.WideningRows,2}/{cand.NarrowingRows,2} alpha={cand.DeflectionAlpha,4:F2} " +
            $"grav={cand.GravityG,5:F3} diam={cand.DefaultDiameter,4:F2} -> " +
            $"val={valResult.metrics.Accuracy:P1} test={testResult.Accuracy:P1} conf={valResult.metrics.AvgConfidence:F3}");

        if (score > prevBest)
        {
            bestCandidate = cand;
            bestVal = valResult.metrics;
            grid.SaveNails("prm_nails_best.bin");
            Console.WriteLine($"  ↳ new best, rolled forward");
        }
        else
        {
            Console.WriteLine($"  ↳ worse than best ({prevBest:P1}), rolled back");
        }
    }

    Console.WriteLine();
    if (bestCandidate is null || bestVal is null)
    {
        Console.WriteLine("No candidate improved the score.");
        return;
    }

    Console.WriteLine($"BEST: {bestCandidate.Name}  val={bestVal.Accuracy:P1}  conf={bestVal.AvgConfidence:F3}");
    if (File.Exists("prm_nails_best.bin"))
    {
        File.Copy("prm_nails_best.bin", "prm_nails.bin", overwrite: true);
        SaveBestConfig(bestCandidate);
        Console.WriteLine("Best nails copied → prm_nails.bin");
    }
}

static DiamondConfig LoadActiveConfig()
{
    const string path = "prm_config.json";
    if (File.Exists(path))
    {
        var json = File.ReadAllText(path);
        var cfg = JsonSerializer.Deserialize<DiamondConfig>(json);
        if (cfg is not null)
            return cfg;
    }

    return new DiamondConfig
    {
        RoleName = "Analyst",
        WideningRows = 8,
        NarrowingRows = 8,
        MaxWidth = 50f,
        EntryWidth = 20f,
        DefaultDiameter = 0.75f
    };
}

static void SaveActiveConfig(DiamondConfig config)
{
    const string path = "prm_config.json";
    var json = JsonSerializer.Serialize(config, new JsonSerializerOptions { WriteIndented = true });
    File.WriteAllText(path, json);
}

static void SaveBestConfig(Candidate cand)
{
    var cfg = new DiamondConfig
    {
        RoleName = "Analyst",
        WideningRows = cand.WideningRows,
        NarrowingRows = cand.NarrowingRows,
        MaxWidth = 50f,
        EntryWidth = 20f,
        DefaultDiameter = cand.DefaultDiameter,
        DeflectionAlpha = cand.DeflectionAlpha,
        GravityG = cand.GravityG,
        ProximityBand = cand.ProximityBand,
        CollisionRadius = cand.CollisionRadius,
    };

    SaveActiveConfig(cfg);
}
