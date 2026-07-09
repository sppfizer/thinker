using System.Diagnostics;
using System.Text;
using System.Text.Json;
using PRM.Core.Engine;
using PRM.Core.Engine.Flat;
using PRM.Core.Models;
using PRM.Core.Modes;
using PRM.Core.Viz;

// Ensure UTF-8 output on all platforms (fixes mojibake in Windows PowerShell console)
Console.OutputEncoding = Encoding.UTF8;

Console.WriteLine("╔══════════════════════════════════════════╗");
Console.WriteLine("║  PRM — Physical Routing Model  v0.1      ║");
Console.WriteLine("╚══════════════════════════════════════════╝");
Console.WriteLine();

if (args.Length > 0 && (args[0].Equals("gpu", StringComparison.OrdinalIgnoreCase) ||
                        args[0].Equals("--gpu-check", StringComparison.OrdinalIgnoreCase)))
{
    RunGpuDiagnostics();
    return;
}

const int MinComparableContextWindow = 16;

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

// ── 2. Mode/options and corpus token ids ───────────────────────────────────
var mode = modeArgs.Length > 0 ? modeArgs[0].ToLowerInvariant() : "train";
if (mode == "train" && modeArgs.Any(a => a.Equals("--gpu", StringComparison.OrdinalIgnoreCase)))
    mode = "gputrain";
var trainOptions = mode is "train" or "gputrain" ? ParseTrainOptions(modeArgs) : TrainOptions.Default;

var tokenIds = VocabularyBuilder.Tokenise(corpus)
    .Select(t => vocab.FirstOrDefault(v => v.Text == t)?.Id ?? -1)
    .Where(id => id >= 0)
    .ToArray();

AutoOptimizer.HyperParams? optimizerParams = null;
if (!trainOptions.NoOptimizer && AutoOptimizer.TryLoadBest(vocab, out var loadedBest))
{
    optimizerParams = loadedBest;
    Console.WriteLine("Optimizer best params loaded.");
}
else if (mode == "train" && !trainOptions.NoOptimizer && trainOptions.WarmupIterations > 0)
{
    Console.WriteLine($"No optimizer best params found; running bounded warmup ({trainOptions.WarmupIterations} iteration(s)).");
    AutoOptimizer.RunWarmup(vocab, tokenIds, trainOptions.WarmupIterations);
    if (AutoOptimizer.TryLoadBest(vocab, out loadedBest))
    {
        optimizerParams = loadedBest;
        Console.WriteLine("Optimizer warmup params loaded.");
    }
}

// ── 3. Load the active configuration ───────────────────────────────────────
var config = optimizerParams?.Config ?? LoadActiveConfig();
config = EnsureComparableContext(config, tokenIds.Length);
if (mode == "gputrain" && modeArgs.Any(a => a.Equals("smoke", StringComparison.OrdinalIgnoreCase)))
{
    config = PrepareGpuSmokeConfig(config);
    Console.WriteLine("GPU smoke uses the supported flat subset (gravity/collisions/summary/downstream disabled for this smoke run only).");
}

// ── 4. Build dataset: causal next-token prediction ─────────────────────────
var splits = BuildDataset(tokenIds, config.InputWindowSize);
int window = splits.Window;
var dataset = splits.Dataset;
var trainSet = splits.TrainSet;
var tuneSet  = splits.TuneSet;
var valSet   = splits.ValSet;
var testSet  = splits.TestSet;

var grid   = new DiamondGrid(config, vocab);
var router = new SpecialistRouter(new[] { grid });

Console.WriteLine($"Dataset: {dataset.Count} causal samples  (maxContext={window})");
Console.WriteLine($"Split: train={trainSet.Count}, tune={tuneSet.Count}, val={valSet.Count}  (70/10/20)");
Console.WriteLine();

// ── 5. Mode selection ─────────────────────────────────────────────────────
Console.WriteLine($"MODE: {mode.ToUpper()}");
Console.WriteLine(new string('─', 50));

switch (mode)
{
    // ── TRAINING ─────────────────────────────────────────────────────────
    case "train":
    {
        // Allow: dotnet run -- train [epochs] [lr] [decay] [--optimizer-warmup [N]] [--no-optimizer]
        int   epochs = trainOptions.Epochs ?? optimizerParams?.TrainEpochs ?? 50;
        float lr     = trainOptions.LearningRate ?? optimizerParams?.LR ?? 0.08f;
        float decay  = trainOptions.Decay ?? 0.97f;
        bool usingOptimizerSchedule = optimizerParams is not null && !trainOptions.HasExplicitSchedule;
        int   trainPasses = usingOptimizerSchedule ? Math.Max(1, optimizerParams!.TrainPasses) : 1;
        int   tuneEpochs  = usingOptimizerSchedule ? Math.Max(0, optimizerParams!.TuneEpochs) : 0;
        float tuneLR      = optimizerParams?.TuneLR ?? Math.Max(0.001f, lr / 8f);

        if (optimizerParams is not null)
            Console.WriteLine($"Using optimizer config: window={config.InputWindowSize}, LR={lr:F4}, epochs={epochs}×{trainPasses}+{tuneEpochs}");

        if (File.Exists("prm_nails.bin")) { grid.LoadNails("prm_nails.bin"); Console.WriteLine("Nails loaded — resuming."); }
        int   epoch    = 0;
        float bestAcc  = -1f;
        for (int pass = 0; pass < trainPasses; pass++)
        {
            float passLR = lr * MathF.Pow(0.85f, pass);
            float curLR  = passLR;
            var trainer = new TrainingMode(router) { LearningRate = passLR, EpochCount = epochs, LrDecayPerEpoch = decay };
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
        }

        // Keep best epoch as active
        if (File.Exists("prm_nails_best.bin"))
        {
            File.Copy("prm_nails_best.bin", "prm_nails.bin", overwrite: true);
            grid.LoadNails("prm_nails.bin");
        }

        if (tuneEpochs > 0)
        {
            var tuner = new TuneMode(router) { LearningRate = tuneLR, EpochCount = tuneEpochs };
            int tuneEpoch = 0;
            foreach (var metrics in tuner.Run(tuneSet))
                Console.WriteLine($"Tune epoch {++tuneEpoch:D2}  {metrics}");
            grid.SaveNails("prm_nails.bin");
        }

        SaveActiveConfig(config);
        Console.WriteLine($"\nBest train acc={bestAcc:P1} → prm_nails.bin");
        break;
    }

    // ── GPU TRAINING (supported flat subset, CPU fallback when unavailable) ──
    case "gputrain":
    {
        bool smoke = modeArgs.Any(a => a.Equals("smoke", StringComparison.OrdinalIgnoreCase));
        int epochs = smoke ? 1 : trainOptions.Epochs ?? optimizerParams?.TrainEpochs ?? 1;
        float lr = trainOptions.LearningRate ?? optimizerParams?.LR ?? 0.08f;
        float decay = trainOptions.Decay ?? 0.97f;
        bool usingOptimizerSchedule = !smoke && optimizerParams is not null && !trainOptions.HasExplicitSchedule;
        int trainPasses = usingOptimizerSchedule ? Math.Max(1, optimizerParams!.TrainPasses) : 1;
        int tuneEpochs = usingOptimizerSchedule ? Math.Max(0, optimizerParams!.TuneEpochs) : 0;
        float tuneLR = optimizerParams?.TuneLR ?? Math.Max(0.001f, lr / 8f);
        var gpuTrainSet = smoke ? trainSet.Take(Math.Min(8, trainSet.Count)).ToList() : trainSet;

        if (optimizerParams is not null && !smoke)
            Console.WriteLine($"Using optimizer config: window={config.InputWindowSize}, LR={lr:F4}, epochs={epochs}×{trainPasses}+{tuneEpochs}");

        if (File.Exists("prm_nails.bin")) { grid.LoadNails("prm_nails.bin"); Console.WriteLine("Nails loaded — resuming."); }

        int epoch = 0;
        for (int pass = 0; pass < trainPasses; pass++)
        {
            float passLR = lr * MathF.Pow(0.85f, pass);
            float curLR = passLR;
            var gpuTrainer = new GpuTrainingMode(router)
            {
                LearningRate = passLR,
                EpochCount = epochs,
                LrDecayPerEpoch = decay
            };

            foreach (var metrics in gpuTrainer.Run(gpuTrainSet))
            {
                Console.WriteLine($"GPU epoch {++epoch:D3}  {metrics}  LR={curLR:F5}  {gpuTrainer.LastExecutionMessage}");
                curLR *= decay;
            }
        }

        if (tuneEpochs > 0 && !smoke)
        {
            var gpuTuner = new GpuTrainingMode(router)
            {
                LearningRate = tuneLR,
                EpochCount = tuneEpochs
            };
            int tuneEpoch = 0;
            foreach (var metrics in gpuTuner.Run(tuneSet))
                Console.WriteLine($"GPU tune epoch {++tuneEpoch:D2}  {metrics}  {gpuTuner.LastExecutionMessage}");
        }

        grid.SaveNails(smoke ? "prm_nails_gpu_smoke.bin" : "prm_nails_gpu.bin");
        if (!smoke)
            SaveActiveConfig(config);
        Console.WriteLine(smoke
            ? "\nGPU smoke nails saved → prm_nails_gpu_smoke.bin"
            : "\nGPU-trained nails saved → prm_nails_gpu.bin");
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
        RunOptimization(vocab, dataset, trainSet, testSet, tuneSet, valSet, window);
        break;
    }

    // ── AUTOOPTIMIZE (real hill-climbing convergence loop) ────────────────
    case "autooptimize":
    {
        bool useGpuTraining = modeArgs.Any(a => a.Equals("--gpu", StringComparison.OrdinalIgnoreCase));
        var autoArgs = modeArgs.Skip(1)
            .Where(a => !a.Equals("--gpu", StringComparison.OrdinalIgnoreCase))
            .ToArray();
        int   maxIter = autoArgs.Length > 0 && int.TryParse(autoArgs[0], out var parsedIter) ? parsedIter : 500;
        float target  = autoArgs.Length > 1 && float.TryParse(autoArgs[1], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var parsedTarget) ? parsedTarget : 0.60f;
        AutoOptimizer.Run(vocab, tokenIds, maxIter, target, useGpuTraining);
        break;
    }

    default:
        Console.WriteLine($"Unknown mode '{mode}'. Use: train | gputrain | test | tune | val | benchmark | optimize | autooptimize | viz | gpu");
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
                vizIds = GetRandomVizSample(VocabularyBuilder.Tokenise(corpus).Select(t => vocab.FirstOrDefault(v => v.Text == t)?.Id ?? -1).Where(id => id >= 0).ToArray(), window);
            }
        }
        else
        {
            vizIds = GetRandomVizSample(VocabularyBuilder.Tokenise(corpus).Select(t => vocab.FirstOrDefault(v => v.Text == t)?.Id ?? -1).Where(id => id >= 0).ToArray(), window);
        }

        var vizSequences = BuildVizSequences(tokenIds, vocab, window);
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
            await server.SendConfigAsync(gridInfo, sequences: vizSequences);
            await Task.Delay(200);
            await VizStreamAsync(server, grid, vocab, vizIds);

            // REPL + browser play requests. The WebSocket path exists only in viz mode.
            var consoleTask = StartConsoleRead(exitCts.Token);
            var wsTask      = server.ReceiveMessageAsync(exitCts.Token);
            while (!exitCts.IsCancellationRequested)
            {
                Task<string?> completed;
                try { completed = await Task.WhenAny(consoleTask, wsTask); }
                catch (OperationCanceledException) { goto nextClient; }

                if (completed == wsTask)
                {
                    string? json;
                    try { json = await wsTask; }
                    catch (OperationCanceledException) { goto nextClient; }
                    catch { break; }
                    if (json == null) break;

                    wsTask = server.ReceiveMessageAsync(exitCts.Token);
                    if (TryParseBrowserPlay(json, vocab, out var browserIds))
                    {
                        vizIds = browserIds;
                        await VizStreamAsync(server, grid, vocab, vizIds);
                    }
                    continue;
                }

                string? line;
                try { line = await consoleTask; }
                catch (OperationCanceledException) { goto nextClient; }
                if (line == null) break;
                consoleTask = StartConsoleRead(exitCts.Token);

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

static VizSequence[] BuildVizSequences(int[] corpusTokenIds, VocabToken[] vocab, int window)
{
    if (window <= 0 || corpusTokenIds.Length < window) return [];

    var sequences = new List<VizSequence>();
    for (int i = 0; i <= corpusTokenIds.Length - window; i++)
    {
        var ids = corpusTokenIds[i..(i + window)];
        if (ids.Any(id => id < 0 || id >= vocab.Length)) continue;
        var label = string.Join(" ", ids.Select(id => vocab[id].Text.Trim()));
        sequences.Add(new VizSequence(ids, label));
    }
    return sequences.ToArray();
}

static Task<string?> StartConsoleRead(CancellationToken ct)
{
    Console.Write("tokens (or Enter to replay)> ");
    return Task.Run(() => Console.ReadLine(), ct);
}

static void RunGpuDiagnostics()
{
    var devices = FlatPrmGpuBackend.DiscoverOpenClDevices();
    Console.WriteLine("OpenCL devices:");
    if (devices.Count == 0)
    {
        Console.WriteLine("  none found");
    }
    else
    {
        foreach (var d in devices)
        {
            Console.WriteLine(
                $"  [{d.Index}] {d.Name} | vendor={d.Vendor} | platform={d.Platform} | type={d.DeviceType} | gpu={d.IsGpu}");
        }
    }

    FlatPrmGpuSelfCheck.TryRunCpuGpuParity(out var result);
    Console.WriteLine(result.Message);
    if (result.Device is { } device)
        Console.WriteLine($"Parity device: [{device.Index}] {device.Name}");
    if (result.Comparison is { } comparison)
    {
        Console.WriteLine(
            $"Compared={comparison.ComparedCount}, countDelta={comparison.CountDelta}, " +
            $"maxPositionDelta={comparison.MaxPositionDelta}, maxVelocityDelta={comparison.MaxVelocityDelta}");
    }

    FlatPrmGpuSelfCheck.TryRunCpuGpuTrainingUpdateParity(out var trainingResult);
    Console.WriteLine(trainingResult.Message);
    if (trainingResult.Device is { } trainingDevice)
        Console.WriteLine($"Training parity device: [{trainingDevice.Index}] {trainingDevice.Name}");
    if (trainingResult.TokenOffsetXComparison is { } tokenX &&
        trainingResult.TokenOffsetYComparison is { } tokenY &&
        trainingResult.SharedOffsetXComparison is { } sharedX &&
        trainingResult.SharedOffsetYComparison is { } sharedY)
    {
        Console.WriteLine(
            $"Training deltas: tokenX={tokenX.MaxDelta}, tokenY={tokenY.MaxDelta}, " +
            $"sharedX={sharedX.MaxDelta}, sharedY={sharedY.MaxDelta}");
    }

    FlatPrmGpuSelfCheck.TryRunCpuGpuFullTrainingParity(out var fullTrainingResult);
    Console.WriteLine(fullTrainingResult.Message);
    if (fullTrainingResult.Device is { } fullDevice)
        Console.WriteLine($"Full training parity device: [{fullDevice.Index}] {fullDevice.Name}");
    if (fullTrainingResult.BallComparison is { } ballComparison &&
        fullTrainingResult.TokenOffsetXComparison is { } fullTokenX &&
        fullTrainingResult.TokenOffsetYComparison is { } fullTokenY &&
        fullTrainingResult.SharedOffsetXComparison is { } fullSharedX &&
        fullTrainingResult.SharedOffsetYComparison is { } fullSharedY)
    {
        Console.WriteLine(
            $"Full training deltas: pos={ballComparison.MaxPositionDelta}, vel={ballComparison.MaxVelocityDelta}, " +
            $"tokenX={fullTokenX.MaxDelta}, tokenY={fullTokenY.MaxDelta}, " +
            $"sharedX={fullSharedX.MaxDelta}, sharedY={fullSharedY.MaxDelta}, " +
            $"active={fullTrainingResult.ActiveStateMatches}, contacts={fullTrainingResult.ContactStateMatches}");
    }
}

static bool TryParseBrowserPlay(string json, VocabToken[] vocab, out int[] ids)
{
    ids = [];
    try
    {
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;
        if (!root.TryGetProperty("type", out var type) ||
            !string.Equals(type.GetString(), "play", StringComparison.OrdinalIgnoreCase) ||
            !root.TryGetProperty("ids", out var arr) ||
            arr.ValueKind != JsonValueKind.Array)
            return false;

        var parsed = new List<int>();
        foreach (var item in arr.EnumerateArray())
        {
            if (item.TryGetInt32(out int id) && id >= 0 && id < vocab.Length)
                parsed.Add(id);
        }
        ids = parsed.Take(4).ToArray();
        return ids.Length >= 2;
    }
    catch (JsonException)
    {
        return false;
    }
}

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
        var events    = r < trace.RowEvents.Length       ? trace.RowEvents[r]       : [];
        int rowNum    = r < trace.TotalRows              ? r                        : trace.TotalRows;
        await server.SendFrameAsync(balls, nailXs, offXs, nailRadii, nailRes, events, rowNum);
    }

    var (predicted, _) = grid.Predict(tokenIds);
    string predLabel   = predicted >= 0 && predicted < vocab.Length ? vocab[predicted].Text.Trim() : "?";
    await server.SendResultAsync(predLabel, target: null, correct: false);
    Console.WriteLine($"[viz] [{string.Join(", ", labels)}] → \"{predLabel}\"");
}

static int[] GetRandomVizSample(int[] tokenIds, int window)
{
    int sampleSize = Math.Clamp(window, 2, Math.Max(2, tokenIds.Length));
    if (tokenIds.Length <= sampleSize) return tokenIds;
    var rng = new Random();
    int start = rng.Next(0, tokenIds.Length - sampleSize + 1);
    return tokenIds[start..(start + sampleSize)];
}

static (int Window,
        List<(int[] input, int target)> Dataset,
        List<(int[] input, int target)> TrainSet,
        List<(int[] input, int target)> TuneSet,
        List<(int[] input, int target)> ValSet,
        List<(int[] input, int target)> TestSet) BuildDataset(int[] tokenIds, int requestedWindow)
{
    int maxWindow = Math.Max(1, tokenIds.Length - 1);
    int window = Math.Clamp(Math.Max(requestedWindow, MinComparableContextWindow), 1, maxWindow);

    var dataset = new List<(int[] input, int target)>();
    for (int targetIndex = 1; targetIndex < tokenIds.Length; targetIndex++)
    {
        int contextLength = Math.Min(window, targetIndex);
        int start = targetIndex - contextLength;
        dataset.Add((tokenIds[start..targetIndex], tokenIds[targetIndex]));
    }

    if (dataset.Count == 0 && tokenIds.Length > 1)
        dataset.Add((tokenIds[..1], tokenIds[1]));

    // Shuffle so all sentences mix across train/val/test — without this, rare words
    // in the last sentences are never seen during training (sequential split problem).
    var shuffleRng = new Random(42);
    for (int i = dataset.Count - 1; i > 0; i--)
    {
        int j = shuffleRng.Next(i + 1);
        (dataset[i], dataset[j]) = (dataset[j], dataset[i]);
    }

    if (dataset.Count == 0)
        return (window, dataset, [], [], [], []);

    // 70 / 10 / 20 split: train on fresh 70%, fine-tune on fresh 10%, validate on fresh 20%.
    // tuneSet is carved out AFTER trainSet so it is genuinely unseen during main training.
    int trainCount = Math.Clamp((int)Math.Round(dataset.Count * 0.70), 1, dataset.Count);
    int tuneCount  = dataset.Count - trainCount > 1
        ? Math.Clamp((int)Math.Round(dataset.Count * 0.10), 1, dataset.Count - trainCount - 1)
        : 0;

    var trainSet = dataset.Take(trainCount).ToList();
    var tuneSet  = dataset.Skip(trainCount).Take(tuneCount).ToList();
    var valSet   = dataset.Skip(trainCount + tuneCount).ToList();

    if (tuneSet.Count == 0) tuneSet = trainSet;
    if (valSet.Count == 0)  valSet  = dataset.TakeLast(1).ToList();

    return (window, dataset, trainSet, tuneSet, valSet, valSet);
}

static DiamondConfig EnsureComparableContext(DiamondConfig cfg, int tokenCount)
{
    int maxWindow = Math.Max(1, tokenCount - 1);
    int contextWindow = Math.Clamp(Math.Max(cfg.InputWindowSize, MinComparableContextWindow), 1, maxWindow);
    if (contextWindow == cfg.InputWindowSize) return cfg;

    return new DiamondConfig
    {
        RoleName = cfg.RoleName,
        EntryWidth = cfg.EntryWidth,
        MaxWidth = cfg.MaxWidth,
        WideningRows = cfg.WideningRows,
        NarrowingRows = cfg.NarrowingRows,
        NailSpacing = cfg.NailSpacing,
        DefaultRadius = cfg.DefaultRadius,
        DeflectionAlpha = cfg.DeflectionAlpha,
        DeflectionIdfPower = cfg.DeflectionIdfPower,
        DeflectionAlphaY = cfg.DeflectionAlphaY,
        SharedOffsetBlend = cfg.SharedOffsetBlend,
        ScoreDistanceSigma = cfg.ScoreDistanceSigma,
        ScoreProbeWeight = cfg.ScoreProbeWeight,
        PredictionProbeTrainingWeight = cfg.PredictionProbeTrainingWeight,
        ContextRelevanceDecay = cfg.ContextRelevanceDecay,
        ContextReinforcementStrength = cfg.ContextReinforcementStrength,
        DownstreamNailInfluence = cfg.DownstreamNailInfluence,
        DownstreamNailInfluenceRows = cfg.DownstreamNailInfluenceRows,
        DownstreamNailInfluenceRadius = cfg.DownstreamNailInfluenceRadius,
        DownstreamNailInfluenceDecay = cfg.DownstreamNailInfluenceDecay,
        DownstreamNailTargetDirectionality = cfg.DownstreamNailTargetDirectionality,
        ContextSummaryBallCount = cfg.ContextSummaryBallCount,
        ContextSummaryRow = cfg.ContextSummaryRow,
        ContextSummaryMassScale = cfg.ContextSummaryMassScale,
        ContextSummaryScoreWeight = cfg.ContextSummaryScoreWeight,
        GravityG = cfg.GravityG,
        ProximityBand = cfg.ProximityBand,
        CollisionRadius = cfg.CollisionRadius,
        DeltaTime = cfg.DeltaTime,
        InputWindowSize = contextWindow,
    };
}

static DiamondConfig PrepareGpuSmokeConfig(DiamondConfig cfg) => new()
{
    RoleName = cfg.RoleName,
    EntryWidth = cfg.EntryWidth,
    MaxWidth = cfg.MaxWidth,
    WideningRows = cfg.WideningRows,
    NarrowingRows = cfg.NarrowingRows,
    NailSpacing = cfg.NailSpacing,
    DefaultRadius = cfg.DefaultRadius,
    DeflectionAlpha = cfg.DeflectionAlpha,
    DeflectionIdfPower = cfg.DeflectionIdfPower,
    DeflectionAlphaY = cfg.DeflectionAlphaY,
    SharedOffsetBlend = cfg.SharedOffsetBlend,
    ScoreDistanceSigma = cfg.ScoreDistanceSigma,
    ScoreProbeWeight = cfg.ScoreProbeWeight,
    PredictionProbeTrainingWeight = cfg.PredictionProbeTrainingWeight,
    ContextRelevanceDecay = cfg.ContextRelevanceDecay,
    ContextReinforcementStrength = cfg.ContextReinforcementStrength,
    DownstreamNailInfluence = 0f,
    DownstreamNailInfluenceRows = 0,
    DownstreamNailInfluenceRadius = cfg.DownstreamNailInfluenceRadius,
    DownstreamNailInfluenceDecay = cfg.DownstreamNailInfluenceDecay,
    DownstreamNailTargetDirectionality = cfg.DownstreamNailTargetDirectionality,
    ContextSummaryBallCount = 0,
    ContextSummaryRow = cfg.ContextSummaryRow,
    ContextSummaryMassScale = cfg.ContextSummaryMassScale,
    ContextSummaryScoreWeight = cfg.ContextSummaryScoreWeight,
    GravityG = 0f,
    ProximityBand = cfg.ProximityBand,
    CollisionRadius = 0f,
    DeltaTime = cfg.DeltaTime,
    InputWindowSize = cfg.InputWindowSize,
};

static TrainOptions ParseTrainOptions(string[] modeArgs)
{
    var positional = new List<string>();
    int warmupIterations = 0;
    bool noOptimizer = false;

    for (int i = 1; i < modeArgs.Length; i++)
    {
        string arg = modeArgs[i];
        if (!arg.StartsWith("--", StringComparison.Ordinal))
        {
            positional.Add(arg);
            continue;
        }

        switch (arg)
        {
            case "--optimizer-warmup":
            case "--warmup":
                warmupIterations = 3;
                if (i + 1 < modeArgs.Length && int.TryParse(modeArgs[i + 1], out var parsedWarmup))
                {
                    warmupIterations = parsedWarmup;
                    i++;
                }
                break;
            case "--no-optimizer":
                noOptimizer = true;
                break;
            case "--gpu":
                break;
            default:
                Console.WriteLine($"Ignoring unknown train option: {arg}");
                break;
        }
    }

    int? epochs = positional.Count > 0 && int.TryParse(positional[0], out var parsedEpochs)
        ? parsedEpochs
        : null;
    float? lr = positional.Count > 1 && float.TryParse(positional[1], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var parsedLR)
        ? parsedLR
        : null;
    float? decay = positional.Count > 2 && float.TryParse(positional[2], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var parsedDecay)
        ? parsedDecay
        : null;

    return new TrainOptions(epochs, lr, decay, Math.Clamp(warmupIterations, 0, 10), noOptimizer, positional.Count > 0);
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
    List<(int[] input, int target)> valSet,
    int window)
{
    var candidates = new[]
    {
        new Candidate("baseline",     0.020f, 0.002f, 8, 8, 0.80f, 0.00f, 0.010f, 10f, 0.50f),
        new Candidate("sqrt-idf",     0.025f, 0.0025f, 10, 10, 0.90f, 0.50f, 0.012f, 12f, 0.45f),
        new Candidate("inverse-idf",  0.015f, 0.0015f, 12, 6, 0.75f, 1.00f, 0.008f, 8f, 0.70f),
        new Candidate("broad-think",  0.012f, 0.0012f, 14, 8, 0.70f, 0.25f, 0.006f, 7f, 0.85f),
        new Candidate("sharp-narrow", 0.025f, 0.003f, 6, 14, 0.95f, 0.75f, 0.014f, 14f, 0.35f),
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
            DefaultRadius = cand.DefaultDiameter,
            DeflectionAlpha = cand.DeflectionAlpha,
            DeflectionIdfPower = cand.DeflectionIdfPower,
            GravityG = cand.GravityG,
            ProximityBand = cand.ProximityBand,
            CollisionRadius = cand.CollisionRadius,
            InputWindowSize = window,
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
            $"rows={cand.WideningRows,2}/{cand.NarrowingRows,2} alpha={cand.DeflectionAlpha,4:F2} idf={cand.DeflectionIdfPower,4:F2} " +
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
        SaveBestConfig(bestCandidate, window);
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
        DefaultRadius = 0.75f
    };
}

static void SaveActiveConfig(DiamondConfig config)
{
    const string path = "prm_config.json";
    var json = JsonSerializer.Serialize(config, new JsonSerializerOptions { WriteIndented = true });
    File.WriteAllText(path, json);
}

static void SaveBestConfig(Candidate cand, int window)
{
    var cfg = new DiamondConfig
    {
        RoleName = "Analyst",
        WideningRows = cand.WideningRows,
        NarrowingRows = cand.NarrowingRows,
        MaxWidth = 50f,
        EntryWidth = 20f,
        DefaultRadius = cand.DefaultDiameter,
        DeflectionAlpha = cand.DeflectionAlpha,
        DeflectionIdfPower = cand.DeflectionIdfPower,
        GravityG = cand.GravityG,
        ProximityBand = cand.ProximityBand,
        CollisionRadius = cand.CollisionRadius,
        InputWindowSize = window,
    };

    SaveActiveConfig(cfg);
}

public sealed record TrainOptions(
    int? Epochs,
    float? LearningRate,
    float? Decay,
    int WarmupIterations,
    bool NoOptimizer,
    bool HasExplicitSchedule)
{
    public static TrainOptions Default { get; } = new(null, null, null, 0, false, false);
}
