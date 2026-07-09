using PRM.Core.Engine;
using PRM.Core.Engine.Flat;
using PRM.Core.Models;
using PRM.Core.Modes;
using System.Text.Json;

/// <summary>
/// Hill-climbing auto-optimizer.
/// Starts from the current best configuration, perturbs one or more hyperparameters,
/// trains from scratch, compares val accuracy, and keeps improvements.
/// Rolls back worse results automatically.
/// Runs until target accuracy is reached or max iterations are exhausted.
/// </summary>
public static class AutoOptimizer
{
    private const string BestParamsPath = "prm_best_params.json";
    private const string LegacyConfigPath = "prm_config.json";
    private const int DefaultBatchSize = 8;
    private const int DefaultSurvivors = 3;
    private const float DefaultQuickEpochScale = 0.25f;
    private const float DefaultQuickSampleFraction = 0.45f;
    private const int MinComparableContextWindow = 16;

    private sealed record DatasetSplits(
        int Window,
        List<(int[] input, int target)> Dataset,
        List<(int[] input, int target)> TrainSet,
        List<(int[] input, int target)> TuneSet,
        List<(int[] input, int target)> ValSet,
        List<(int[] input, int target)> TestSet);

    private readonly record struct SearchOptions(
        int BatchSize,
        int Survivors,
        float QuickEpochScale,
        float QuickSampleFraction,
        float FullEpochScale,
        float FullSampleFraction);

    // ── Hyper-parameter bundle ─────────────────────────────────────────────────

    public record HyperParams(
        DiamondConfig Config,
        float LR,
        float TuneLR,
        int   TrainEpochs,
        int   TuneEpochs,
        float WideningRatio = 0.70f,  // fraction of total rows that are widening (thinking)
        int   TrainPasses   = 1,      // how many times to re-run the full training set before eval
        float DeflectionIdfPower = 0f); // 0=flat, 0.5=sqrt-IDF, 1=inverse-mass

    // ── Entry point ───────────────────────────────────────────────────────────

    public static void Run(
        VocabToken[]                        vocab,
        List<(int[] input, int target)>     trainSet,
        List<(int[] input, int target)>     tuneSet,
        List<(int[] input, int target)>     valSet,
        List<(int[] input, int target)>     testSet,
        int   maxIterations = 500,
        float targetAcc     = 0.60f,
        bool  useGpuTraining = false)
    {
        var window = trainSet.Count > 0 ? trainSet[0].input.Length : 3;
        var fixedSplits = new DatasetSplits(
            window,
            trainSet.Concat(tuneSet).Concat(valSet).ToList(),
            trainSet,
            tuneSet,
            valSet,
            testSet);

        RunInternal(
            vocab,
            tokenIds: null,
            fixedSplits,
            maxIterations,
            targetAcc,
            new SearchOptions(DefaultBatchSize, DefaultSurvivors, DefaultQuickEpochScale, DefaultQuickSampleFraction, 1f, 1f),
            allowWindowSearch: false,
            useGpuTraining,
            title: "AUTO-OPTIMIZE");
    }

    public static void Run(
        VocabToken[] vocab,
        int[]        tokenIds,
        int          maxIterations = 500,
        float        targetAcc     = 0.60f,
        bool         useGpuTraining = false)
    {
        RunInternal(
            vocab,
            tokenIds,
            fixedSplits: null,
            maxIterations,
            targetAcc,
            new SearchOptions(DefaultBatchSize, DefaultSurvivors, DefaultQuickEpochScale, DefaultQuickSampleFraction, 1f, 1f),
            allowWindowSearch: true,
            useGpuTraining,
            title: "AUTO-OPTIMIZE");
    }

    public static void RunWarmup(
        VocabToken[] vocab,
        int[]        tokenIds,
        int          maxIterations = 3,
        float        targetAcc     = 0.60f)
    {
        maxIterations = Clamp(maxIterations, 1, 10);
        RunInternal(
            vocab,
            tokenIds,
            fixedSplits: null,
            maxIterations,
            targetAcc,
            new SearchOptions(BatchSize: 4, Survivors: 1, QuickEpochScale: 0.15f, QuickSampleFraction: 0.30f, FullEpochScale: 0.25f, FullSampleFraction: 0.50f),
            allowWindowSearch: true,
            useGpuTraining: false,
            title: "OPTIMIZER WARMUP");
    }

    private static void RunInternal(
        VocabToken[]    vocab,
        int[]?          tokenIds,
        DatasetSplits?  fixedSplits,
        int             maxIterations,
        float           targetAcc,
        SearchOptions   options,
        bool            allowWindowSearch,
        bool            useGpuTraining,
        string          title)
    {
        Console.WriteLine();
        Console.WriteLine(new string('═', 70));
        Console.WriteLine($"  {title}  target={targetAcc:P0}  max-iters={maxIterations}  batch={options.BatchSize}");
        if (useGpuTraining)
            Console.WriteLine("  training backend: GPU when candidate config is supported, CPU fallback otherwise");
        if (options.FullEpochScale < 1f || options.FullSampleFraction < 1f)
            Console.WriteLine($"  bounded: full-eval epochs={options.FullEpochScale:P0} samples={options.FullSampleFraction:P0}");
        Console.WriteLine(new string('═', 70));

        var rng = new Random(42);
        var splitCache = new Dictionary<int, DatasetSplits>();

        DatasetSplits SplitsFor(HyperParams hp) =>
            GetSplits(hp, tokenIds, fixedSplits, splitCache);

        // ── Load / build starting params ──────────────────────────────────────
        var currentParams = LoadBest(vocab);
        var currentSplits = SplitsFor(currentParams);
        var (currentVal, currentTest, currentGrid) =
            TrainAndEval(currentParams, vocab, currentSplits, options.FullEpochScale, options.FullSampleFraction, useGpuTraining: useGpuTraining);

        Console.WriteLine($"\nSTART  val={currentVal:P1}  test={currentTest:P1}  window={currentSplits.Window}");
        PrintParams(currentParams, "       ");

        // Global best — never regresses even when exploring random restarts.
        float globalBestVal    = currentVal;
        float globalBestTest   = currentTest;
        var   globalBestParams = currentParams;

        // Persist start as initial best.
        currentGrid.SaveNails("prm_nails.bin");
        Save(currentParams);

        int stuckFor = 0;

        for (int iter = 1; iter <= maxIterations;)
        {
            int batchSize = options.BatchSize + (stuckFor >= 40 ? 4 : stuckFor >= 20 ? 2 : 0);
            var batch = BuildCandidateBatch(currentParams, rng, stuckFor, vocab, batchSize, allowWindowSearch);
            var probes = new List<(HyperParams Params, float QuickVal)>(batch.Count);

            foreach (var cand in batch)
            {
                var splits = SplitsFor(cand);
                var (quickVal, _, _) = TrainAndEval(
                    cand,
                    vocab,
                    splits,
                    options.QuickEpochScale,
                    options.QuickSampleFraction,
                    includeTest: false,
                    useGpuTraining);
                probes.Add((cand, quickVal));
            }

            var ranked = probes.OrderByDescending(p => p.QuickVal).ToList();
            float pruneFloor = Math.Max(0f, currentVal - (stuckFor >= 20 ? 0.05f : 0.02f));
            var survivors = ranked
                .Where((p, idx) => idx == 0 || p.QuickVal >= pruneFloor)
                .Take(Math.Max(1, options.Survivors))
                .ToList();

            int pruned = Math.Max(0, ranked.Count - survivors.Count);
            Console.WriteLine(
                $"       batch quick best={ranked[0].QuickVal:P1}  survivors={survivors.Count}/{ranked.Count}  pruned={pruned}");

            foreach (var probe in survivors)
            {
                if (iter > maxIterations) break;

                var cand = probe.Params;
                var splits = SplitsFor(cand);
                var (valAcc, testAcc, grid) =
                    TrainAndEval(cand, vocab, splits, options.FullEpochScale, options.FullSampleFraction, useGpuTraining: useGpuTraining);

                char arrow = valAcc > currentVal ? '↑' : valAcc < currentVal ? '↓' : '=';
                string jumpTag = stuckFor >= 40 ? " [JUMP]" : stuckFor >= 20 ? " [nudge]" : "";

                Console.WriteLine(
                    $"[{iter,4}] {arrow} q={probe.QuickVal:P1} val={valAcc:P1} test={testAcc:P1} | " +
                    $"win={cand.Config.InputWindowSize} rows={cand.Config.WideningRows}/{cand.Config.NarrowingRows}({cand.WideningRatio:P0}w) " +
                    $"α={cand.Config.DeflectionAlpha:F2} idf={cand.Config.DeflectionIdfPower:F2} αY={cand.Config.DeflectionAlphaY:F2} " +
                    $"share={cand.Config.SharedOffsetBlend:F2} σ={cand.Config.ScoreDistanceSigma:F2} probe={cand.Config.ScoreProbeWeight:F2} " +
                    $"down={cand.Config.DownstreamNailInfluence:F2}/{cand.Config.DownstreamNailInfluenceRows}/{cand.Config.DownstreamNailInfluenceRadius:F1}/{cand.Config.DownstreamNailInfluenceDecay:F1} " +
                    $"probeTrain={cand.Config.PredictionProbeTrainingWeight:F2} ctxDecay={cand.Config.ContextRelevanceDecay:F2} sum={cand.Config.ContextSummaryBallCount} " +
                    $"sp={cand.Config.NailSpacing:F1} r={cand.Config.DefaultRadius:F2} LR={cand.LR:F3} ep={cand.TrainEpochs}×{cand.TrainPasses}+{cand.TuneEpochs}" +
                    jumpTag);

                if (valAcc > currentVal)
                {
                    currentVal    = valAcc;
                    currentTest   = testAcc;
                    currentParams = cand;
                    currentGrid   = grid;
                    stuckFor      = 0;

                    if (valAcc > globalBestVal)
                    {
                        globalBestVal    = valAcc;
                        globalBestTest   = testAcc;
                        globalBestParams = cand;
                        currentGrid.SaveNails("prm_nails.bin");
                        Save(cand);
                        Console.WriteLine($"       *** GLOBAL BEST  val={globalBestVal:P1}  test={globalBestTest:P1} ***");
                    }
                    else
                    {
                        Console.WriteLine($"       *** local best  val={currentVal:P1}  test={currentTest:P1} ***");
                    }
                }
                else
                {
                    stuckFor++;
                }

                iter++;

                if (globalBestVal >= targetAcc)
                {
                    Console.WriteLine($"\n🎯  TARGET REACHED: val={globalBestVal:P1} ≥ {targetAcc:P0}");
                    iter = maxIterations + 1;
                    break;
                }

                // Emergency: if completely stuck after many tries, do a random restart
                // but KEEP the globally best params/nails so we never go backward.
                if (stuckFor >= 80 && iter <= maxIterations)
                {
                    var restartParams = RandomRestart(rng, vocab, allowWindowSearch);
                    var restartSplits = SplitsFor(restartParams);
                    var (rv, rt, rg) = TrainAndEval(
                        restartParams,
                        vocab,
                        restartSplits,
                        options.FullEpochScale,
                        options.FullSampleFraction,
                        useGpuTraining: useGpuTraining);
                    Console.WriteLine($"       [RESTART] val={rv:P1}  test={rt:P1}");
                    stuckFor      = 0;
                    currentVal    = rv;
                    currentTest   = rt;
                    currentParams = restartParams;
                    currentGrid   = rg;
                    if (rv > globalBestVal)
                    {
                        globalBestVal    = rv;
                        globalBestTest   = rt;
                        globalBestParams = restartParams;
                        currentGrid.SaveNails("prm_nails.bin");
                        Save(restartParams);
                        Console.WriteLine($"       Restart IS global best: val={globalBestVal:P1}");
                    }
                }
            }
        }

        Console.WriteLine();
        Console.WriteLine(new string('═', 70));
        Console.WriteLine($"GLOBAL BEST: val={globalBestVal:P1}  test={globalBestTest:P1}");
        PrintParams(globalBestParams, "  ");
        Console.WriteLine(new string('═', 70));
    }

    // ── Train + evaluate ──────────────────────────────────────────────────────

    private static (float valAcc, float testAcc, DiamondGrid grid) TrainAndEval(
        HyperParams  hp,
        VocabToken[] vocab,
        DatasetSplits splits,
        float epochScale = 1f,
        float sampleFraction = 1f,
        bool  includeTest = true,
        bool  useGpuTraining = false)
    {
        var grid   = new DiamondGrid(hp.Config, vocab, new Random(42));
        var router = new SpecialistRouter(new[] { grid });

        var trainSet = LimitSet(splits.TrainSet, sampleFraction);
        var tuneSet  = LimitSet(splits.TuneSet, sampleFraction);
        var valSet   = LimitSet(splits.ValSet, sampleFraction);
        var testSet  = includeTest ? LimitSet(splits.TestSet, sampleFraction) : valSet;

        int trainEpochs = Math.Max(1, (int)Math.Ceiling(hp.TrainEpochs * Math.Clamp(epochScale, 0.05f, 1f)));
        int tuneEpochs  = hp.TuneEpochs <= 0 ? 0 : Math.Max(1, (int)Math.Ceiling(hp.TuneEpochs * Math.Clamp(epochScale, 0.05f, 1f)));
        int trainPasses = epochScale >= 1f ? Math.Max(1, hp.TrainPasses) : 1;

        // Multiple forward passes: each pass reinforces nail positions from the previous pass.
        // Later passes act as fine-tuning on an already partially-learned structure.
        for (int pass = 0; pass < trainPasses; pass++)
        {
            float passLR = hp.LR * MathF.Pow(0.85f, pass); // slight LR decay between passes
            if (useGpuTraining && FlatPrmGpuTrainingRunner.IsSupported(hp.Config, out _))
            {
                var trainer = new GpuTrainingMode(router) { LearningRate = passLR, EpochCount = trainEpochs, LrDecayPerEpoch = 0.97f };
                foreach (var _ in trainer.Run(trainSet)) { }
            }
            else
            {
                var trainer = new TrainingMode(router) { LearningRate = passLR, EpochCount = trainEpochs, LrDecayPerEpoch = 0.97f };
                foreach (var _ in trainer.Run(trainSet)) { }
            }
        }

        if (tuneEpochs > 0)
        {
            if (useGpuTraining && FlatPrmGpuTrainingRunner.IsSupported(hp.Config, out _))
            {
                var tuner = new GpuTrainingMode(router) { LearningRate = hp.TuneLR, EpochCount = tuneEpochs };
                foreach (var _ in tuner.Run(tuneSet)) { }
            }
            else
            {
                var tuner = new TuneMode(router) { LearningRate = hp.TuneLR, EpochCount = tuneEpochs };
                foreach (var _ in tuner.Run(tuneSet)) { }
            }
        }

        float valAcc  = valSet.Count  > 0 ? new ValMode(router).Run(valSet).metrics.Accuracy : 0f;
        float testAcc = testSet.Count > 0 ? new TestMode(router).Run(testSet).Accuracy : valAcc;

        return (valAcc, testAcc, grid);
    }

    private static DatasetSplits GetSplits(
        HyperParams hp,
        int[]? tokenIds,
        DatasetSplits? fixedSplits,
        Dictionary<int, DatasetSplits> cache)
    {
        if (fixedSplits is not null) return fixedSplits;
        if (tokenIds is null) throw new InvalidOperationException("Token ids are required for dynamic context-window search.");

        int window = Math.Max(MinComparableContextWindow, hp.Config.InputWindowSize);
        if (!cache.TryGetValue(window, out var splits))
        {
            splits = BuildDataset(tokenIds, window);
            cache[window] = splits;
        }
        return splits;
    }

    private static DatasetSplits BuildDataset(int[] tokenIds, int requestedWindow)
    {
        int maxWindow = Math.Max(1, tokenIds.Length - 1);
        int window = Clamp(Math.Max(requestedWindow, MinComparableContextWindow), 1, maxWindow);

        var dataset = new List<(int[] input, int target)>();
        for (int targetIndex = 1; targetIndex < tokenIds.Length; targetIndex++)
        {
            int contextLength = Math.Min(window, targetIndex);
            int start = targetIndex - contextLength;
            dataset.Add((tokenIds[start..targetIndex], tokenIds[targetIndex]));
        }

        if (dataset.Count == 0 && tokenIds.Length > 1)
            dataset.Add((tokenIds[..1], tokenIds[1]));

        var shuffleRng = new Random(42);
        for (int i = dataset.Count - 1; i > 0; i--)
        {
            int j = shuffleRng.Next(i + 1);
            (dataset[i], dataset[j]) = (dataset[j], dataset[i]);
        }

        if (dataset.Count == 0)
            return new DatasetSplits(window, dataset, [], [], [], []);

        int trainCount = Math.Clamp((int)Math.Round(dataset.Count * 0.70), 1, dataset.Count);
        int tuneCount  = dataset.Count - trainCount > 1
            ? Math.Clamp((int)Math.Round(dataset.Count * 0.10), 1, dataset.Count - trainCount - 1)
            : 0;

        var trainSet = dataset.Take(trainCount).ToList();
        var tuneSet  = dataset.Skip(trainCount).Take(tuneCount).ToList();
        var valSet   = dataset.Skip(trainCount + tuneCount).ToList();

        if (tuneSet.Count == 0) tuneSet = trainSet;
        if (valSet.Count == 0)  valSet  = dataset.TakeLast(1).ToList();

        return new DatasetSplits(window, dataset, trainSet, tuneSet, valSet, valSet);
    }

    private static List<(int[] input, int target)> LimitSet(List<(int[] input, int target)> source, float fraction)
    {
        if (source.Count == 0 || fraction >= 0.999f) return source;
        int take = Math.Clamp((int)Math.Ceiling(source.Count * Math.Clamp(fraction, 0.01f, 1f)), 1, source.Count);
        return source.Take(take).ToList();
    }

    // ── Perturbation strategies ───────────────────────────────────────────────

    private static List<HyperParams> BuildCandidateBatch(
        HyperParams best,
        Random rng,
        int stuckFor,
        VocabToken[] vocab,
        int count,
        bool allowWindowSearch)
    {
        var candidates = new List<HyperParams>(count);
        var seen = new HashSet<string> { Signature(best) };
        int attempts = 0;

        AddFeatureFlipCandidates(best, candidates, seen, count);

        while (candidates.Count < count && attempts++ < count * 6)
        {
            HyperParams cand =
                stuckFor >= 60 && candidates.Count == count - 1
                    ? RandomRestart(rng, vocab, allowWindowSearch)
                    : Perturb(best, rng, stuckFor, vocab, allowWindowSearch);

            if (seen.Add(Signature(cand)))
                candidates.Add(cand);
        }

        if (candidates.Count == 0)
            candidates.Add(Perturb(best, rng, stuckFor, vocab, allowWindowSearch));

        return candidates;
    }

    private static void AddFeatureFlipCandidates(
        HyperParams best,
        List<HyperParams> candidates,
        HashSet<string> seen,
        int maxCount)
    {
        var cfg = best.Config;
        var featureConfigs = new[]
        {
            WithFeatureSettings(cfg, shared: cfg.SharedOffsetBlend, sigma: 0.5f, probe: 0.10f),
            WithFeatureSettings(cfg, shared: cfg.SharedOffsetBlend, sigma: 1.0f, probe: 0.20f),
            WithFeatureSettings(cfg, shared: cfg.SharedOffsetBlend, sigma: 1.5f, probe: 0.25f),
            WithDownstreamSettings(cfg, influence: 0.10f, rows: 2, radius: 1.5f, decay: 0.50f),
            WithDownstreamSettings(cfg, influence: 0.20f, rows: 3, radius: 2.0f, decay: 0.75f),
            WithDownstreamSettings(cfg, influence: 0.35f, rows: 4, radius: 2.5f, decay: 1.00f),
            WithModelDynamics(cfg, probeTrain: 0.15f, contextDecay: 0.08f, summaryCount: 0),
            WithModelDynamics(cfg, probeTrain: 0.30f, contextDecay: 0.12f, summaryCount: 1),
            WithModelDynamics(cfg, probeTrain: 0.45f, contextDecay: 0.18f, summaryCount: 2),
            WithTargetDirectionalDownstream(cfg, influence: 0.20f, rows: 3, directionality: 1.0f),
            WithFeatureSettings(cfg, shared: 0.25f, sigma: 1.0f, probe: 0.15f),
            WithFeatureSettings(cfg, shared: 0.45f, sigma: 1.5f, probe: 0.25f),
            WithFeatureSettings(cfg, shared: 0.0f,  sigma: 1.0f, probe: 0.20f),
            WithFeatureSettings(cfg, shared: 0.35f, sigma: 0.0f, probe: 0.0f),
        };

        foreach (var featureCfg in featureConfigs)
        {
            if (candidates.Count >= maxCount) break;
            var cand = best with
            {
                Config = featureCfg,
                LR = Math.Clamp(best.LR * 0.85f, 0.005f, 0.4f),
                TuneLR = Math.Clamp(best.TuneLR * 0.85f, 0.001f, 0.05f),
            };
            if (seen.Add(Signature(cand)))
                candidates.Add(cand);
        }
    }

    private static DiamondConfig WithFeatureSettings(DiamondConfig cfg, float shared, float sigma, float probe) =>
        new()
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
            SharedOffsetBlend = shared,
            ScoreDistanceSigma = sigma,
            ScoreProbeWeight = probe,
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
            InputWindowSize = cfg.InputWindowSize,
        };

    private static DiamondConfig WithModelDynamics(
        DiamondConfig cfg, float probeTrain, float contextDecay, int summaryCount) =>
        CopyConfig(cfg,
            predictionProbeTrainingWeight: probeTrain,
            contextRelevanceDecay: contextDecay,
            contextSummaryBallCount: summaryCount,
            contextSummaryRow: -1);

    private static DiamondConfig WithTargetDirectionalDownstream(
        DiamondConfig cfg, float influence, int rows, float directionality) =>
        CopyConfig(cfg,
            downstreamNailInfluence: influence,
            downstreamNailInfluenceRows: rows,
            downstreamNailTargetDirectionality: directionality);

    private static DiamondConfig WithDownstreamSettings(
        DiamondConfig cfg, float influence, int rows, float radius, float decay) =>
        new()
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
            DownstreamNailInfluence = influence,
            DownstreamNailInfluenceRows = rows,
            DownstreamNailInfluenceRadius = radius,
            DownstreamNailInfluenceDecay = decay,
            DownstreamNailTargetDirectionality = cfg.DownstreamNailTargetDirectionality,
            ContextSummaryBallCount = cfg.ContextSummaryBallCount,
            ContextSummaryRow = cfg.ContextSummaryRow,
            ContextSummaryMassScale = cfg.ContextSummaryMassScale,
            ContextSummaryScoreWeight = cfg.ContextSummaryScoreWeight,
            GravityG = cfg.GravityG,
            ProximityBand = cfg.ProximityBand,
            CollisionRadius = cfg.CollisionRadius,
            DeltaTime = cfg.DeltaTime,
            InputWindowSize = cfg.InputWindowSize,
        };

    private static string Signature(HyperParams hp) =>
        string.Join('|',
            hp.Config.InputWindowSize,
            hp.Config.WideningRows,
            hp.Config.NarrowingRows,
            Math.Round(hp.Config.DeflectionAlpha, 3),
            Math.Round(hp.Config.DeflectionAlphaY, 3),
            Math.Round(hp.Config.DeflectionIdfPower, 3),
            Math.Round(hp.Config.GravityG, 4),
            Math.Round(hp.Config.DefaultRadius, 3),
            Math.Round(hp.Config.NailSpacing, 3),
            Math.Round(hp.Config.SharedOffsetBlend, 3),
            Math.Round(hp.Config.ScoreDistanceSigma, 3),
            Math.Round(hp.Config.ScoreProbeWeight, 3),
            Math.Round(hp.Config.DownstreamNailInfluence, 3),
            hp.Config.DownstreamNailInfluenceRows,
            Math.Round(hp.Config.DownstreamNailInfluenceRadius, 3),
            Math.Round(hp.Config.DownstreamNailInfluenceDecay, 3),
            Math.Round(hp.Config.DownstreamNailTargetDirectionality, 3),
            Math.Round(hp.Config.PredictionProbeTrainingWeight, 3),
            Math.Round(hp.Config.ContextRelevanceDecay, 3),
            Math.Round(hp.Config.ContextReinforcementStrength, 3),
            hp.Config.ContextSummaryBallCount,
            hp.Config.ContextSummaryRow,
            Math.Round(hp.Config.ContextSummaryMassScale, 3),
            Math.Round(hp.Config.ContextSummaryScoreWeight, 3),
            Math.Round(hp.LR, 4),
            Math.Round(hp.TuneLR, 4),
            hp.TrainEpochs,
            hp.TuneEpochs,
            hp.TrainPasses);

    private static HyperParams Perturb(HyperParams best, Random rng, int stuckFor, VocabToken[] vocab, bool allowWindowSearch)
    {
        var cfg = best.Config;

        float scale  = stuckFor < 20  ? 0.20f
                     : stuckFor < 40  ? 0.40f
                                      : 0.70f;
        int nParams  = stuckFor < 20  ? 1
                     : stuckFor < 40  ? 2
                                      : 3;

        // TotalRows + WideningRatio (instead of independent rows) lets the optimizer
        // explore 70/30, 80/20, 50/50 thinking vs summarizing splits explicitly.
        int   totalRows = cfg.TotalRows;
        float wRatio    = best.WideningRatio;
        float alpha     = cfg.DeflectionAlpha;
        float grav      = cfg.GravityG;
        float prox      = cfg.ProximityBand;
        float colR      = cfg.CollisionRadius;
        float mw        = cfg.MaxWidth;
        float lr        = best.LR;
        float tuneLr    = best.TuneLR;
        int   trEp      = best.TrainEpochs;
        int   tuEp      = best.TuneEpochs;
        float alphaY    = cfg.DeflectionAlphaY;
        float radius    = cfg.DefaultRadius;
        float spacing   = cfg.NailSpacing;
        int   passes    = best.TrainPasses;
        float idfPow    = cfg.DeflectionIdfPower;
        int   window    = Math.Max(MinComparableContextWindow, cfg.InputWindowSize);
        float sharedBlend = cfg.SharedOffsetBlend;
        float scoreSigma  = cfg.ScoreDistanceSigma;
        float probeWeight = cfg.ScoreProbeWeight;
        float downInfluence = cfg.DownstreamNailInfluence;
        int downRows = cfg.DownstreamNailInfluenceRows;
        float downRadius = cfg.DownstreamNailInfluenceRadius;
        float downDecay = cfg.DownstreamNailInfluenceDecay;
        float downDirectionality = cfg.DownstreamNailTargetDirectionality;
        float probeTrain = cfg.PredictionProbeTrainingWeight;
        float contextDecay = cfg.ContextRelevanceDecay;
        float contextReinforce = cfg.ContextReinforcementStrength;
        int summaryCount = cfg.ContextSummaryBallCount;
        int summaryRow = cfg.ContextSummaryRow;
        float summaryMass = cfg.ContextSummaryMassScale;
        float summaryScore = cfg.ContextSummaryScoreWeight;

        var pool = Enumerable.Range(0, 32)
            .Where(i => allowWindowSearch || i != 16)
            .ToList();
        Shuffle(pool, rng);

        for (int p = 0; p < nParams; p++)
        {
            float f = 1f + (float)(rng.NextDouble() * 2 - 1) * scale;
            float d = (float)(rng.NextDouble() * 2 - 1) * scale * 0.5f;
            switch (pool[p])
            {
                // Total depth — more rows = more routing steps = more memory
                // When rows grow, scale up epochs proportionally so the model still converges
                case 0:
                {
                    int oldRows = totalRows;
                    totalRows = Clamp((int)(totalRows * f), 6, 200);
                    if (totalRows > oldRows)
                        trEp = Clamp((int)(trEp * ((float)totalRows / oldRows)), 10, 500);
                    break;
                }
                // Widening ratio — 0.70 = 70% thinking, 30% summarising
                case 1:  wRatio    = Clamp(wRatio + d, 0.30f, 0.85f);              break;
                case 2:  alpha     = Clamp(alpha  * f, 0.1f, 3.0f);               break;
                case 3:  // ball gravity — additive so it can escape from or return to 0
                         grav   = Clamp(grav + (float)(rng.NextDouble()-0.5)*0.02f*scale*3, 0.0f, 0.08f); break;
                case 4:  radius    = Clamp(radius * f, 0.1f, 1.0f);               break;
                case 5:  prox      = Clamp(prox   * f, 1f,   20f);                break;
                case 6:  // ball collision — additive so it can escape from or return to 0
                         colR   = Clamp(colR + (float)(rng.NextDouble()-0.5)*0.5f*scale*3,  0.0f, 3.0f);  break;
                case 7:  lr        = Clamp(lr     * f, 0.005f, 0.4f);             break;
                case 8:  tuneLr    = Clamp(tuneLr * f, 0.001f, 0.05f);            break;
                case 9:  trEp      = Clamp((int)(trEp * f), 10, 300);             break;
                case 10: tuEp      = Clamp((int)(tuEp * f), 0, 40);               break;
                // MaxWidth — wider thinking phase = more nails at widest point
                case 11: mw        = Clamp(mw * f, vocab.Length * spacing,
                                              vocab.Length * spacing * 10f);      break;
                case 12: alphaY    = Clamp(alphaY * f, 0.0f, 0.8f);              break;
                // NailSpacing — smaller = more nails per row = finer resolution
                case 13: spacing   = Clamp(spacing * f, 0.5f, 3.0f);             break;
                // TrainPasses — multiple forward re-passes through the training set
                case 14: passes    = Clamp((int)Math.Round(passes * f), 1, 5);   break;
                // Deflection IDF power — 0 flat, 0.5 sqrt, 1 inverse-mass
                case 15: idfPow    = Clamp(idfPow + d, 0.0f, 1.25f);            break;
                // Context window — number of previous tokens used for next-token prediction
                case 16:
                {
                    int delta = rng.NextDouble() < 0.5 ? -1 : 1;
                    window = Clamp(window + delta, MinComparableContextWindow, 64);
                    if (delta > 0)
                        trEp = Clamp((int)Math.Round(trEp * 1.10f), 10, 500);
                    break;
                }
                // Shared token routing prior — helps learning generalize across context positions.
                case 17: sharedBlend = Clamp(sharedBlend + d, 0.0f, 1.0f); break;
                // Soft output-slot scoring radius. 0 keeps legacy hard-bucket voting.
                case 18: scoreSigma = Clamp(scoreSigma + d * 4f, 0.0f, 6.0f); break;
                // Neutral probe-ball voting weight. 0 keeps legacy real-ball-only scoring.
                case 19: probeWeight = Clamp(probeWeight + d, 0.0f, 1.0f); break;
                // Downstream nail influence — forward-only local credit spreading.
                case 20: downInfluence = Clamp(downInfluence + d, 0.0f, 0.6f); break;
                case 21: downRows = Clamp(downRows + Math.Sign(d == 0f ? 1f : d), 0, 8); break;
                case 22: downRadius = Clamp(downRadius + d * 4f, 0.5f, 6.0f); break;
                case 23: downDecay = Clamp(downDecay + d * 2f, 0.0f, 3.0f); break;
                case 24: downDirectionality = Clamp(downDirectionality + d, 0.0f, 1.0f); break;
                case 25: probeTrain = Clamp(probeTrain + d, 0.0f, 1.0f); break;
                case 26: contextDecay = Clamp(contextDecay + d * 0.5f, 0.0f, 0.75f); break;
                case 27: contextReinforce = Clamp(contextReinforce * f, 0.0f, 4.0f); break;
                case 28: summaryCount = Clamp(summaryCount + Math.Sign(d == 0f ? 1f : d), 0, 4); break;
                case 29: summaryRow = Clamp(summaryRow + Math.Sign(d == 0f ? 1f : d), -1, 200); break;
                case 30: summaryMass = Clamp(summaryMass * f, 0.05f, 4.0f); break;
                case 31: summaryScore = Clamp(summaryScore * f, 0.05f, 4.0f); break;
            }
        }

        int wRows      = Math.Max(3, (int)Math.Round(totalRows * wRatio));
        int nRows      = Math.Max(3, totalRows - wRows);
        float entryW   = vocab.Length * spacing;

        var newCfg = new DiamondConfig
        {
            RoleName         = cfg.RoleName,
            WideningRows     = wRows,
            NarrowingRows    = nRows,
            DeflectionAlpha  = alpha,
            DeflectionAlphaY = alphaY,
            DeflectionIdfPower = idfPow,
            GravityG         = grav,
            DefaultRadius    = radius,
            ProximityBand    = prox,
            CollisionRadius  = colR,
            MaxWidth         = mw,
            EntryWidth       = entryW,
            NailSpacing      = spacing,
            DeltaTime        = cfg.DeltaTime,
            InputWindowSize  = window,
            SharedOffsetBlend = sharedBlend,
            ScoreDistanceSigma = scoreSigma,
            ScoreProbeWeight = probeWeight,
            DownstreamNailInfluence = downInfluence,
            DownstreamNailInfluenceRows = downRows,
            DownstreamNailInfluenceRadius = downRadius,
            DownstreamNailInfluenceDecay = downDecay,
            DownstreamNailTargetDirectionality = downDirectionality,
            PredictionProbeTrainingWeight = probeTrain,
            ContextRelevanceDecay = contextDecay,
            ContextReinforcementStrength = contextReinforce,
            ContextSummaryBallCount = summaryCount,
            ContextSummaryRow = summaryRow,
            ContextSummaryMassScale = summaryMass,
            ContextSummaryScoreWeight = summaryScore,
        };

        return new HyperParams(newCfg, lr, tuneLr, trEp, tuEp, wRatio, passes, idfPow);
    }

    private static HyperParams RandomRestart(Random rng, VocabToken[] vocab, bool allowWindowSearch)
    {
        float ns         = 2.0f;
        float entryWidth = vocab.Length * ns;
        float mw         = entryWidth * (rng.NextSingle() * 6f + 2f);
        // Scale rows and epochs with vocab size so restarts are viable for large vocabs
        int   minRows    = Math.Max(10, vocab.Length / 8);
        int   maxRows    = Math.Max(80, vocab.Length / 2);
        int   totalRows  = rng.Next(minRows, maxRows);
        float wRatio     = rng.NextSingle() * 0.55f + 0.30f;
        int   wRows      = Math.Max(3, (int)(totalRows * wRatio));
        int   nRows      = Math.Max(3, totalRows - wRows);
        int   minEp      = Math.Max(30, vocab.Length);
        int   maxEp      = Math.Max(200, vocab.Length * 3);
        int   passes     = rng.Next(1, vocab.Length > 60 ? 4 : 2);
        float idfPow     = rng.NextSingle() * 1.25f;
        int   window     = allowWindowSearch ? rng.Next(MinComparableContextWindow, 33) : MinComparableContextWindow;
        float sharedBlend = rng.NextSingle();
        float scoreSigma  = rng.NextSingle() < 0.25f ? 0f : rng.NextSingle() * 4f;
        float probeWeight = rng.NextSingle() < 0.25f ? 0f : rng.NextSingle();
        float downInfluence = rng.NextSingle() < 0.30f ? 0f : rng.NextSingle() * 0.45f;
        int downRows = downInfluence <= 0f ? 0 : rng.Next(1, 6);
        float downRadius = rng.NextSingle() * 3.5f + 0.75f;
        float downDecay = rng.NextSingle() * 1.5f;
        float downDirectionality = rng.NextSingle();
        float probeTrain = rng.NextSingle() < 0.35f ? 0f : rng.NextSingle() * 0.7f;
        float contextDecay = rng.NextSingle() < 0.35f ? 0f : rng.NextSingle() * 0.35f;
        float contextReinforce = rng.NextSingle() * 2.5f;
        int summaryCount = rng.NextSingle() < 0.50f ? 0 : rng.Next(1, 3);
        int summaryRow = rng.NextSingle() < 0.70f ? -1 : rng.Next(0, Math.Max(1, totalRows));
        float summaryMass = rng.NextSingle() * 2f + 0.25f;
        float summaryScore = rng.NextSingle() * 2f + 0.25f;

        var cfg = new DiamondConfig
        {
            RoleName         = "Analyst",
            WideningRows     = wRows,
            NarrowingRows    = nRows,
            DeflectionAlpha  = rng.NextSingle() * 2.5f + 0.1f,
            DeflectionIdfPower = idfPow,
            DeflectionAlphaY = rng.NextSingle() * 0.6f,
            GravityG         = rng.NextSingle() * 0.04f,
            DefaultRadius    = rng.NextSingle() * 0.9f + 0.05f,
            ProximityBand    = rng.NextSingle() * 20f + 2f,
            CollisionRadius  = rng.NextSingle() * 1.5f,
            MaxWidth         = mw,
            EntryWidth       = entryWidth,
            NailSpacing      = ns,
            InputWindowSize  = window,
            SharedOffsetBlend = sharedBlend,
            ScoreDistanceSigma = scoreSigma,
            ScoreProbeWeight = probeWeight,
            DownstreamNailInfluence = downInfluence,
            DownstreamNailInfluenceRows = downRows,
            DownstreamNailInfluenceRadius = downRadius,
            DownstreamNailInfluenceDecay = downDecay,
            DownstreamNailTargetDirectionality = downDirectionality,
            PredictionProbeTrainingWeight = probeTrain,
            ContextRelevanceDecay = contextDecay,
            ContextReinforcementStrength = contextReinforce,
            ContextSummaryBallCount = summaryCount,
            ContextSummaryRow = summaryRow,
            ContextSummaryMassScale = summaryMass,
            ContextSummaryScoreWeight = summaryScore,
        };

        float lr = rng.NextSingle() * 0.25f + 0.01f;
        return new HyperParams(cfg, lr, lr / 8f, rng.Next(minEp, maxEp), rng.Next(0, 20), wRatio, passes, idfPow);
    }

    // ── Config persistence ────────────────────────────────────────────────────

    public static bool TryLoadBest(VocabToken[] vocab, out HyperParams best)
    {
        if (TryReadBestParams(out var saved) && saved is not null)
        {
            best = Normalize(saved, vocab);
            return true;
        }

        best = LoadBest(vocab);
        return false;
    }

    private static HyperParams LoadBest(VocabToken[] vocab)
    {
        if (TryReadBestParams(out var saved) && saved is not null)
            return Normalize(saved, vocab);

        if (File.Exists(LegacyConfigPath))
        {
            var legacyCfg = JsonSerializer.Deserialize<DiamondConfig>(File.ReadAllText(LegacyConfigPath));
            if (legacyCfg is not null) return BuildDefaultHyperParams(vocab, legacyCfg);
        }

        return BuildDefaultHyperParams(vocab, BuildDefaultConfig(vocab));
    }

    private static bool TryReadBestParams(out HyperParams? hp)
    {
        hp = null;
        if (!File.Exists(BestParamsPath)) return false;

        try
        {
            hp = JsonSerializer.Deserialize<HyperParams>(File.ReadAllText(BestParamsPath));
            return hp is not null;
        }
        catch (JsonException)
        {
            return false;
        }
        catch (IOException)
        {
            return false;
        }
        catch (NotSupportedException)
        {
            return false;
        }
    }

    private static HyperParams Normalize(HyperParams hp, VocabToken[] vocab)
    {
        var cfg = hp.Config ?? BuildDefaultConfig(vocab);
        if (cfg.InputWindowSize < MinComparableContextWindow)
            cfg = CopyConfig(cfg, inputWindowSize: MinComparableContextWindow);

        return hp with
        {
            Config = cfg,
            TrainEpochs = Math.Max(1, hp.TrainEpochs),
            TuneEpochs = Math.Max(0, hp.TuneEpochs),
            TrainPasses = Math.Max(1, hp.TrainPasses),
            WideningRatio = hp.WideningRatio > 0f
                ? hp.WideningRatio
                : (float)cfg.WideningRows / Math.Max(cfg.TotalRows, 1),
            DeflectionIdfPower = cfg.DeflectionIdfPower
        };
    }

    private static DiamondConfig CopyConfig(
        DiamondConfig cfg,
        int? inputWindowSize = null,
        float? predictionProbeTrainingWeight = null,
        float? contextRelevanceDecay = null,
        float? contextReinforcementStrength = null,
        float? downstreamNailInfluence = null,
        int? downstreamNailInfluenceRows = null,
        float? downstreamNailInfluenceRadius = null,
        float? downstreamNailInfluenceDecay = null,
        float? downstreamNailTargetDirectionality = null,
        int? contextSummaryBallCount = null,
        int? contextSummaryRow = null,
        float? contextSummaryMassScale = null,
        float? contextSummaryScoreWeight = null) => new()
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
        PredictionProbeTrainingWeight = predictionProbeTrainingWeight ?? cfg.PredictionProbeTrainingWeight,
        ContextRelevanceDecay = contextRelevanceDecay ?? cfg.ContextRelevanceDecay,
        ContextReinforcementStrength = contextReinforcementStrength ?? cfg.ContextReinforcementStrength,
        DownstreamNailInfluence = downstreamNailInfluence ?? cfg.DownstreamNailInfluence,
        DownstreamNailInfluenceRows = downstreamNailInfluenceRows ?? cfg.DownstreamNailInfluenceRows,
        DownstreamNailInfluenceRadius = downstreamNailInfluenceRadius ?? cfg.DownstreamNailInfluenceRadius,
        DownstreamNailInfluenceDecay = downstreamNailInfluenceDecay ?? cfg.DownstreamNailInfluenceDecay,
        DownstreamNailTargetDirectionality = downstreamNailTargetDirectionality ?? cfg.DownstreamNailTargetDirectionality,
        ContextSummaryBallCount = contextSummaryBallCount ?? cfg.ContextSummaryBallCount,
        ContextSummaryRow = contextSummaryRow ?? cfg.ContextSummaryRow,
        ContextSummaryMassScale = contextSummaryMassScale ?? cfg.ContextSummaryMassScale,
        ContextSummaryScoreWeight = contextSummaryScoreWeight ?? cfg.ContextSummaryScoreWeight,
        GravityG = cfg.GravityG,
        ProximityBand = cfg.ProximityBand,
        CollisionRadius = cfg.CollisionRadius,
        DeltaTime = cfg.DeltaTime,
        InputWindowSize = inputWindowSize ?? cfg.InputWindowSize,
    };

    private static HyperParams BuildDefaultHyperParams(VocabToken[] vocab, DiamondConfig cfg)
    {
        int totalRows  = Math.Clamp(vocab.Length / 6, 20, 80);
        float wRatio   = (float)cfg.WideningRows / Math.Max(cfg.TotalRows, 1);
        int trainEpoch = Math.Clamp(vocab.Length * 4, 100, 600);
        int passes     = vocab.Length > 60 ? 2 : 1;  // multi-pass for larger vocab

        return new HyperParams(cfg, LR: 0.10f, TuneLR: 0.01f,
                               TrainEpochs: trainEpoch, TuneEpochs: 10,
                               WideningRatio: wRatio, TrainPasses: passes,
                               DeflectionIdfPower: cfg.DeflectionIdfPower);
    }

    private static DiamondConfig BuildDefaultConfig(VocabToken[] vocab)
    {
        float ns       = 2.0f;
        float minEntry = vocab.Length * ns;
        // Cap at 2× entry: 4× created a 27M-parameter grid for 209 tokens that
        // training (337K steps) could never fill — average 2 updates/param, no convergence.
        float maxWidth = minEntry * 2f;

        // Scale starting config with vocabulary size.
        // Larger vocab → more routing depth and more training needed.
        // Rule of thumb: ~1 row per 6 tokens (capped), ~4 epochs per token, 2 passes for large vocabs.
        int   totalRows  = Math.Clamp(vocab.Length / 6, 20, 80);
        float wRatio     = 0.75f;
        int   wRows      = (int)(totalRows * wRatio);
        int   nRows      = Math.Max(3, totalRows - wRows);
        float idfPow     = vocab.Length > 60 ? 0.5f : 0f;

        return new DiamondConfig
        {
            RoleName         = "Analyst",
            WideningRows     = wRows,
            NarrowingRows    = nRows,
            DeflectionAlpha  = 0.6f,
            DeflectionAlphaY = 0.15f,
            DeflectionIdfPower = idfPow,
            GravityG         = 0.01f,
            CollisionRadius  = 0.5f,
            ProximityBand    = 8f,
            DefaultRadius    = 0.5f,
            MaxWidth         = maxWidth,
            EntryWidth       = minEntry,
            NailSpacing      = ns,
            InputWindowSize  = MinComparableContextWindow,
        };
    }

    private static void Save(HyperParams hp)
    {
        var options = new JsonSerializerOptions { WriteIndented = true };
        File.WriteAllText(BestParamsPath, JsonSerializer.Serialize(hp, options));
        File.WriteAllText(LegacyConfigPath, JsonSerializer.Serialize(hp.Config, options));
    }

    // ── Display helpers ───────────────────────────────────────────────────────

    private static void PrintParams(HyperParams hp, string indent)
    {
        Console.WriteLine(
            $"{indent}win={hp.Config.InputWindowSize} rows={hp.Config.WideningRows}/{hp.Config.NarrowingRows}" +
            $"(ratio={hp.WideningRatio:P0} think) " +
            $"α={hp.Config.DeflectionAlpha:F3} idf={hp.Config.DeflectionIdfPower:F3} αY={hp.Config.DeflectionAlphaY:F3} " +
            $"share={hp.Config.SharedOffsetBlend:F3} σ={hp.Config.ScoreDistanceSigma:F3} probe={hp.Config.ScoreProbeWeight:F3} " +
            $"down={hp.Config.DownstreamNailInfluence:F3}/{hp.Config.DownstreamNailInfluenceRows}/{hp.Config.DownstreamNailInfluenceRadius:F2}/{hp.Config.DownstreamNailInfluenceDecay:F2} " +
            $"probeTrain={hp.Config.PredictionProbeTrainingWeight:F3} ctxDecay={hp.Config.ContextRelevanceDecay:F3} sum={hp.Config.ContextSummaryBallCount} " +
            $"g={hp.Config.GravityG:F4} radius={hp.Config.DefaultRadius:F3} " +
            $"sp={hp.Config.NailSpacing:F1} MaxW={hp.Config.MaxWidth:F0} " +
            $"LR={hp.LR:F4} ep={hp.TrainEpochs}×{hp.TrainPasses}+{hp.TuneEpochs}");
    }

    // ── Utility ───────────────────────────────────────────────────────────────

    private static float Clamp(float v, float lo, float hi) => Math.Max(lo, Math.Min(hi, v));
    private static int   Clamp(int v, int lo, int hi)       => Math.Max(lo, Math.Min(hi, v));

    private static void Shuffle<T>(List<T> list, Random rng)
    {
        for (int i = list.Count - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (list[i], list[j]) = (list[j], list[i]);
        }
    }
}
