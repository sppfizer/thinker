using PRM.Core.Engine;
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
        float targetAcc     = 0.60f)
    {
        Console.WriteLine();
        Console.WriteLine(new string('═', 70));
        Console.WriteLine($"  AUTO-OPTIMIZE  target={targetAcc:P0}  max-iters={maxIterations}");
        Console.WriteLine(new string('═', 70));

        var rng = new Random(42);

        // ── Load / build starting params ──────────────────────────────────────
        var currentParams = LoadBest(vocab);
        var (currentVal, currentTest, currentGrid) = TrainAndEval(currentParams, vocab, trainSet, tuneSet, valSet, testSet);

        Console.WriteLine($"\nSTART  val={currentVal:P1}  test={currentTest:P1}");
        PrintParams(currentParams, "       ");

        // Global best — never regresses even when exploring random restarts
        float globalBestVal   = currentVal;
        float globalBestTest  = currentTest;
        var   globalBestParams = currentParams;

        // Persist start as initial best
        currentGrid.SaveNails("prm_nails.bin");
        Save(currentParams);

        int stuckFor = 0;

        for (int iter = 1; iter <= maxIterations; iter++)
        {
            var cand = Perturb(currentParams, rng, stuckFor, vocab);
            var (valAcc, testAcc, grid) = TrainAndEval(cand, vocab, trainSet, tuneSet, valSet, testSet);

            char arrow = valAcc > currentVal ? '↑' : valAcc < currentVal ? '↓' : '=';
            string jumpTag = stuckFor >= 40 ? " [JUMP]" : stuckFor >= 20 ? " [nudge]" : "";

            Console.WriteLine(
                $"[{iter,4}] {arrow} val={valAcc:P1} test={testAcc:P1} | " +
                $"rows={cand.Config.WideningRows}/{cand.Config.NarrowingRows}({cand.WideningRatio:P0}w) " +
                $"α={cand.Config.DeflectionAlpha:F2} idf={cand.Config.DeflectionIdfPower:F2} αY={cand.Config.DeflectionAlphaY:F2} " +
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

            if (globalBestVal >= targetAcc)
            {
                Console.WriteLine($"\n🎯  TARGET REACHED: val={globalBestVal:P1} ≥ {targetAcc:P0}");
                break;
            }

            // Emergency: if completely stuck after many tries, do a random restart
            // but KEEP the globally best params/nails so we never go backward
            if (stuckFor >= 80)
            {
                var restartParams = RandomRestart(rng, vocab);
                var (rv, rt, rg)  = TrainAndEval(restartParams, vocab, trainSet, tuneSet, valSet, testSet);
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

        Console.WriteLine();
        Console.WriteLine(new string('═', 70));
        Console.WriteLine($"GLOBAL BEST: val={globalBestVal:P1}  test={globalBestTest:P1}");
        PrintParams(globalBestParams, "  ");
        Console.WriteLine(new string('═', 70));
    }

    // ── Train + evaluate ──────────────────────────────────────────────────────

    private static (float valAcc, float testAcc, DiamondGrid grid) TrainAndEval(
        HyperParams hp, VocabToken[] vocab,
        List<(int[] input, int target)> trainSet,
        List<(int[] input, int target)> tuneSet,
        List<(int[] input, int target)> valSet,
        List<(int[] input, int target)> testSet)
    {
        var grid   = new DiamondGrid(hp.Config, vocab, new Random(42));
        var router = new SpecialistRouter(new[] { grid });

        // Multiple forward passes: each pass reinforces nail positions from the previous pass.
        // Later passes act as fine-tuning on an already partially-learned structure.
        for (int pass = 0; pass < hp.TrainPasses; pass++)
        {
            float passLR = hp.LR * MathF.Pow(0.85f, pass); // slight LR decay between passes
            var trainer = new TrainingMode(router) { LearningRate = passLR, EpochCount = hp.TrainEpochs, LrDecayPerEpoch = 0.97f };
            foreach (var _ in trainer.Run(trainSet)) { }
        }

        if (hp.TuneEpochs > 0)
        {
            var tuner = new TuneMode(router) { LearningRate = hp.TuneLR, EpochCount = hp.TuneEpochs };
            foreach (var _ in tuner.Run(tuneSet)) { }
        }

        float valAcc  = new ValMode(router).Run(valSet).metrics.Accuracy;
        float testAcc = new TestMode(router).Run(testSet).Accuracy;

        return (valAcc, testAcc, grid);
    }

    // ── Perturbation strategies ───────────────────────────────────────────────

    private static HyperParams Perturb(HyperParams best, Random rng, int stuckFor, VocabToken[] vocab)
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

        var pool = Enumerable.Range(0, 16).ToList();
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
            InputWindowSize  = cfg.InputWindowSize,
        };

        return new HyperParams(newCfg, lr, tuneLr, trEp, tuEp, wRatio, passes, idfPow);
    }

    private static HyperParams RandomRestart(Random rng, VocabToken[] vocab)
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
            InputWindowSize  = 3,
        };

        float lr = rng.NextSingle() * 0.25f + 0.01f;
        return new HyperParams(cfg, lr, lr / 8f, rng.Next(minEp, maxEp), rng.Next(0, 20), wRatio, passes, idfPow);
    }

    // ── Config persistence ────────────────────────────────────────────────────

    private static HyperParams LoadBest(VocabToken[] vocab)
    {
        if (File.Exists(BestParamsPath))
        {
            var saved = JsonSerializer.Deserialize<HyperParams>(File.ReadAllText(BestParamsPath));
            if (saved is not null) return saved;
        }

        if (File.Exists(LegacyConfigPath))
        {
            var legacyCfg = JsonSerializer.Deserialize<DiamondConfig>(File.ReadAllText(LegacyConfigPath));
            if (legacyCfg is not null) return BuildDefaultHyperParams(vocab, legacyCfg);
        }

        return BuildDefaultHyperParams(vocab, BuildDefaultConfig(vocab));
    }

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
            InputWindowSize  = 3,
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
            $"{indent}rows={hp.Config.WideningRows}/{hp.Config.NarrowingRows}" +
            $"(ratio={hp.WideningRatio:P0} think) " +
            $"α={hp.Config.DeflectionAlpha:F3} idf={hp.Config.DeflectionIdfPower:F3} αY={hp.Config.DeflectionAlphaY:F3} " +
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
