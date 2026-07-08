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
    // ── Hyper-parameter bundle ─────────────────────────────────────────────────

    public record HyperParams(
        DiamondConfig Config,
        float LR,
        float TuneLR,
        int   TrainEpochs,
        int   TuneEpochs,
        float WideningRatio = 0.70f);   // fraction of total rows that are widening (thinking)

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
        var bestParams = LoadBest(vocab);
        var (bestVal, bestTest, bestGrid) = TrainAndEval(bestParams, vocab, trainSet, tuneSet, valSet, testSet);

        Console.WriteLine($"\nSTART  val={bestVal:P1}  test={bestTest:P1}");
        PrintParams(bestParams, "       ");

        // Global best — never regresses even when exploring random restarts
        float globalBestVal   = bestVal;
        float globalBestTest  = bestTest;
        var   globalBestParams = bestParams;

        // Persist start as initial best
        bestGrid.SaveNails("prm_nails.bin");
        Save(bestParams.Config);

        int stuckFor = 0;

        for (int iter = 1; iter <= maxIterations; iter++)
        {
            var cand = Perturb(bestParams, rng, stuckFor, vocab);
            var (valAcc, testAcc, grid) = TrainAndEval(cand, vocab, trainSet, tuneSet, valSet, testSet);

            char arrow = valAcc > bestVal ? '↑' : valAcc < bestVal ? '↓' : '=';
            string jumpTag = stuckFor >= 40 ? " [JUMP]" : stuckFor >= 20 ? " [nudge]" : "";

            Console.WriteLine(
                $"[{iter,4}] {arrow} val={valAcc:P1} test={testAcc:P1} | " +
                $"rows={cand.Config.WideningRows}/{cand.Config.NarrowingRows}({cand.WideningRatio:P0}w) " +
                $"α={cand.Config.DeflectionAlpha:F2} αY={cand.Config.DeflectionAlphaY:F2} " +
                $"sp={cand.Config.NailSpacing:F1} r={cand.Config.DefaultRadius:F2} LR={cand.LR:F3} ep={cand.TrainEpochs}+{cand.TuneEpochs}" +
                jumpTag);

            if (valAcc > bestVal)
            {
                bestVal    = valAcc;
                bestTest   = testAcc;
                bestParams = cand;
                bestGrid   = grid;
                stuckFor   = 0;

                bestGrid.SaveNails("prm_nails.bin");
                Save(bestParams.Config);
                Console.WriteLine($"       *** NEW BEST  val={bestVal:P1}  test={bestTest:P1} ***");

                if (valAcc > globalBestVal)
                {
                    globalBestVal    = valAcc;
                    globalBestTest   = testAcc;
                    globalBestParams = cand;
                }
            }
            else
            {
                stuckFor++;
            }

            if (bestVal >= targetAcc)
            {
                Console.WriteLine($"\n🎯  TARGET REACHED: val={bestVal:P1} ≥ {targetAcc:P0}");
                break;
            }

            // Emergency: if completely stuck after many tries, do a random restart
            // but KEEP the globally best params/nails so we never go backward
            if (stuckFor >= 80)
            {
                var restartParams = RandomRestart(rng, vocab);
                var (rv, rt, rg)  = TrainAndEval(restartParams, vocab, trainSet, tuneSet, valSet, testSet);
                Console.WriteLine($"       [RESTART] random restart: val={rv:P1}  test={rt:P1}");
                stuckFor = 0;
                // Only switch to restart if it beats the global best
                if (rv > bestVal)
                {
                    bestVal    = rv;
                    bestTest   = rt;
                    bestParams = restartParams;
                    bestGrid   = rg;
                    bestGrid.SaveNails("prm_nails.bin");
                    Save(bestParams.Config);
                    Console.WriteLine($"       Restart IS new best: val={bestVal:P1}");
                }
                else
                {
                    // Hill-climb from restart params but don't lose global best nails
                    Console.WriteLine($"       Continuing hill-climb from restart (global best still {bestVal:P1})");
                    bestParams = restartParams;  // explore new region
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

        var trainer = new TrainingMode(router) { LearningRate = hp.LR, EpochCount = hp.TrainEpochs, LrDecayPerEpoch = 0.97f };
        foreach (var _ in trainer.Run(trainSet)) { }

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

        var pool = Enumerable.Range(0, 14).ToList();
        Shuffle(pool, rng);

        for (int p = 0; p < nParams; p++)
        {
            float f = 1f + (float)(rng.NextDouble() * 2 - 1) * scale;
            float d = (float)(rng.NextDouble() * 2 - 1) * scale * 0.5f;
            switch (pool[p])
            {
                // Total depth — more rows = more routing steps = more memory
                case 0:  totalRows = Clamp((int)(totalRows * f), 6, 100);          break;
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

        return new HyperParams(newCfg, lr, tuneLr, trEp, tuEp, wRatio);
    }

    private static HyperParams RandomRestart(Random rng, VocabToken[] vocab)
    {
        float ns         = 2.0f;
        float entryWidth = vocab.Length * ns;
        float mw         = entryWidth * (rng.NextSingle() * 6f + 2f);
        int   totalRows  = rng.Next(10, 80);
        float wRatio     = rng.NextSingle() * 0.55f + 0.30f;           // 30–85%
        int   wRows      = Math.Max(3, (int)(totalRows * wRatio));
        int   nRows      = Math.Max(3, totalRows - wRows);

        var cfg = new DiamondConfig
        {
            RoleName         = "Analyst",
            WideningRows     = wRows,
            NarrowingRows    = nRows,
            DeflectionAlpha  = rng.NextSingle() * 2.5f + 0.1f,
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
        return new HyperParams(cfg, lr, lr / 8f, rng.Next(30, 200), rng.Next(0, 20), wRatio);
    }

    // ── Config persistence ────────────────────────────────────────────────────

    private static HyperParams LoadBest(VocabToken[] vocab)
    {
        float ns       = 2.0f;
        float minEntry = vocab.Length * ns;     // 40 tokens × 2 = 80 units
        float maxWidth = minEntry * 4f;         // 4× widening
        // Start at 70/30 thinking/summarizing split, 30 total rows
        int   totalRows = 30;
        float wRatio    = 0.70f;
        int   wRows     = (int)(totalRows * wRatio);   // 21 widening (thinking)
        int   nRows     = totalRows - wRows;           //  9 narrowing (summarizing)

        var cfg = new DiamondConfig
        {
            RoleName         = "Analyst",
            WideningRows     = wRows,
            NarrowingRows    = nRows,
            DeflectionAlpha  = 0.6f,
            DeflectionAlphaY = 0.15f,
            GravityG         = 0.01f,    // ball-to-ball attraction — part of original design
            CollisionRadius  = 0.5f,     // elastic collision between balls
            ProximityBand    = 8f,       // max distance for ball interaction
            DefaultRadius    = 0.5f,
            MaxWidth         = maxWidth,
            EntryWidth       = minEntry,
            NailSpacing      = ns,
            InputWindowSize  = 3,
        };
        return new HyperParams(cfg, LR: 0.15f, TuneLR: 0.02f,
                               TrainEpochs: 80, TuneEpochs: 10, WideningRatio: wRatio);
    }

    private static void Save(DiamondConfig cfg)
    {
        var json = JsonSerializer.Serialize(cfg, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText("prm_config.json", json);
    }

    // ── Display helpers ───────────────────────────────────────────────────────

    private static void PrintParams(HyperParams hp, string indent)
    {
        Console.WriteLine(
            $"{indent}rows={hp.Config.WideningRows}/{hp.Config.NarrowingRows}" +
            $"(ratio={hp.WideningRatio:P0} think) " +
            $"α={hp.Config.DeflectionAlpha:F3} αY={hp.Config.DeflectionAlphaY:F3} " +
            $"g={hp.Config.GravityG:F4} radius={hp.Config.DefaultRadius:F3} " +
            $"sp={hp.Config.NailSpacing:F1} MaxW={hp.Config.MaxWidth:F0} " +
            $"LR={hp.LR:F4} ep={hp.TrainEpochs}+{hp.TuneEpochs}");
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
