using PRM.Core.Models;

namespace PRM.Core.Engine;

/// <summary>
/// The diamond grid for one specialist role.
/// Owns the nail array, simulator, and magnet field.
/// Provides forward pass (inference) and training pass.
/// </summary>
public class DiamondGrid
{
    public DiamondConfig Config { get; }
    public string        Role   => Config.RoleName;

    private readonly Nail[,]       _nails;
    private readonly BallSimulator _simulator;
    private readonly MagnetField   _magnet;
    private readonly VocabToken[]  _vocab;

    public DiamondGrid(DiamondConfig config, VocabToken[] vocab, Random? rng = null)
    {
        _vocab  = vocab;
        _magnet = new MagnetField(config);

        // ── Auto-scale grid width to vocabulary size ─────────────────────────
        // The output row (narrowest point) must have at least one nail per vocabulary
        // token so every token has a distinct addressable landing zone.
        // Min required output width = vocab.Length × NailSpacing.
        float minEntryWidth = vocab.Length * config.NailSpacing;
        if (config.EntryWidth < minEntryWidth)
        {
            float scale = minEntryWidth / config.EntryWidth;
            config = new DiamondConfig
            {
                RoleName         = config.RoleName,
                EntryWidth       = minEntryWidth,
                MaxWidth         = config.MaxWidth * scale,
                WideningRows     = config.WideningRows,
                NarrowingRows    = config.NarrowingRows,
                NailSpacing      = config.NailSpacing,
                DefaultRadius    = config.DefaultRadius,
                DeflectionAlpha  = config.DeflectionAlpha,
                DeflectionIdfPower = config.DeflectionIdfPower,
                DeflectionAlphaY = config.DeflectionAlphaY,
                GravityG         = config.GravityG,
                ProximityBand    = config.ProximityBand,
                CollisionRadius  = config.CollisionRadius,
                DeltaTime        = config.DeltaTime,
                InputWindowSize  = config.InputWindowSize,
            };
        }

        Config = config;

        // Allocate nail grid — rows × max_columns
        int maxCols = (int)(config.MaxWidth / config.NailSpacing) + 2;
        _nails = new Nail[config.TotalRows, maxCols];

        rng ??= new Random(42);
        InitNails(rng);

        // Pass vocab size so BallSimulator can maintain per-token tilt banks
        _simulator = new BallSimulator(config, _nails, _magnet, vocab.Length);
    }

    // ── Inference ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Forward pass with no nail updates.
    /// Returns (predicted token id, retained ball-mass ratio).
    /// </summary>
    public (int tokenId, float confidence) Predict(int[] inputTokenIds)
    {
        var balls = CreateBalls(inputTokenIds);
        var survivors = _simulator.Simulate(balls);
        return Score(survivors, balls);
    }

    /// <summary>
    /// Forward pass that also records the full trajectory for visualisation.
    /// Returns the GridTrace with ball positions at every row.
    /// </summary>
    public GridTrace SimulateWithTrace(int[] inputTokenIds)
    {
        var balls = CreateBalls(inputTokenIds);
        return _simulator.SimulateWithTrace(balls);
    }

    // ── Training ──────────────────────────────────────────────────────────────

    /// <summary>
    /// Training forward pass with live nail updates via magnetic force.
    /// Returns (predicted token id, correct, retained mass ratio).
    /// </summary>
    public (int predicted, bool correct, float confidence) Train(
        int[] inputTokenIds, int targetTokenId, float learningRate)
    {
        float targetCentre = SlotCentre(targetTokenId);
        var balls     = CreateBalls(inputTokenIds);
        var survivors = _simulator.Simulate(balls, targetCentre, learningRate);
        var (pred, conf) = Score(survivors, balls);
        if (pred == targetTokenId)
            _simulator.ReinforceContacts(survivors, targetCentre, learningRate);
        else
            _simulator.SoftenContacts(survivors, learningRate);
        return (pred, pred == targetTokenId, conf);
    }

    // ── Checkpointing ─────────────────────────────────────────────────────────

    public void SaveNails(string path)
    {
        using var bw = new BinaryWriter(File.Open(path, FileMode.Create));

        // Nail shared properties (radius, resistance)
        bw.Write(_nails.GetLength(0));
        bw.Write(_nails.GetLength(1));
        for (int r = 0; r < _nails.GetLength(0); r++)
        for (int c = 0; c < _nails.GetLength(1); c++)
        {
            bw.Write(_nails[r, c].Id);
            bw.Write(_nails[r, c].Radius);
            bw.Write(_nails[r, c].Resistance);
            bw.Write(_nails[r, c].Density);
        }

        // Per-token 2D offsets (X)
        var offX = _simulator.GetTokenOffX();
        bw.Write(offX.GetLength(0)); bw.Write(offX.GetLength(1)); bw.Write(offX.GetLength(2));
        for (int r = 0; r < offX.GetLength(0); r++)
        for (int c = 0; c < offX.GetLength(1); c++)
        for (int t = 0; t < offX.GetLength(2); t++)
            bw.Write(offX[r, c, t]);

        // Per-token 2D offsets (Y)
        var offY = _simulator.GetTokenOffY();
        bw.Write(offY.GetLength(0)); bw.Write(offY.GetLength(1)); bw.Write(offY.GetLength(2));
        for (int r = 0; r < offY.GetLength(0); r++)
        for (int c = 0; c < offY.GetLength(1); c++)
        for (int t = 0; t < offY.GetLength(2); t++)
            bw.Write(offY[r, c, t]);
    }

    public void LoadNails(string path)
    {
        using var br = new BinaryReader(File.Open(path, FileMode.Open));
        try
        {
            // Nail shared properties
            int rows = br.ReadInt32(), cols = br.ReadInt32();
            for (int r = 0; r < rows && r < _nails.GetLength(0); r++)
            for (int c = 0; c < cols && c < _nails.GetLength(1); c++)
            {
                _nails[r, c].Id          = br.ReadInt32();
                _nails[r, c].Radius     = br.ReadSingle();
                _nails[r, c].Resistance = br.ReadSingle();
                _nails[r, c].Density    = br.ReadSingle();
            }

            // Per-token offsets X
            int xr = br.ReadInt32(), xc = br.ReadInt32(), xt = br.ReadInt32();
            var loadedX = new float[xr, xc, xt];
            for (int r = 0; r < xr; r++)
            for (int c = 0; c < xc; c++)
            for (int t = 0; t < xt; t++)
                loadedX[r, c, t] = br.ReadSingle();

            // Per-token offsets Y
            int yr = br.ReadInt32(), yc = br.ReadInt32(), yt = br.ReadInt32();
            var loadedY = new float[yr, yc, yt];
            for (int r = 0; r < yr; r++)
            for (int c = 0; c < yc; c++)
            for (int t = 0; t < yt; t++)
                loadedY[r, c, t] = br.ReadSingle();

            _simulator.SetTokenOffsets(loadedX, loadedY);
        }
        catch (EndOfStreamException) { /* old/partial checkpoint — keep defaults */ }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private List<Ball> CreateBalls(int[] tokenIds)
    {
        var balls = new List<Ball>(tokenIds.Length + 1);
        float gridCentre = Config.MaxWidth / 2f;
        float spacing    = Config.EntryWidth / Math.Max(tokenIds.Length, 1);

        for (int i = 0; i < tokenIds.Length; i++)
        {
            int id = tokenIds[i];
            if (id < 0 || id >= _vocab.Length) continue;
            float x = gridCentre - Config.EntryWidth / 2f + spacing * i + spacing / 2f;
            balls.Add(new Ball(id, x, _vocab[id].Mass, contextPosition: i));
        }

        // Prediction slot ball (near-zero mass, position=-1 → uses last window slot)
        balls.Add(new Ball(-1, gridCentre, 0.001f, contextPosition: tokenIds.Length));
        return balls;
    }

    private (int tokenId, float confidence) Score(List<Ball> survivors, List<Ball> all)
    {
        float totalInput = all.Sum(b => b.Mass);
        var scores = new float[_vocab.Length];

        if (_vocab.Length == 0) return (0, 0f);

        // Ball positions are in grid coordinates [gridLeft, gridRight] at the output row.
        // Vocab slots are laid out in frequency-normalized units [0, totalSlotSpan].
        // We must normalize ball positions to slot space before matching.
        int lastRow = Config.TotalRows - 1;
        float gridLeft     = _simulator.LeftBorder(lastRow);
        float gridRight    = _simulator.RightBorder(lastRow);
        float gridSpan     = Math.Max(gridRight - gridLeft, 1e-6f);
        float totalSlotSpan = _vocab[^1].SlotRight;

        foreach (var ball in survivors)
        {
            if (ball.TokenId < 0) continue;   // skip the neutral probe ball

            // Map grid coord → slot coord
            float norm = (ball.Position - gridLeft) / gridSpan * totalSlotSpan;

            // Flat weight: 1 vote per ball regardless of rarity.
            // IDF belongs in the training nudge (rare tokens get stronger nail updates),
            // not here — otherwise rare balls dominate every prediction regardless of routing.
            const float weight = 1f;

            for (int t = 0; t < _vocab.Length; t++)
            {
                if (norm >= _vocab[t].SlotLeft && norm < _vocab[t].SlotRight)
                {
                    scores[t] += weight;
                    break;
                }
            }
        }

        int winner    = Array.IndexOf(scores, scores.Max());
        float retMass = survivors.Sum(b => b.Mass);
        float confidence = totalInput > 0 ? retMass / totalInput : 0f;
        return (winner, confidence);
    }

    /// <summary>
    /// Returns the x grid-coordinate that corresponds to the centre of the given token's
    /// output slot at the bottom (output) row of the diamond.
    /// Used by training so the magnet target is in the same coordinate space as ball positions.
    /// </summary>
    private float SlotCentre(int tokenId)
    {
        if (tokenId < 0 || tokenId >= _vocab.Length) return Config.MaxWidth / 2f;

        // Slot centre in frequency-normalized units
        float slotCentre    = _vocab[tokenId].SlotLeft + _vocab[tokenId].SlotWidth / 2f;
        float totalSlotSpan = _vocab[^1].SlotRight;

        // Convert to grid coordinates at the output row
        int   lastRow  = Config.TotalRows - 1;
        float gridLeft = _simulator.LeftBorder(lastRow);
        float gridSpan = _simulator.GridWidth(lastRow);

        return gridLeft + (slotCentre / totalSlotSpan) * gridSpan;
    }

    /// <summary>
    /// Decays nail stiffness toward baseline each epoch so nails don't permanently freeze.
    /// decayRate=0.02 means nails regress 2% toward baseline per epoch.
    /// </summary>
    public void DecayNailStiffness(float decayRate = 0.02f)
    {
        float baseResistance = Config.DefaultRadius;   // initial resistance = radius
        for (int r = 0; r < _nails.GetLength(0); r++)
        for (int c = 0; c < _nails.GetLength(1); c++)
        {
            float baseDensity = 1f + ((r + c) % 5) * 0.05f;   // matches InitNails pattern
            _nails[r, c].Resistance = _nails[r, c].Resistance * (1f - decayRate) + baseResistance * decayRate;
            _nails[r, c].Density    = _nails[r, c].Density    * (1f - decayRate) + baseDensity    * decayRate;
        }
    }

    private void InitNails(Random rng)
    {
        for (int r = 0; r < _nails.GetLength(0); r++)
        for (int c = 0; c < _nails.GetLength(1); c++)
            _nails[r, c] = new Nail(
                id:         r * 10000 + c,
                radius:     Config.DefaultRadius,
                resistance: Config.DefaultRadius,  // resistance mirrors radius initially
                density:    1f + ((r + c) % 5) * 0.05f
            );
    }
}
