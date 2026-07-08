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
        Config = config;
        _vocab = vocab;
        _magnet = new MagnetField(config);

        // Allocate nail grid — rows × max_columns
        int maxCols = (int)(config.MaxWidth / config.NailSpacing) + 2;
        _nails = new Nail[config.TotalRows, maxCols];

        rng ??= new Random(42);
        InitNails(rng);

        _simulator = new BallSimulator(config, _nails, _magnet);
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
        return (pred, pred == targetTokenId, conf);
    }

    // ── Checkpointing ─────────────────────────────────────────────────────────

    public void SaveNails(string path)
    {
        using var bw = new BinaryWriter(File.Open(path, FileMode.Create));
        bw.Write(_nails.GetLength(0));
        bw.Write(_nails.GetLength(1));
        for (int r = 0; r < _nails.GetLength(0); r++)
        for (int c = 0; c < _nails.GetLength(1); c++)
        {
            bw.Write(_nails[r, c].Tilt);
            bw.Write(_nails[r, c].Diameter);
        }
    }

    public void LoadNails(string path)
    {
        using var br = new BinaryReader(File.Open(path, FileMode.Open));
        int rows = br.ReadInt32(), cols = br.ReadInt32();
        for (int r = 0; r < rows && r < _nails.GetLength(0); r++)
        for (int c = 0; c < cols && c < _nails.GetLength(1); c++)
        {
            _nails[r, c].Tilt     = br.ReadSingle();
            _nails[r, c].Diameter = br.ReadSingle();
        }
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
            balls.Add(new Ball(id, x, _vocab[id].Mass));
        }

        // Prediction slot ball (near-zero mass)
        balls.Add(new Ball(-1, gridCentre, 0.001f));
        return balls;
    }

    private (int tokenId, float confidence) Score(List<Ball> survivors, List<Ball> all)
    {
        float totalInput = all.Sum(b => b.Mass);
        var scores = new float[_vocab.Length];

        foreach (var ball in survivors)
        {
            for (int t = 0; t < _vocab.Length; t++)
            {
                if (ball.Position >= _vocab[t].SlotLeft && ball.Position < _vocab[t].SlotRight)
                {
                    scores[t] += ball.Mass;
                    break;
                }
            }
        }

        int winner   = Array.IndexOf(scores, scores.Max());
        float retMass = survivors.Sum(b => b.Mass);
        float confidence = totalInput > 0 ? retMass / totalInput : 0f;
        return (winner, confidence);
    }

    private float SlotCentre(int tokenId)
        => tokenId >= 0 && tokenId < _vocab.Length
            ? _vocab[tokenId].SlotLeft + _vocab[tokenId].SlotWidth / 2f
            : Config.MaxWidth / 2f;

    private void InitNails(Random rng)
    {
        for (int r = 0; r < _nails.GetLength(0); r++)
        for (int c = 0; c < _nails.GetLength(1); c++)
            _nails[r, c] = new Nail(
                tilt:     (float)(rng.NextDouble() * 0.2 - 0.1),   // small random init
                diameter: Config.DefaultDiameter
            );
    }
}
