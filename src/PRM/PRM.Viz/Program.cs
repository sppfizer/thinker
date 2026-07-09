using System.Diagnostics;
using System.Text;
using System.Text.Json;
using PRM.Core.Engine;
using PRM.Core.Models;
using PRM.Viz;

Console.OutputEncoding = Encoding.UTF8;

// ── 1. Load corpus + vocabulary ───────────────────────────────────────────────
string? corpusFile = null;
string[] vizArgs   = args;
if (args.Length >= 2 && args[0] == "--corpus") { corpusFile = args[1]; vizArgs = args[2..]; }

var corpus = LoadCorpus(corpusFile);

var builder = new VocabularyBuilder();
builder.Feed(VocabularyBuilder.Tokenise(corpus));
VocabToken[] vocab = builder.Build();

Console.WriteLine($"Vocabulary: {vocab.Length} tokens");

// ── 2. Load active config + grid ──────────────────────────────────────────────
var config = LoadConfig();
var grid   = new DiamondGrid(config, vocab, new Random(42));

if (File.Exists("prm_nails.bin"))
{
    grid.LoadNails("prm_nails.bin");
    Console.WriteLine("Nails loaded from prm_nails.bin");
}
else
{
    Console.WriteLine("No prm_nails.bin found — nails are untrained (random offsets).");
}

// ── 3. Decide which inputs to visualise ───────────────────────────────────────
// vizArgs can be word tokens (e.g. "the cat sat") or empty → pick a random training sample.
int[] inputIds;
if (vizArgs.Length >= 2)
{
    var wTokens = VocabularyBuilder.Tokenise(string.Join(" ", vizArgs));
    inputIds = wTokens
        .Select(t => vocab.FirstOrDefault(v => v.Text.Trim().Equals(t.Trim(), StringComparison.OrdinalIgnoreCase))?.Id ?? -1)
        .Where(id => id >= 0)
        .Take(3)
        .ToArray();

    if (inputIds.Length < 2)
    {
        Console.WriteLine($"Warning: only {inputIds.Length} of your words found in vocabulary. Using random sample.");
        inputIds = GetRandomSample(vocab, corpus);
    }
}
else
{
    inputIds = GetRandomSample(vocab, corpus);
}

string[] inputLabels = inputIds.Select(id => vocab[id].Text.Trim()).ToArray();
Console.WriteLine($"Visualising: [{string.Join(", ", inputLabels)}]");

// ── 4. Start WebSocket server ─────────────────────────────────────────────────
int port = 5050;
{ var pi = Array.IndexOf(vizArgs, "--port"); if (pi >= 0 && pi + 1 < vizArgs.Length) int.TryParse(vizArgs[pi + 1], out port); }
bool noBrowser = vizArgs.Contains("--no-browser");

await using var server = new VizServer(vocab, port);
Console.WriteLine($"Visualizer running at http://localhost:{server.Port}/");

if (!noBrowser)
{
    Console.WriteLine("Opening browser…");
    try { Process.Start(new ProcessStartInfo($"http://localhost:{server.Port}/") { UseShellExecute = true }); }
    catch { Console.WriteLine("Could not open browser automatically — please open the URL manually."); }
}

using var exitCts = new CancellationTokenSource();
Console.CancelKeyPress += (_, e) => { e.Cancel = true; exitCts.Cancel(); };

// ── 5. Serve loop — accept a client, stream simulation, repeat ────────────────
// Grid config sent once per client connection
var gridInfo = new DiamondGridInfo(
    grid.Config.TotalRows, grid.Config.WideningRows,
    grid.Config.EntryWidth, grid.Config.MaxWidth);

Console.WriteLine("Waiting for browser to connect (Ctrl+C to quit)…");

while (!exitCts.IsCancellationRequested)
{
    // Accept next client (handles favicon, 204 for junk, waits for WS upgrade)
    using var connCts = CancellationTokenSource.CreateLinkedTokenSource(exitCts.Token);
    connCts.CancelAfter(TimeSpan.FromSeconds(noBrowser ? 30 : 300));
    try
    {
        await server.WaitForClientAsync(connCts.Token);
    }
    catch (OperationCanceledException) { break; }

    // Send config + initial simulation
    await server.SendConfigAsync(gridInfo);
    await Task.Delay(300);
    Console.WriteLine("Streaming simulation frames…");
    await VisualiseSingle(server, grid, vocab, inputIds);

    // If no-browser (test mode): exit after first client
    if (noBrowser) break;

    // Interactive REPL for the connected client
    Console.WriteLine("\n  → Enter new words to visualise (e.g. 'the cat sat'), or Enter to replay");
    Console.WriteLine("  → Press Ctrl+C to quit\n");

    while (!exitCts.IsCancellationRequested)
    {
        Console.Write("tokens> ");
        string? line;
        try { line = await Task.Run(() => Console.ReadLine(), exitCts.Token); }
        catch (OperationCanceledException) { goto nextClient; }

        if (line == null) break;  // stdin closed → wait for next browser connection
        line = line.Trim();
        if (line == "") { await VisualiseSingle(server, grid, vocab, inputIds); continue; }

        var newWords = VocabularyBuilder.Tokenise(line)
            .Select(t => vocab.FirstOrDefault(v => v.Text.Trim().Equals(t.Trim(), StringComparison.OrdinalIgnoreCase))?.Id ?? -1)
            .Where(id => id >= 0)
            .Take(3)
            .ToArray();

        if (newWords.Length < 2) { Console.WriteLine("  Need at least 2 known words."); continue; }
        inputIds = newWords;
        await VisualiseSingle(server, grid, vocab, inputIds);
    }
    nextClient:;
}

Console.WriteLine("Bye!");

// ═════════════════════════════════════════════════════════════════════════════

static async Task VisualiseSingle(VizServer server, DiamondGrid grid, VocabToken[] vocab, int[] inputIds)
{
    string[] labels = inputIds.Select(id => vocab[id].Text.Trim()).ToArray();

    await server.SendClearAsync(labels);
    await Task.Delay(100);

    // Run the simulation WITH trace recording
    var trace = grid.SimulateWithTrace(inputIds);

    // Stream each row to the browser with a small delay for animation effect
    for (int r = 0; r < trace.RowFrames.Length; r++)
    {
        var balls  = trace.RowFrames[r];
        var nailXs    = r < trace.NailBaseXs.Length      ? trace.NailBaseXs[r]      : [];
        var offXs     = r < trace.NailOffXs.Length       ? trace.NailOffXs[r]       : [];
        var nailRadii = r < trace.NailRadii.Length       ? trace.NailRadii[r]       : [];
        var nailRes   = r < trace.NailResistances.Length ? trace.NailResistances[r] : [];
        int rowNum = r < trace.TotalRows ? r : trace.TotalRows;

        await server.SendFrameAsync(balls, nailXs, offXs, nailRadii, nailRes, rowNum);
        await Task.Delay(30); // 30ms per row → smooth animation; browser can replay at any speed
    }

    // Compute prediction from final frame
    var (predicted, _) = grid.Predict(inputIds);
    string predLabel   = predicted >= 0 && predicted < vocab.Length ? vocab[predicted].Text.Trim() : "?";
    await server.SendResultAsync(predLabel, target: null, correct: false);

    Console.WriteLine($"Prediction: [{string.Join(" ", labels)}] → \"{predLabel}\"");
}

static int[] GetRandomSample(VocabToken[] vocab, string corpus)
{
    var tokenIds = VocabularyBuilder.Tokenise(corpus)
        .Select(t => vocab.FirstOrDefault(v => v.Text.Trim() == t.Trim())?.Id ?? -1)
        .Where(id => id >= 0)
        .ToArray();
    if (tokenIds.Length < 3) return tokenIds.Take(3).ToArray();
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
        Path.Combine(Directory.GetCurrentDirectory(), "data", "simple_corpus.txt"),
    };
    foreach (var p in candidates) { var full = Path.GetFullPath(p); if (File.Exists(full)) return File.ReadAllText(full); }
    return "the cat sat on the mat the dog ran to the park";
}

static DiamondConfig LoadConfig()
{
    if (File.Exists("prm_config.json"))
    {
        var json = File.ReadAllText("prm_config.json");
        var cfg  = JsonSerializer.Deserialize<DiamondConfig>(json);
        if (cfg is not null) return cfg;
    }
    return new DiamondConfig { RoleName = "Analyst", WideningRows = 8, NarrowingRows = 8, MaxWidth = 50f, EntryWidth = 20f };
}
