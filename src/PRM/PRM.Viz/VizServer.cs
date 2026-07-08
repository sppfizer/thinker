using System.Net;
using System.Net.WebSockets;
using System.Text;
using System.Text.Json;
using PRM.Core.Models;

namespace PRM.Viz;

/// <summary>
/// Embedded HTTP + WebSocket server that streams PRM simulation frames to the browser.
/// Serves the single-page visualizer at http://localhost:5050/
/// and accepts WebSocket connections at ws://localhost:5050/ws
/// </summary>
public sealed class VizServer : IAsyncDisposable
{
    private readonly HttpListener   _http = new();
    private readonly int            _port;
    private readonly VocabToken[]   _vocab;
    private readonly string         _html;

    // Active WebSocket client (only one at a time needed)
    private HttpListenerWebSocketContext? _wsCtx;

    public VizServer(VocabToken[] vocab, int port = 5050)
    {
        _port  = port;
        _vocab = vocab;
        _html  = HtmlPage.Build(port);

        _http.Prefixes.Add($"http://localhost:{port}/");
        _http.Start();
    }

    /// <summary>
    /// Loop accepting HTTP requests until the browser's WebSocket upgrade arrives.
    /// Serves the HTML page on the first GET /, returns 204 for everything else
    /// (favicon, robots.txt, etc.) so those requests don't consume the WS slot.
    /// </summary>
    public async Task WaitForClientAsync(CancellationToken ct)
    {
        bool htmlServed = false;
        while (_wsCtx == null)
        {
            var ctx = await _http.GetContextAsync().WaitAsync(ct);

            if (ctx.Request.IsWebSocketRequest)
            {
                _wsCtx = await ctx.AcceptWebSocketAsync(subProtocol: null);
                Console.WriteLine("[viz] WebSocket connected.");
            }
            else if (!htmlServed &&
                     (ctx.Request.Url?.AbsolutePath is "/" or "" or null))
            {
                var bytes = Encoding.UTF8.GetBytes(_html);
                ctx.Response.ContentType     = "text/html; charset=utf-8";
                ctx.Response.ContentLength64 = bytes.Length;
                await ctx.Response.OutputStream.WriteAsync(bytes, ct);
                ctx.Response.OutputStream.Close();
                htmlServed = true;
            }
            else
            {
                // favicon.ico, /ws GET-before-upgrade, robots.txt, etc. — discard gracefully
                ctx.Response.StatusCode = 204;
                ctx.Response.Close();
            }
        }
    }

    /// <summary>Send a JSON message to the connected browser client.</summary>
    public async Task SendAsync(object msg, CancellationToken ct = default)
    {
        if (_wsCtx?.WebSocket.State != WebSocketState.Open) return;
        var json  = JsonSerializer.Serialize(msg);
        var bytes = Encoding.UTF8.GetBytes(json);
        await _wsCtx.WebSocket.SendAsync(bytes, WebSocketMessageType.Text, true, ct);
    }

    /// <summary>
    /// Send the vocab + grid config so the browser can draw slots and scale the grid.
    /// </summary>
    public Task SendConfigAsync(DiamondGridInfo info, CancellationToken ct = default) =>
        SendAsync(new
        {
            type        = "config",
            totalRows   = info.TotalRows,
            wideningRows= info.WideningRows,
            entryWidth  = info.EntryWidth,
            maxWidth    = info.MaxWidth,
            vocab       = _vocab.Select(v => new { v.Id, v.Text, v.Mass, v.SlotLeft, v.SlotRight, v.SlotWidth }).ToArray()
        }, ct);

    /// <summary>Signal the browser that a new prediction sequence is starting.</summary>
    public Task SendClearAsync(string[] inputLabels, CancellationToken ct = default) =>
        SendAsync(new { type = "clear", tokens = inputLabels }, ct);

    /// <summary>Send one row of ball positions + nail data to the browser.</summary>
    public Task SendFrameAsync(PRM.Core.Models.BallFrame[] balls, float[] nailXs, float[] offXs,
                               int row, CancellationToken ct = default) =>
        SendAsync(new
        {
            type  = "frame",
            row,
            balls = balls.Select(b => new { b.TokenId, b.Position, b.Mass, b.Velocity }).ToArray(),
            nails = nailXs.Zip(offXs, (x, ox) => new { x, ox }).ToArray()
        }, ct);

    /// <summary>Send the final prediction result.</summary>
    public Task SendResultAsync(string predicted, string? target, bool correct, CancellationToken ct = default) =>
        SendAsync(new { type = "result", predicted, target, correct }, ct);

    public async ValueTask DisposeAsync()
    {
        if (_wsCtx?.WebSocket.State == WebSocketState.Open)
            await _wsCtx.WebSocket.CloseAsync(WebSocketCloseStatus.NormalClosure, "bye", default);
        _http.Stop();
    }
}

/// <summary>Minimal grid geometry info needed by the browser.</summary>
public record DiamondGridInfo(int TotalRows, int WideningRows, float EntryWidth, float MaxWidth);
