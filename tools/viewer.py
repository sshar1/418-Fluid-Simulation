#!/usr/bin/env python3
"""Live browser viewer for `render <ref|cuda> --stream`. Standard library only.

    ssh ghcXX 'cd .../src && ./render cuda --stream' | python3 tools/viewer.py

Frame format (little-endian, world units), written by src/stream.cpp:
    uint32 magic 'SPH3' | uint32 n | float32 boundsW | float32 boundsH
    | float32 radius | float32 xyzw[4n] | uint8 rgba[4n]
The sim box is centred on the origin; the canvas is sized to its aspect ratio.
Only the newest frame is kept, so a slow browser never stalls the sim.
"""
import struct, sys, threading, webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = 8418
HDR = struct.Struct("<IIfff")
MAGIC = 0x33485053   # "SPH3"
BYTES_PER_PARTICLE = 16 + 4   # float32 xyzw + uint8 rgba
latest, frame_id, done = None, 0, False
cond = threading.Condition()


def reader():
    global latest, frame_id, done
    f = sys.stdin.buffer
    while True:
        hdr = f.read(HDR.size)
        if len(hdr) < HDR.size:
            break
        magic, n = HDR.unpack(hdr)[:2]
        if magic != MAGIC:
            sys.stderr.write("viewer: bad frame header %r (is something else printing to stdout, "
                             "e.g. your remote ~/.bashrc? or is ./render out of date?)\n" % hdr)
            break
        body = f.read(BYTES_PER_PARTICLE * n)
        if len(body) < BYTES_PER_PARTICLE * n:
            break
        with cond:
            latest, frame_id = hdr + body, frame_id + 1
            cond.notify_all()
    with cond:
        done = True
        cond.notify_all()
    sys.stderr.write("viewer: stream ended (Ctrl-C to quit)\n")


PAGE = b"""<!doctype html><meta charset="utf-8"><title>SPH viewer</title>
<style>body{margin:0;height:100vh;display:grid;place-items:center;background:#111;color:#ccc;font:13px monospace}
canvas{background:#000}</style>
<div><canvas id="c" width="800" height="800"></canvas><div id="s">connecting...</div></div>
<script>
const HDR = 20;   // magic, n, boundsW, boundsH, radius
const cv = document.getElementById('c'), ctx = cv.getContext('2d'), s = document.getElementById('s');
let pending = null, frames = 0, last = performance.now();
function draw() {
  requestAnimationFrame(draw);
  if (!pending) return;
  const dv = new DataView(pending), n = dv.getUint32(4, true);
  const bw = dv.getFloat32(8, true), bh = dv.getFloat32(12, true);
  // On-screen (CSS) size: as big as fits the window, at the box's aspect ratio
  const cssW = `min(95vw, ${(90 * bw / bh).toFixed(2)}vh)`;
  if (cv.style.width !== cssW) { cv.style.width = cssW; cv.style.aspectRatio = `${bw} / ${bh}`; }
  // Drawing-buffer size: one canvas pixel per physical screen pixel (x2 on
  // Retina), so nothing is stretched. Re-checked every frame to follow resizes.
  const dpr = window.devicePixelRatio || 1;
  const W = Math.round(cv.clientWidth * dpr), H = Math.round(W * bh / bw), k = W / bw;   // k = px per world unit
  if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
  const r = Math.max(0.5, dv.getFloat32(16, true) * k), p = new Float32Array(pending, HDR, 4 * n);
  const col = new Uint32Array(pending, HDR + 16 * n, n);   // rgba bytes as little-endian uint32
  pending = null;
  ctx.fillStyle = '#000'; ctx.fillRect(0, 0, W, H);
  // One path per distinct colour, so there are only a few fill() calls per frame
  const paths = new Map();
  for (let i = 0; i < n; i++) {
    const x = (p[4*i] + bw / 2) * k, y = (bh / 2 - p[4*i+1]) * k;
    let path = paths.get(col[i]);
    if (!path) paths.set(col[i], path = new Path2D());
    path.moveTo(x + r, y); path.arc(x, y, r, 0, 2 * Math.PI);
  }
  for (const [c, path] of paths) {
    ctx.fillStyle = `rgb(${c & 255},${(c >>> 8) & 255},${(c >>> 16) & 255})`;
    ctx.fill(path);
  }
  frames++;
  const now = performance.now();
  if (now - last >= 1000) { s.textContent = `${n} particles, ${(frames * 1000 / (now - last)).toFixed(1)} FPS`; frames = 0; last = now; }
}
requestAnimationFrame(draw);
(async () => {
  const rd = (await fetch('/stream')).body.getReader();
  let acc = new Uint8Array(0);
  for (;;) {
    const { value, done } = await rd.read();
    if (done) break;
    const t = new Uint8Array(acc.length + value.length); t.set(acc); t.set(value, acc.length); acc = t;
    while (acc.length >= HDR) {
      const size = HDR + 20 * new DataView(acc.buffer, acc.byteOffset).getUint32(4, true);
      if (acc.length < size) break;
      pending = acc.slice(0, size).buffer; acc = acc.slice(size);
    }
  }
  s.textContent = 'stream ended';
})().catch(e => s.textContent = 'error: ' + e);
</script>"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"   # body streams until the connection closes

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path not in ("/", "/stream"):
            return self.send_error(404)
        self.send_response(200)
        self.send_header("Content-Type", "text/html" if self.path == "/" else "application/octet-stream")
        self.end_headers()
        if self.path == "/":
            return self.wfile.write(PAGE)
        seen = 0   # frame_id starts at 0, so this waits for the first real frame
        try:
            while True:
                with cond:
                    cond.wait_for(lambda: frame_id != seen or done)
                    if frame_id == seen:
                        return
                    seen, data = frame_id, latest
                self.wfile.write(data)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


if __name__ == "__main__":
    if sys.stdin.isatty():
        sys.exit(__doc__)
    threading.Thread(target=reader, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    srv.daemon_threads = True
    sys.stderr.write("viewer: http://localhost:%d/\n" % PORT)
    webbrowser.open("http://localhost:%d/" % PORT)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
