// Record the Chainmind demo video with headless Chrome (no extra npm deps).
//
//   python3 -m http.server 8766 --directory .        (from the repo root)
//   node demo/record.mjs [outDir]                    (writes frames + chainmind-demo.mp4)
//
// Scenes: intro card, the real testnet transaction, a directed tour of the
// live landing page, outro card. Needs Chrome and ffmpeg (libx264).

import { spawn, execFileSync } from "node:child_process";
import { mkdirSync, writeFileSync, rmSync } from "node:fs";
import { join, resolve } from "node:path";

const CHROME = process.env.CHROME ?? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const BASE = process.env.BASE ?? "http://localhost:8766";
const OUT = resolve(process.argv[2] ?? "demo/out");
const W = 1920, H = 1080, PORT = 9334;
// Chrome can't screencast 1080p fast enough for smooth motion, so the page's
// clock runs at SLOW of real time (timers, rAF, performance.now, CSS
// animations) and frames are retimed back to page time when encoding.
const SLOW = Number(process.env.SLOW ?? 0.25);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const TIME_WARP = `(() => {
  const S = ${SLOW};
  const now0 = performance.now.bind(performance), date0 = Date.now, t0 = now0(), d0 = date0();
  const vnow = () => t0 + (now0() - t0) * S;
  performance.now = vnow;
  Date.now = () => d0 + (now0() - t0) * S;
  const raf = window.requestAnimationFrame.bind(window);
  window.requestAnimationFrame = (cb) => raf(() => cb(vnow()));
  const st = window.setTimeout.bind(window), si = window.setInterval.bind(window);
  window.setTimeout = (fn, ms, ...a) => st(fn, (ms || 0) / S, ...a);
  window.setInterval = (fn, ms, ...a) => si(fn, (ms || 0) / S, ...a);
})();`;

rmSync(OUT, { recursive: true, force: true });
mkdirSync(OUT, { recursive: true });

const chrome = spawn(CHROME, [
  "--headless=new", "--disable-gpu", "--hide-scrollbars", "--mute-audio",
  `--user-data-dir=${join(OUT, "profile")}`, `--remote-debugging-port=${PORT}`,
  `--window-size=${W},${H}`, "about:blank",
], { stdio: "ignore" });

let target;
for (let i = 0; i < 60 && !target; i++) {
  try { target = (await (await fetch(`http://127.0.0.1:${PORT}/json`)).json()).find((t) => t.type === "page"); } catch {}
  if (!target) await sleep(250);
}
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((r) => ws.addEventListener("open", r));
let nextId = 0;
const pending = new Map();
const listeners = new Set();
ws.addEventListener("message", (e) => {
  const m = JSON.parse(e.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
  else for (const l of listeners) l(m);
});
const send = (method, params = {}) => new Promise((r) => { const id = ++nextId; pending.set(id, r); ws.send(JSON.stringify({ id, method, params })); });
const evaluate = async (expression) => (await send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true })).result?.result?.value;

await send("Page.enable");
await send("Runtime.enable");
await send("Animation.enable");
await send("Page.addScriptToEvaluateOnNewDocument", { source: TIME_WARP });

const segments = [];
async function record(name, url, play, css = { w: W, h: H }) {
  await send("Emulation.setDeviceMetricsOverride", { width: css.w, height: css.h, deviceScaleFactor: W / css.w, mobile: false });
  await send("Page.navigate", { url });
  await sleep(2500);
  await send("Animation.setPlaybackRate", { playbackRate: SLOW });
  const dir = join(OUT, name);
  mkdirSync(dir, { recursive: true });
  const frames = [];
  const onFrame = (m) => {
    if (m.method !== "Page.screencastFrame") return;
    const file = join(dir, `${String(frames.length).padStart(5, "0")}.jpg`);
    writeFileSync(file, Buffer.from(m.params.data, "base64"));
    frames.push({ file, t: m.params.metadata.timestamp });
    send("Page.screencastFrameAck", { sessionId: m.params.sessionId });
  };
  listeners.add(onFrame);
  await send("Page.startScreencast", { format: "jpeg", quality: 92, maxWidth: W, maxHeight: H, everyNthFrame: 1 });
  const t0 = Date.now() / 1000;
  await play();
  const t1 = Date.now() / 1000;
  await send("Page.stopScreencast");
  listeners.delete(onFrame);
  segments.push({ name, frames, t0, t1 });
  console.log(`${name}: ${frames.length} frames, ${(t1 - t0).toFixed(1)} s`);
}

const waitDone = async (hold) => {
  for (let i = 0; i < 1500 && !(await evaluate("!!window.__done")); i++) await sleep(100);
  await sleep(hold / SLOW);
};

// The landing-page tour: smooth scrolls, captions and real interaction.
const DIRECTOR = `(async () => {
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const $ = (s) => document.querySelector(s);
  const cap = document.createElement("div");
  cap.style.cssText = "position:fixed;left:50%;bottom:56px;transform:translate(-50%,20px);z-index:99;padding:16px 28px;border-radius:999px;background:rgba(9,10,16,.86);backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);box-shadow:inset 0 0 0 1px rgba(255,255,255,.1),0 20px 40px -20px rgba(0,0,0,.6);color:#E9ECF5;font:500 26px Geist,system-ui,sans-serif;letter-spacing:-.01em;opacity:0;transition:opacity .5s,transform .6s cubic-bezier(.2,.7,.1,1);white-space:nowrap;display:flex;gap:14px;align-items:center";
  cap.innerHTML = '<i style="width:10px;height:10px;border-radius:50%;background:#2440FF;box-shadow:0 0 12px #2440FF"></i><span></span>';
  document.body.append(cap);
  const say = async (t) => { cap.style.opacity = 0; cap.style.transform = "translate(-50%,20px)"; await sleep(t ? 280 : 0); if (!t) return; cap.lastChild.textContent = t; cap.style.opacity = 1; cap.style.transform = "translate(-50%,0)"; };
  const ease = (x) => x < .5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2;
  const go = (y, ms) => new Promise((done) => {
    const from = scrollY, t0 = performance.now();
    y = Math.max(0, Math.min(y, document.documentElement.scrollHeight - innerHeight));
    const f = (now) => { const p = Math.min(1, (now - t0) / ms); scrollTo({ top: from + (y - from) * ease(p), behavior: "instant" }); p < 1 ? requestAnimationFrame(f) : done(); };
    requestAnimationFrame(f);
  });
  const top = (sel, off = 0) => $(sel).getBoundingClientRect().top + scrollY + off;
  const type = async (el, text) => { el.focus(); el.value = ""; for (const ch of text) { el.value += ch; el.dispatchEvent(new Event("input")); await sleep(70); } };

  await sleep(600);
  await say("A real 260,032-parameter language model, running live in this page");
  await sleep(6200);
  await say("Today, AI on a blockchain means trusting an oracle, an API or a server");
  await go(top("#clip", -260), 2600);
  await go(top("#manifesto-text", -140), 1800);
  await go(top("#manifesto-text", 260), 3200);
  await say("One transaction: every validator runs the model");
  const steps = [...document.querySelectorAll(".step")];
  await go(top(".tx-head", -120), 1600);
  await sleep(900);
  for (const s of steps) { await go(top(".step:nth-child(" + (steps.indexOf(s) + 1) + ")", -innerHeight * 0.3), 1300); await sleep(850); }
  await say("194.4 Tgas on NEAR testnet for a 48-token story");
  await sleep(1600);
  await say("Same tokens on NEAR's VM, a laptop and your browser");
  await go(top("#reels", -300), 2000);
  await sleep(4200);
  await say("A building block for games, NFTs, DeFi and agent safety");
  await go(top(".numbers", -260), 1400);
  await sleep(900);
  await go(top(".bento", -140), 1600);
  await sleep(2600);
  await say("Try it: the sealed model, in your browser");
  await go(top(".console", -150), 1800);
  await sleep(500);
  await type($("#d-prompt"), "The dragon was");
  $("#d-seed").value = "7";
  await sleep(400);
  $("#d-go").click();
  await sleep(4200);
  await say("Call it from the CLI, or from your own contract");
  await go(top(".term", -320), 1700);
  await sleep(600);
  document.querySelectorAll(".tab")[1].click();
  await sleep(2300);
  document.querySelectorAll(".tab")[4].click();
  await sleep(2600);
  await say("Measured: up to 70 tokens per transaction, stopping safely before the limit");
  await go(top("#bars", -300), 1800);
  await sleep(2400);
  await go(top("#road", -340), 1600);
  await sleep(2200);
  await say("");
  await go(top(".finale", 0), 2000);
  await sleep(3600);
})()`;

const playScene = (hold) => async () => { await sleep(300); await evaluate("window.__play()"); await waitDone(hold); };
await record("intro", `${BASE}/demo/scenes.html?scene=intro`, playScene(3200));
await record("terminal", `${BASE}/demo/scenes.html?scene=terminal`, playScene(3000));
await record("site", `${BASE}/site/index.html`, async () => {
  await sleep(1200 / SLOW);
  await evaluate(DIRECTOR);
}, { w: 1600, h: 900 });
await record("outro", `${BASE}/demo/scenes.html?scene=outro`, playScene(3800));

ws.close();
chrome.kill();

// Build a concat list with real frame timings, then encode a constant 30 fps MP4.
let list = "ffconcat version 1.0\n";
for (const seg of segments) {
  const f = seg.frames;
  for (let i = 0; i < f.length; i++) {
    const end = i + 1 < f.length ? f[i + 1].t : Math.max(seg.t1, f[i].t + 0.04);
    list += `file '${f[i].file}'\nduration ${Math.max(0.001, (end - f[i].t) * SLOW).toFixed(4)}\n`;
  }
}
const last = segments.at(-1).frames.at(-1).file;
list += `file '${last}'\n`;
writeFileSync(join(OUT, "frames.ffconcat"), list);

const video = join(OUT, "chainmind-demo.mp4");
execFileSync("ffmpeg", [
  "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", join(OUT, "frames.ffconcat"),
  "-vf", `scale=${W}:${H}:flags=lanczos,fps=30,format=yuv420p,fade=t=in:st=0:d=0.6`,
  "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-movflags", "+faststart", video,
], { stdio: "inherit" });
console.log(`wrote ${video}`);
