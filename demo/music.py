"""Original soundtrack for the Chainmind demo, synthesized from scratch.

    python3 demo/music.py demo/out/frames.ffconcat 101.97 demo/out/music.wav

Reads the scene boundaries from the recorder's frame list so the hits land on
the cuts: a pad intro, ticking during the testnet transaction, an impact when
the landing page appears, then an arpeggio and a beat that builds, a
breakdown, and a resolving chord for the outro. 96 BPM, A minor.
"""

import re
import sys
import wave

import numpy as np
from scipy.signal import butter, sosfilt

SR = 44100
BPM = 96
BEAT = 60 / BPM
BAR = 4 * BEAT
SIX = BEAT / 4
rng = np.random.default_rng(7)


def midi(n):
    return 440.0 * 2 ** ((n - 69) / 12)


def sections(ffconcat):
    """Start time of each recorded scene, in video seconds."""
    durs = {}
    order = []
    for m in re.finditer(r"/out/(\w+)/\d+\.jpg'\nduration ([\d.]+)", open(ffconcat).read()):
        name = m.group(1)
        if name not in durs:
            durs[name] = 0.0
            order.append(name)
        durs[name] += float(m.group(2))
    t, starts = 0.0, {}
    for name in order:
        starts[name] = t
        t += durs[name]
    return starts


def env(n, attack, release, total):
    """Linear attack, hold, linear release over n samples."""
    e = np.ones(n)
    a, r = int(attack * SR), int(release * SR)
    if a:
        e[: min(a, n)] = np.linspace(0, 1, min(a, n))
    if r and n > r:
        e[-r:] *= np.linspace(1, 0, r)
    return e * total


def lowpass(x, hz, order=2):
    return sosfilt(butter(order, hz, "low", fs=SR, output="sos"), x)


def highpass(x, hz, order=2):
    return sosfilt(butter(order, hz, "high", fs=SR, output="sos"), x)


def bandpass(x, lo, hi):
    return sosfilt(butter(2, [lo, hi], "band", fs=SR, output="sos"), x)


class Track:
    def __init__(self, seconds):
        self.n = int(seconds * SR)
        self.L = np.zeros(self.n)
        self.R = np.zeros(self.n)

    def add(self, t, sig, gain=1.0, pan=0.0):
        i = int(t * SR)
        if i >= self.n or i + len(sig) <= 0:
            return
        j = min(self.n, i + len(sig))
        s = sig[: j - i] * gain
        self.L[i:j] += s * np.sqrt(0.5 * (1 - pan))
        self.R[i:j] += s * np.sqrt(0.5 * (1 + pan))


# Am9 · Fmaj7(9) · Cmaj7(9) · G6 — two bars each, 20 s loop.
PROG = [
    (45, [57, 60, 64, 67, 71]),
    (41, [53, 57, 60, 64, 67]),
    (48, [60, 64, 67, 71, 74]),
    (43, [55, 59, 62, 64, 69]),
]


def chord_at(t):
    return PROG[int(t // (2 * BAR)) % len(PROG)]


def pad_note(freq, dur, bright):
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = np.zeros(n)
    for det in (-0.12, 0.0, 0.11):  # detuned saw stack, a few harmonics
        f = freq * 2 ** (det / 12)
        for h in range(1, 7):
            x += np.sin(2 * np.pi * f * h * t + rng.uniform(0, 6.28)) / h
    x = lowpass(x, bright)
    return x * env(n, 1.2, 1.4, 1.0) / 6


def pluck(freq, dur=0.5):
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = np.sin(2 * np.pi * freq * t) + 0.25 * np.sin(2 * np.pi * freq * 2 * t) + 0.08 * np.sin(2 * np.pi * freq * 3 * t)
    return x * np.exp(-t / 0.16) * np.minimum(1, t / 0.004)


def bell(freq, dur=3.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = np.sin(2 * np.pi * freq * t) + 0.4 * np.sin(2 * np.pi * freq * 2.76 * t) * np.exp(-t / 0.4)
    return x * np.exp(-t / 1.1) * np.minimum(1, t / 0.003)


def kick():
    n = int(0.45 * SR)
    t = np.arange(n) / SR
    f = 45 + 110 * np.exp(-t / 0.045)
    ph = 2 * np.pi * np.cumsum(f) / SR
    click = rng.standard_normal(n) * np.exp(-t / 0.004) * 0.3
    return (np.sin(ph) * np.exp(-t / 0.28) + click) * 0.95


def hat(open_=False):
    n = int((0.22 if open_ else 0.06) * SR)
    t = np.arange(n) / SR
    x = highpass(rng.standard_normal(n), 7000, 4)
    return x * np.exp(-t / (0.07 if open_ else 0.018))


def clap():
    n = int(0.3 * SR)
    t = np.arange(n) / SR
    x = bandpass(rng.standard_normal(n), 900, 3500)
    e = np.exp(-t / 0.09)
    for d in (0.0, 0.011, 0.022):  # the three little hand slaps
        k = int(d * SR)
        e[k:] += 0.5 * np.exp(-t[: n - k] / 0.006)
    return x * e * 0.6


def tick():
    n = int(0.03 * SR)
    t = np.arange(n) / SR
    return np.sin(2 * np.pi * 2400 * t) * np.exp(-t / 0.006)


def riser(dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = rng.standard_normal(n)
    # Sweep a band upward by filtering chunks with rising cutoff.
    out = np.zeros(n)
    chunks = 40
    for c in range(chunks):
        a, b = c * n // chunks, (c + 1) * n // chunks
        hz = 300 * (40 ** (c / chunks))
        out[a:b] = bandpass(x[a:b], hz, min(hz * 2.2, 18000))
    return out * (t / dur) ** 2


def impact():
    n = int(2.5 * SR)
    t = np.arange(n) / SR
    boom = np.sin(2 * np.pi * (38 + 60 * np.exp(-t / 0.08)) * t) * np.exp(-t / 0.6)
    air = lowpass(rng.standard_normal(n), 6000) * np.exp(-t / 0.5) * 0.35
    return boom + air


def bass_note(freq, dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(2 * np.pi * freq * 2 * t)
    return lowpass(x, 400) * np.exp(-t / (dur * 0.9)) * np.minimum(1, t / 0.005)


def build(ffconcat, seconds, out):
    S = sections(ffconcat)
    t_term, t_site, t_out = S.get("terminal", 4.3), S.get("site", 15.3), S.get("outro", seconds - 9)
    end = seconds
    t_kick = t_site + 2 * BAR          # beat arrives after the hero settles
    t_full = t_kick + 4 * BAR          # hats and claps join
    t_break = t_out - 4 * BAR          # breakdown before the outro
    music, drums = Track(end), Track(end)

    # Pad: whole piece, darker in the intro, opening up when the site appears.
    t = 0.0
    while t < end:
        root, notes = chord_at(t)
        bright = 900 if t < t_site else 2400 if t < t_break else 1500
        for k, nn in enumerate(notes):
            music.add(t, pad_note(midi(nn), 2 * BAR + 1.4, bright), 0.22, pan=(k - 2) * 0.25)
        t += 2 * BAR

    # Intro and outro bells.
    for i, nn in enumerate([76, 79, 83, 81]):
        music.add(0.6 + i * BEAT * 1.5, bell(midi(nn)), 0.12, pan=0.3 * (-1) ** i)
    for i, nn in enumerate([69, 72, 76, 79, 83]):
        music.add(t_out + 0.2 + i * BEAT, bell(midi(nn)), 0.13, pan=0.25 * (-1) ** i)

    # Terminal: clock ticks and a quiet bass pulse while the transaction runs.
    k = 0
    tt = t_term
    while tt < t_site - 0.4:
        music.add(tt, tick(), 0.07 if k % 4 else 0.12, pan=0.4 if k % 2 else -0.4)
        tt += SIX
        k += 1
    music.add(t_site - 2.6, riser(2.6), 0.22)
    music.add(t_out - 2.2, riser(2.2), 0.16)
    music.add(t_site, impact(), 0.55)

    # Bass: eighth-note pulse from the terminal on, louder once the site appears.
    tt = t_term
    while tt < t_out:
        root, _ = chord_at(tt)
        g = 0.18 if tt < t_site else 0.34 if tt < t_break else 0.2
        music.add(tt, bass_note(midi(root), BEAT / 2), g)
        tt += BEAT / 2
    music.add(t_out, bass_note(midi(45), 6.0), 0.4)

    # Arpeggio: 16ths over the chord tones, with a dotted-eighth echo.
    arp = Track(end)
    tt, k = t_site, 0
    pattern = [0, 2, 4, 1, 3, 4, 2, 1]
    while tt < t_out:
        _, notes = chord_at(tt)
        nn = notes[pattern[k % 8]] + 12
        vel = 0.16 if k % 4 == 0 else 0.1
        if tt >= t_break:
            vel *= 0.7
        arp.add(tt, pluck(midi(nn)), vel, pan=0.35 * np.sin(k * 0.7))
        tt += SIX
        k += 1
    d = int(BEAT * 0.75 * SR)
    for fb in (0.38, 0.16):
        arp.L[d:] += arp.R[:-d] * fb
        arp.R[d:] += arp.L[:-d] * fb
        d *= 2
    music.L += arp.L
    music.R += arp.R

    # Drums.
    tt, beat = t_kick, 0
    while tt < t_break:
        drums.add(tt, kick(), 0.9)
        if tt >= t_full:
            drums.add(tt + BEAT / 2, hat(open_=(beat % 4 == 3)), 0.12, pan=0.2)
            drums.add(tt + BEAT / 4, hat(), 0.05, pan=-0.2)
            drums.add(tt + 3 * BEAT / 4, hat(), 0.05, pan=-0.2)
            if beat % 2 == 1:
                drums.add(tt, clap(), 0.4, pan=0.05)
        tt += BEAT
        beat += 1

    # Sidechain the music to the kick so the groove breathes.
    duck = np.ones(music.n)
    tt = t_kick
    dn = int(0.32 * SR)
    shape = 1 - 0.45 * np.exp(-np.arange(dn) / (0.09 * SR))
    while tt < t_break:
        i = int(tt * SR)
        j = min(music.n, i + dn)
        duck[i:j] = np.minimum(duck[i:j], shape[: j - i])
        tt += BEAT
    L = music.L * duck + drums.L
    R = music.R * duck + drums.R

    # Simple room: a few early reflections, low-passed.
    for delay, g in ((0.031, 0.18), (0.047, 0.14), (0.071, 0.1), (0.113, 0.07)):
        k = int(delay * SR)
        L[k:] += lowpass(R[:-k], 5000, 1) * g
        R[k:] += lowpass(L[:-k], 5000, 1) * g

    # Master: high-pass rumble, fades, gentle saturation, normalize to -1 dBFS.
    L, R = highpass(L, 28), highpass(R, 28)
    fade = np.ones(len(L))
    fi, fo = int(1.2 * SR), int(4.0 * SR)
    fade[:fi] = np.linspace(0, 1, fi)
    fade[-fo:] = np.linspace(1, 0, fo) ** 1.5
    L, R = L * fade, R * fade
    peak = max(np.abs(L).max(), np.abs(R).max())
    L, R = np.tanh(1.4 * L / peak) / np.tanh(1.4), np.tanh(1.4 * R / peak) / np.tanh(1.4)
    gain = 10 ** (-1 / 20) / max(np.abs(L).max(), np.abs(R).max())
    pcm = (np.stack([L, R], axis=1) * gain * 32767).astype(np.int16)

    with wave.open(out, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    print(f"wrote {out}: {seconds:.2f} s · site at {t_site:.2f} s · beat at {t_kick:.2f} s · outro at {t_out:.2f} s")


if __name__ == "__main__":
    build(sys.argv[1], float(sys.argv[2]), sys.argv[3])
