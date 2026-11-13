import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram

SR = 44100
OUT = "."

def music(dur=3.0, sr=SR):
    t = np.arange(int(dur * sr)) / sr
    x = np.zeros_like(t)
    for f0 in (220.0, 277.18, 329.63):
        for k, a in enumerate([1, .5, .3, .18, .1, .06], start=1):
            x += a * np.sin(2 * np.pi * f0 * k * t + k)
    x *= np.minimum(1, t * 8) * np.exp(-0.35 * t)
    return 0.85 * x / np.abs(x).max()

def save_wav(name, x, sr=SR):
    x = np.asarray(x, dtype=float)
    peak = np.max(np.abs(x))
    if peak > 1:
        x = x / peak * 0.98
    wavfile.write(name, sr, (np.clip(x, -1, 1) * 32767).astype(np.int16))

def spectrum_plot(x, name, sr=SR, title="Спектр", max_freq=5000):
    n = min(len(x), 8192)
    segment = x[:n] * np.hanning(n)
    sp = np.abs(np.fft.rfft(segment))
    db = 20 * np.log10(sp / (sp.max() + 1e-20) + 1e-12)
    fr = np.fft.rfftfreq(n, 1 / sr)
    mask = fr <= max_freq
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(fr[mask], db[mask], lw=.8)
    ax.set(xlabel="Частота, Гц", ylabel="Відносний рівень, дБ", title=title, ylim=(-100, 3))
    ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(name, dpi=140); plt.close(fig)

def spectrogram_plot(x, name, n, sr=SR, title="Спектрограма"):
    f, t, S = spectrogram(x, sr, nperseg=n, noverlap=int(n * .9))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    im = ax.pcolormesh(t, f, 10 * np.log10(S + 1e-12), cmap="magma", shading="auto", vmin=-110, vmax=-25)
    ax.set_ylim(0, 2000); ax.set(xlabel="Час, с", ylabel="Частота, Гц", title=title)
    fig.colorbar(im, ax=ax, label="Потужність, дБ")
    fig.tight_layout(); fig.savefig(name, dpi=140); plt.close(fig)

def window_leakage():
    n = 1024
    windows = (("Прямокутне", np.ones(n)), ("Ганна", np.hanning(n)), ("Блекмана", np.blackman(n)))
    rows = []
    for k, case in ((100.0, "Рівно на біні"), (100.5, "Між бінами")):
        f0 = k * SR / n
        t = np.arange(n) / SR
        x = np.sin(2 * np.pi * f0 * t)
        for label, w in windows:
            sp = np.abs(np.fft.rfft(x * w))
            db = 20 * np.log10(sp / (sp.max() + 1e-300) + 1e-15)
            far = np.concatenate([db[:95], db[106:]])
            rows.append((case, label, float(far.max())))
    # Plot centered around the peak for the between-bin case.
    fig, ax = plt.subplots(figsize=(9, 4))
    n = 1024; k = 100.5; t = np.arange(n) / SR; f0 = k * SR / n
    x = np.sin(2 * np.pi * f0 * t); fr = np.fft.rfftfreq(n, 1/SR)
    for label, w in windows:
        sp = np.abs(np.fft.rfft(x*w)); db = 20*np.log10(sp/(sp.max()+1e-300)+1e-15)
        m = (fr > 3500) & (fr < 5200)
        ax.plot(fr[m], db[m], label=label)
    ax.set(xlabel="Частота, Гц", ylabel="дБ від піку", title="Витік спектра: тон між бінами", ylim=(-100, 3))
    ax.grid(alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig("window_leakage.png", dpi=140); plt.close(fig)
    return rows

def time_frequency_signal():
    dur = 1.4; t = np.arange(int(dur * SR)) / SR; x = np.zeros_like(t)
    seg = t < .6
    x[seg] = .5*np.sin(2*np.pi*440*t[seg]) + .5*np.sin(2*np.pi*480*t[seg])
    for c in (.80, .92, 1.04, 1.16):
        i = int(c*SR); x[i:i+40] += np.hanning(40)
    return x

def eq(x, bands, sr=SR):
    sp = np.fft.rfft(x)
    fr = np.fft.rfftfreq(len(x), 1/sr)
    gain = np.ones(len(sp))
    for lo, hi, db in bands:
        gain[(fr >= lo) & (fr < hi)] = 10**(db/20)
    return np.fft.irfft(sp*gain, len(x))

def band_energy(x, lo, hi, sr=SR):
    sp = np.abs(np.fft.rfft(x)); fr = np.fft.rfftfreq(len(x), 1/sr)
    return 10*np.log10((sp[(fr >= lo) & (fr < hi)]**2).sum()+1e-20)

def main():
    # 1. Source sound, spectrum and spectrogram.
    x = music(); save_wav("source.wav", x)
    spectrum_plot(x, "spectrum.png", title="Спектр вихідного сигналу")
    f, t, S = spectrogram(x, SR, nperseg=2048, noverlap=1792)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    im = ax.pcolormesh(t, f, 10*np.log10(S+1e-12), shading="auto", cmap="magma")
    ax.set_ylim(0, 2500); ax.set(xlabel="Час, с", ylabel="Частота, Гц", title="Спектрограма вихідного сигналу")
    fig.colorbar(im, ax=ax, label="Потужність, дБ"); fig.tight_layout(); fig.savefig("spectrogram.png", dpi=140); plt.close(fig)

    # 2. Window leakage measurements.
    rows = window_leakage()
    print("Розтікання спектра (найвища стороння пелюстка, дБ):")
    for case, label, value in rows:
        print(f"{case:16s} {label:14s} {value:7.1f}")

    # 3. Time-frequency tradeoff.
    xtf = time_frequency_signal(); save_wav("time_frequency.wav", xtf)
    spectrogram_plot(xtf, "spectrogram_128.png", 128, title="Коротке вікно: 128 відліків")
    spectrogram_plot(xtf, "spectrogram_4096.png", 4096, title="Довге вікно: 4096 відліків")

    # 4. EQ presets and verification.
    presets = {
        "bass": [(0, 250, 12)],
        "voice": [(250, 1000, 4), (1000, 4000, 3)],
        "telephone": [(0, 300, -60), (3000, SR//2 + 1, -60)],
    }
    boosted = eq(x, presets["bass"])
    bands = ((0,250),(250,1000),(1000,4000),(4000,SR//2))
    print("Енергія смуг, до → після підсилення басу:")
    for lo, hi in bands:
        a, b = band_energy(x,lo,hi), band_energy(boosted,lo,hi)
        print(f"{lo:4d}–{hi:4d} Гц: {a:7.2f} → {b:7.2f} дБ  (зміна {b-a:+.2f} дБ)")
    spectrum_plot(x, "eq_before.png", title="Спектр до еквалайзера")
    spectrum_plot(boosted, "eq_after.png", title="Спектр після підсилення басу")
    for name, bandspec in presets.items():
        save_wav(f"preset_{name}.wav", eq(x,bandspec))
    print("Створено source.wav, три пресети WAV і всі графіки.")

if __name__ == "__main__":
    main()
