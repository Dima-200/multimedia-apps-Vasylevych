import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram

SR = 44100          # частота дискретизації за замовчуванням


# ════════════════════════════════════════════════════════════════════
#  ГОТОВЕ. Нічого тут міняти не треба.
# ════════════════════════════════════════════════════════════════════

def save_wav(name, x, sr=SR):
    """Зберігає масив float у діапазоні -1…1 як 16-бітний WAV."""
    x = np.clip(np.asarray(x, dtype=float), -1, 1)
    wavfile.write(name, sr, (x * 32767).astype(np.int16))
    print(f"   збережено: {name}   ({len(x)/sr:.1f} с, {sr} Гц)")


def load_wav(name):
    sr, x = wavfile.read(name)
    return sr, x.astype(float) / 32767


def peak_freq(x, sr=SR):
    """Частота найгучнішої складової — щоб перевірити, що ж ми чуємо."""
    sp = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return np.fft.rfftfreq(len(x), 1 / sr)[sp.argmax()]


def snr_db(clean, noisy):
    """Відношення сигнал/шум у децибелах."""
    err = np.asarray(noisy, float) - np.asarray(clean, float)
    return 10 * np.log10((np.asarray(clean, float) ** 2).mean()
                         / ((err ** 2).mean() + 1e-20))


def plot_spectrum(sigs, name, sr=SR, fmax=6000):
    """sigs — список (сигнал, підпис). Малює спектри в децибелах."""
    fig, ax = plt.subplots(1, len(sigs), figsize=(5.2 * len(sigs), 3.6))
    if len(sigs) == 1:
        ax = [ax]
    for a, (x, lab) in zip(ax, sigs):
        sp = 20 * np.log10(np.abs(np.fft.rfft(x * np.hanning(len(x)))) + 1e-9)
        sp -= sp.max()
        fr = np.fft.rfftfreq(len(x), 1 / sr)
        m = fr < fmax
        a.plot(fr[m], sp[m], lw=0.8)
        a.set_title(lab); a.set_xlabel("частота, Гц"); a.set_ylabel("дБ")
        a.set_ylim(-95, 5); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(name, dpi=130); plt.close(fig)
    print("   збережено:", name)


def plot_spectrogram(x, name, sr=SR):
    f, t, S = spectrogram(x, sr, nperseg=512, noverlap=448)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.pcolormesh(t, f, 10 * np.log10(S + 1e-12), cmap="magma", shading="gouraud")
    ax.axhline(sr / 2, color="cyan", lw=2, ls="--")
    ax.set_xlabel("час, с"); ax.set_ylabel("частота, Гц")
    fig.tight_layout(); fig.savefig(name, dpi=130); plt.close(fig)
    print("   збережено:", name)


# ════════════════════════════════════════════════════════════════════
#  ВАША ЧАСТИНА. Чотири місця.
# ════════════════════════════════════════════════════════════════════

def tone(freq, dur=2.0, amp=0.5, sr=SR):
    """Синусоїда заданої частоти.

    Крок 1: масив моментів часу.  t = np.arange(dur * sr) / sr
            Поділивши номер відліку на частоту дискретизації,
            отримуємо час у секундах.
    Крок 2: сама хвиля.           amp * np.sin(2 * np.pi * freq * t)

    Повернути масив.
    """
    t = np.arange(int(dur * sr)) / sr; return amp * np.sin(2 * np.pi * freq * t)


def sweep(f0, f1, dur=6.0, amp=0.5, sr=SR):
    """Тон, частота якого рівномірно зростає від f0 до f1.

    Тут є пастка. Написати sin(2*pi*f(t)*t) НЕ можна: це дає зовсім
    не ту частоту. Правильно спершу порахувати фазу, а вже тоді синус.

        t  = np.arange(dur * sr) / sr
        ph = 2 * np.pi * (f0 * t + (f1 - f0) / (2 * dur) * t**2)
        return amp * np.sin(ph)

    Чому саме так: частота — це швидкість зростання фази. Якщо
    частота росте лінійно, фаза росте квадратично.
    """
    t = np.arange(int(dur * sr)) / sr; ph = 2 * np.pi * (f0 * t + (f1 - f0) / (2 * dur) * t**2); return amp * np.sin(ph)


def quantize(x, bits):
    """Квантування до заданої кількості бітів.

        рівнів = 2**bits
        крок   = 2 / рівнів          (діапазон -1…1)

    Треба округлити x до найближчого рівня й повернути
    знову в діапазоні -1…1:

        L = 2**bits
        return np.round(x * (L/2 - 1)) / (L/2 - 1)
    """
    L = 2**bits; return np.round(x * (L/2 - 1)) / (L/2 - 1)


def dither(x, bits, rng=None):
    """Квантування з дизером.

    Перед округленням додаємо рівномірний шум амплітудою в пів кроку:

        крок = 2 / 2**bits
        шум  = rng.uniform(-крок/2, крок/2, len(x))
        повернути quantize(x + шум, bits)

    Саме пів кроку: більше — зайвий шум, менше — ефекту не буде.
    """
    if rng is None:
        rng = np.random.default_rng(7)
    step = 2 / 2**bits; noise = rng.uniform(-step/2, step/2, len(x)); return quantize(x + noise, bits)


# ════════════════════════════════════════════════════════════════════
#  ТЕСТИ
# ════════════════════════════════════════════════════════════════════

def check(label, ok, detail=""):
    print(f"   [{'OK ' if ok else 'НІ '}] {label}" + ("" if ok or not detail else f"   — {detail}"))
    if not ok:
        print("\n   Зупиняюся. Виправте це місце й запустіть знову.")
        raise SystemExit(1)


def self_test():
    print("\n1. Синтез тону")
    x = tone(440, 1.0)
    check("довжина відповідає тривалості", len(x) == SR, f"{len(x)} замість {SR}")
    check("амплітуда близько 0,5", abs(np.abs(x).max() - 0.5) < 0.01, f"{np.abs(x).max():.3f}")
    check("частота справді 440 Гц", abs(peak_freq(x) - 440) < 2, f"{peak_freq(x):.0f} Гц")

    print("\n2. Свіп")
    s = sweep(200, 800, 4.0)
    check("довжина правильна", len(s) == 4 * SR, str(len(s)))
    # ріжемо на п'ять вікон: частота має зростати рівномірно
    n = len(s) // 5
    pk = [peak_freq(s[i*n:(i+1)*n]) for i in range(5)]
    want = [200 + 150 * (i + 0.5) * 0.8 for i in range(5)]   # 260 380 500 620 740
    check("частота зростає від вікна до вікна",
          all(pk[i] < pk[i+1] for i in range(4)),
          " ".join(f"{p:.0f}" for p in pk))
    check("зростає саме рівномірно", max(abs(a - b) for a, b in zip(pk, want)) < 40,
          "виміряно " + " ".join(f"{p:.0f}" for p in pk)
          + ", очікувано " + " ".join(f"{w:.0f}" for w in want))

    print("\n3. Квантування")
    t = tone(440, 1.0, amp=0.9)
    for b, want in ((16, 98.1), (8, 49.9), (4, 25.8)):
        got = snr_db(t, quantize(t, b))
        check(f"{b:>2} біт: SNR близько {want:.0f} дБ", abs(got - want) < 3.0,
              f"у вас {got:.1f} дБ")
    check("4 біти дають не більше 16 різних значень",
          len(np.unique(quantize(t, 4))) <= 16, str(len(np.unique(quantize(t, 4)))))

    print("\n4. Дизер")
    q = tone(440, 2.0, amp=0.25)
    a, b = quantize(q, 4), dither(q, 4)
    check("дизер щось змінив", not np.allclose(a, b))
    check("дизер трохи погіршив SNR", snr_db(q, b) < snr_db(q, a),
          f"{snr_db(q,b):.1f} проти {snr_db(q,a):.1f} дБ")

    print("\n   Усі тести пройдено. Далі — завдання 2 і 4.\n")


def make_files():
    print("Тони")
    for f in (220, 440, 880):
        save_wav(f"tone_{f}.wav", tone(f, 2.0))

    print("\nСвіп за межу Найквіста")
    sr = 8000
    s = sweep(0, sr, 6.0, sr=sr)
    save_wav("sweep.wav", s, sr=sr)
    plot_spectrogram(s, "sweep.png", sr=sr)
    print(f"   межа Найквіста тут: {sr//2} Гц")
    for f in (3000, 5000, 7000):
        x = tone(f, 1.0, sr=sr)
        save_wav(f"alias_{f}.wav", x, sr=sr)
        print(f"   задано {f} Гц → чути {peak_freq(x, sr):.0f} Гц")

    print("\nКвантування")
    t = tone(440, 2.0, amp=0.9)
    for b in (16, 8, 4):
        q = quantize(t, b)
        save_wav(f"quant_{b}bit.wav", q)
        print(f"   {b:>2} біт: SNR {snr_db(t, q):.1f} дБ")

    print("\nДизер")
    q = tone(440, 3.0, amp=0.25)
    save_wav("dither_off.wav", quantize(q, 4))
    save_wav("dither_on.wav", dither(q, 4))
    plot_spectrum([(quantize(q, 4), "без дизеру"), (dither(q, 4), "з дизером")],
                  "dither.png")


if __name__ == "__main__":
    try:
        self_test()
    except NotImplementedError as e:
        print(f"\n   Місце {e} ще не зроблено. Знайдіть його у файлі за словом TODO,")
        print("   заповніть за підказкою поруч — і запустіть файл знову.\n")
        raise SystemExit(1)
    make_files()
    print("\n   Тепер ПОСЛУХАЙТЕ всі файли .wav — у цьому вся суть роботи.")
