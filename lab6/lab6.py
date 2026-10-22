"""
ЛАБОРАТОРНА РОБОТА 6 — JPEG-lite                     ЗАГОТОВКА
Розробка мультимедійних застосунків · КН-3/1 · 2026/2027

Усе введення-виведення, розбиття на блоки, вимірювання та побудова
графіка вже написані. Вам треба заповнити ЧОТИРИ місця, позначені
як  # TODO.  Нижче кожного TODO є підказка.

Щоб перевірити себе, запустіть файл:   python lab6.py
Тести виконуються по черзі й зупиняються на першому незробленому місці.
"""

import io
import zlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from skimage import data
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


# ════════════════════════════════════════════════════════════════════
#  ГОТОВЕ. Нічого тут міняти не треба.
# ════════════════════════════════════════════════════════════════════

# Базова таблиця квантування JPEG для яскравості (та сама, що на лекції).
Q50 = np.array([
    [16, 11, 10, 16,  24,  40,  51,  61],
    [12, 12, 14, 19,  26,  58,  60,  55],
    [14, 13, 16, 24,  40,  57,  69,  56],
    [14, 17, 22, 29,  51,  87,  80,  62],
    [18, 22, 37, 56,  68, 109, 103,  77],
    [24, 35, 55, 64,  81, 104, 113,  92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103,  99],
], dtype=float)


def load_gray(path=None):
    """Чорно-біле зображення. Без аргументу — вбудоване фото 512×512.
    Сторони обрізаються до кратних 8, щоб воно рівно ділилося на блоки."""
    if path is None:
        img = np.array(Image.fromarray(data.astronaut()).convert("L"))
    else:
        img = np.array(Image.open(path).convert("L"))
    h, w = img.shape
    return img[:h // 8 * 8, :w // 8 * 8]


def save_gray(img, path):
    Image.fromarray(img.astype(np.uint8)).save(path)
    print("   збережено:", path)


def blockify(img):
    """Зображення (h, w) → масив блоків (кількість, 8, 8).
    Самі числа не змінюються, міняється лише їхнє групування."""
    h, w = img.shape
    return (img.reshape(h // 8, 8, w // 8, 8)
               .transpose(0, 2, 1, 3)
               .reshape(-1, 8, 8))


def unblockify(blocks, h, w):
    """Зворотна операція до blockify."""
    return (blocks.reshape(h // 8, w // 8, 8, 8)
                  .transpose(0, 2, 1, 3)
                  .reshape(h, w))


def packed(coeffs):
    """Скільки байтів займуть квантовані коефіцієнти.

    Справжній JPEG складає їх зигзагом і кодує Хаффманом. Ми замість
    цього стискаємо їх zlib — роль та сама: порахувати, скільки місця
    насправді займають числа, де більшість нулів."""
    return len(zlib.compress(coeffs.astype(np.int16).tobytes(), 9))


def quality_of(orig, out):
    """Повертає (PSNR у дБ, SSIM)."""
    return (psnr(orig, out, data_range=255),
            ssim(orig, out, data_range=255))


def jpeg_bytes(img, q):
    """Для порівняння: скільки займе справжній JPEG тієї ж якості."""
    bio = io.BytesIO()
    Image.fromarray(img).save(bio, "JPEG", quality=q)
    return len(bio.getvalue())


def plot_curve(rows, path="curve.png"):
    """rows — список кортежів (q, байти, PSNR, SSIM)."""
    q  = [r[0] for r in rows]
    kb = [r[1] / 1024 for r in rows]
    P  = [r[2] for r in rows]
    S  = [r[3] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(q, kb, "o-", color="#B35A1F", lw=2)
    ax[0].set_xlabel("якість q"); ax[0].set_ylabel("розмір, КБ")
    ax[0].set_title("Розмір"); ax[0].grid(alpha=0.3)
    ax[1].plot(q, P, "o-", color="#1F6F66", lw=2)
    ax[1].set_xlabel("якість q"); ax[1].set_ylabel("PSNR, дБ")
    ax[1].tick_params(axis="y", labelcolor="#1F6F66")
    tw = ax[1].twinx()
    tw.plot(q, S, "^-", color="#2E5E8A", lw=2)
    tw.set_ylabel("SSIM"); tw.tick_params(axis="y", labelcolor="#2E5E8A")
    ax[1].set_title("Якість"); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("   збережено:", path)


# ════════════════════════════════════════════════════════════════════
#  ВАША ЧАСТИНА. Чотири місця.
# ════════════════════════════════════════════════════════════════════

def dct_matrix(N=8):
    """Матриця базису DCT розміру N×N.

    Рядок k цієї матриці — це k-й візерунок з лекції, записаний числами.
    Формула для елемента в рядку k, стовпці n:

        a = sqrt(1/N)   якщо k == 0,   інакше  sqrt(2/N)
        D[k, n] = a * cos(pi * (2*n + 1) * k / (2*N))

    Зробіть два вкладені цикли по k і по n.
    """
    D = np.zeros((N, N))
    # TODO 1: заповнити D за формулою вище
    for k in range(N):
        a = np.sqrt(1 / N) if k == 0 else np.sqrt(2 / N)
        for n in range(N):
            D[k, n] = a * np.cos(np.pi * (2 * n + 1) * k / (2 * N))
    return D


def dct2(block):
    """Пряме DCT блоку 8×8: D @ block @ D.T

    @ — це множення матриць у NumPy. D.T — транспонована матриця.
    Один рядок коду."""
    D = dct_matrix()
    # TODO 2a: повернути D @ block @ D.T
    return D @ block @ D.T


def idct2(coeffs):
    """Зворотне DCT: D.T @ coeffs @ D   (ті самі множники, інший порядок)."""
    D = dct_matrix()
    # TODO 2b: повернути D.T @ coeffs @ D
    return D.T @ coeffs @ D


def qtable(q):
    """Таблиця квантування для заданої якості q (від 1 до 99).

    Та сама формула, що в справжньому JPEG:

        S = 5000 / q        якщо q < 50
        S = 200 - 2 * q     якщо q >= 50
        T = floor((S * Q50 + 50) / 100),  обмежене знизу 1, зверху 255

    Підказки: np.floor(...), np.clip(масив, 1, 255).
    Перевірка: при q = 50 має вийти рівно Q50."""
    q = max(1, min(99, q))
    # TODO 3: порахувати й повернути таблицю
    S = 5000 / q if q < 50 else 200 - 2 * q
    return np.clip(np.floor((S * Q50 + 50) / 100), 1, 255)


def encode(img, q):
    """Зображення → квантовані коефіцієнти.

    Порядок дій:
      1. Q = qtable(q)
      2. відняти 128 від усіх пікселів (і перевести у float!)
      3. розбити на блоки через blockify
      4. для кожного блоку: np.round(dct2(блок) / Q)
      5. зібрати результат у масив np.int16

    Повернути (коефіцієнти, Q).

    Підказка для кроку 4:
        coeffs = np.array([np.round(dct2(b) / Q) for b in blocks])
    """
    # TODO 4a
    Q = qtable(q)
    blocks = blockify(img.astype(np.float64) - 128)
    coeffs = np.array([np.round(dct2(b) / Q) for b in blocks])
    return coeffs.astype(np.int16), Q


def decode(coeffs, Q, h, w):
    """Квантовані коефіцієнти → зображення. Усе те саме, але навпаки:

      1. для кожного блоку: idct2(коефіцієнти * Q)
      2. зібрати назад через unblockify(блоки, h, w)
      3. додати 128
      4. обрізати в діапазон 0…255 через np.clip і перевести в np.uint8
    """
    # TODO 4b
    blocks = np.array([idct2(c * Q) for c in coeffs])
    img = unblockify(blocks, h, w) + 128
    return np.clip(img, 0, 255).round().astype(np.uint8)


# ════════════════════════════════════════════════════════════════════
#  ТЕСТИ. Запускаються по черзі, зупиняються на першій помилці.
# ════════════════════════════════════════════════════════════════════

def check(label, ok, detail=""):
    print(f"   [{'OK ' if ok else 'НІ '}] {label}" + ("" if ok or not detail else f"   — {detail}"))
    if not ok:
        print("\n   Зупиняюся. Виправте це місце й запустіть знову.")
        raise SystemExit(1)


def self_test():
    img = load_gray()
    h, w = img.shape

    print("\n1. Матриця базису DCT")
    D = dct_matrix()
    check("розмір 8×8", D.shape == (8, 8))
    check("не порожня", np.abs(D).sum() > 0, "(TODO 1 ще не зроблено?)")
    check("рядки одиничної довжини", np.allclose((D**2).sum(axis=1), 1),
          f"сума квадратів рядка 0 = {(D[0]**2).sum():.3f}")
    check("рядки взаємно перпендикулярні", np.allclose(D @ D.T, np.eye(8), atol=1e-9))

    print("\n2. DCT туди й назад")
    b = img[88:96, 444:452].astype(float) - 128
    back = idct2(dct2(b))
    e = np.abs(back - b).max()
    check("блок відновлюється точно", e < 1e-6, f"макс похибка {e:.1e}")

    print("\n3. Таблиця квантування")
    check("при q=50 збігається з Q50", np.allclose(qtable(50), Q50))
    check("при більшій якості числа менші", qtable(90).mean() < qtable(50).mean(),
          f"{qtable(90).mean():.0f} проти {qtable(50).mean():.0f}")
    check("ніде не перевищує 255", qtable(1).max() <= 255)
    check("ніде не менше 1", qtable(99).min() >= 1)

    print("\n4. Квантування контрольного блоку")
    cq = np.round(dct2(b) / qtable(50))
    nz = int((cq != 0).sum())
    check("лишилося рівно 2 ненульові числа з 64", nz == 2, f"у вас {nz}")
    check("перші два числа — 26 і 16",
          [cq[0, 0], cq[0, 1]] == [26, 16], f"у вас {cq[0,0]:.0f} і {cq[0,1]:.0f}")

    print("\n5. Кодер і декодер на всьому зображенні")
    co, Q = encode(img, 50)
    check("форма коефіцієнтів", co.shape == (h * w // 64, 8, 8), str(co.shape))
    out = decode(co, Q, h, w)
    check("форма відновленого", out.shape == img.shape, str(out.shape))
    check("тип uint8", out.dtype == np.uint8, str(out.dtype))
    alive = (co != 0).sum(axis=(1, 2)).mean()
    check("у середньому вижило близько 8 коефіцієнтів із 64",
          abs(alive - 8.0) < 0.8, f"у вас {alive:.1f}")
    P, S = quality_of(img, out)
    check("PSNR при q=50 близько 34.7", abs(P - 34.7) < 0.6, f"у вас {P:.1f}")
    check("SSIM при q=50 близько 0.956", abs(S - 0.956) < 0.01, f"у вас {S:.3f}")

    print("\n   Усі тести пройдено. Можна братися за завдання 4 і 5.\n")
    return img


def build_curve(img):
    """Завдання 4: крива якості. Запускається після того, як тести пройшли."""
    h, w = img.shape
    rows = []
    print(f"   {'q':>3}{'байт':>9}{'стиснення':>11}{'нулів':>8}{'PSNR':>8}{'SSIM':>8}{'JPEG, байт':>12}")
    for q in (95, 90, 75, 50, 30, 15, 5):
        co, Q = encode(img, q)
        out = decode(co, Q, h, w)
        n = packed(co)
        P, S = quality_of(img, out)
        rows.append((q, n, P, S))
        print(f"   {q:>3}{n:>9}{h*w/n:>10.1f}x"
              f"{(co == 0).mean()*100:>7.1f}%{P:>8.1f}{S:>8.3f}{jpeg_bytes(img, q):>12}")
        save_gray(out, f"q{q:02d}.png")
    plot_curve(rows)
    return rows


if __name__ == "__main__":
    image = self_test()
    print("Завдання 4. Крива якості")
    build_curve(image)
