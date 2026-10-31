"""
ЛАБОРАТОРНА РОБОТА 9 — Кодер BC1                     ЗАГОТОВКА
Розробка мультимедійних застосунків · КН-3/1 · 2026/2027

Готові: завантаження зображень, розбиття на блоки, квантування RGB565,
вимірювання похибки, кутова похибка для нормал-мап і збереження картинок.

Вам треба заповнити ТРИ місця, позначені як # TODO.

Перевірка:   python lab9.py
Тести йдуть по черзі й зупиняються на першому незробленому місці.
"""

import numpy as np
from PIL import Image
from skimage import data


# ════════════════════════════════════════════════════════════════════
#  ГОТОВЕ. Нічого тут міняти не треба.
# ════════════════════════════════════════════════════════════════════

def to565(c):
    """Квантування кольору під RGB565: 5 біт на червоний і синій,
    6 на зелений. Повертає кольори знову в діапазоні 0…255."""
    c = np.asarray(c, dtype=float)
    r = np.round(c[..., 0] / 255 * 31) / 31 * 255
    g = np.round(c[..., 1] / 255 * 63) / 63 * 255
    b = np.round(c[..., 2] / 255 * 31) / 31 * 255
    return np.stack([r, g, b], axis=-1)


def blocks_of(img):
    """Зображення (h, w, 3) → масив блоків (кількість, 16, 3)."""
    h, w, _ = img.shape
    return (img.reshape(h // 4, 4, w // 4, 4, 3)
               .transpose(0, 2, 1, 3, 4)
               .reshape(-1, 16, 3).astype(float))


def unblocks(bl, h, w):
    """Зворотна операція до blocks_of."""
    return (bl.reshape(h // 4, w // 4, 4, 4, 3)
              .transpose(0, 2, 1, 3, 4)
              .reshape(h, w, 3))


def stats(a, b):
    """Повертає (макс, середня, RMSE, PSNR)."""
    d = np.abs(np.asarray(a, float) - np.asarray(b, float))
    rmse = np.sqrt((d ** 2).mean())
    psnr = 20 * np.log10(255 / rmse) if rmse > 0 else float("inf")
    return d.max(), d.mean(), rmse, psnr


def angle_error(a, b):
    """Кутова похибка між нормалями, у градусах. Саме цим міряють
    нормал-мапи: RMSE на них нічого не каже."""
    def unit(x):
        v = np.asarray(x, float) / 255 * 2 - 1
        return v / np.linalg.norm(v, axis=-1, keepdims=True)
    cos = (unit(a) * unit(b)).sum(-1)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def save(img, name):
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(name)
    print("   збережено:", name)


# ── тестові зображення ──
def photo():
    return data.astronaut()

def gradient(N=256):
    x = np.linspace(0, 255, N)
    g = np.zeros((N, N, 3))
    for c in range(3):
        g[..., c] = x[None, :]
    return g.astype(np.uint8)

def normalmap(N=256, freq=14, amp=0.75):
    y, x = np.indices((N, N)).astype(float) / N * 2 * np.pi
    hx = amp * np.cos(freq * x) * np.cos(freq * 0.7 * y)
    hy = amp * np.cos(freq * 0.7 * x) * np.cos(freq * y)
    n = np.stack([hx, hy, np.ones_like(hx)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return ((n * 0.5 + 0.5) * 255).round().astype(np.uint8)

def stripes(N=256, wd=37):
    """Кольорові смуги шириною 37 пікселів — навмисне НЕ кратною 4,
    щоб межі потрапляли всередину блоків."""
    im = np.zeros((N, N, 3), np.uint8)
    cols = [(255,0,0), (0,255,0), (0,0,255), (255,255,0),
            (0,255,255), (255,0,255), (255,255,255)]
    for i in range(0, N, wd):
        im[:, i:i+wd] = cols[(i // wd) % len(cols)]
    return im

TESTS = {"фотографія": photo(), "градієнт": gradient(),
         "нормал-мапа": normalmap(), "різкі межі": stripes()}


# ════════════════════════════════════════════════════════════════════
#  ВАША ЧАСТИНА. Три місця.
# ════════════════════════════════════════════════════════════════════

def make_palette(c0, c1):
    """Чотири кольори BC1 із двох опорних.

    Кольори 0 і 1 — це самі опорні. Кольори 2 і 3 лежать на прямій
    між ними, на одній і на двох третинах шляху:

        колір 2 = (2*c0 + c1) / 3
        колір 3 = (c0 + 2*c1) / 3

    Повернути масив форми (4, 3). Підказка: np.stack([...]).
    """
    # TODO 1
    c0 = np.asarray(c0, float)
    c1 = np.asarray(c1, float)
    return np.stack([c0, c1, (2*c0 + c1) / 3, (c0 + 2*c1) / 3])


def pick_endpoints(blk, mode="bbox"):
    """Два опорні кольори для блоку. blk — масив (16, 3).

    mode='bbox'  — обмежувальна коробка: поканальний максимум і мінімум.
                   c0 = blk.max(axis=0),  c1 = blk.min(axis=0)
                   Увага: такого пікселя в блоці може й не бути взагалі.

    mode='pixel' — два реальні пікселі: найсвітліший і найтемніший.
                   Яскравість рахуйте як blk @ [0.299, 0.587, 0.114],
                   далі візьміть argmax і argmin.

    Обидва способи потрібні — у завданні 3 ви їх порівняєте.
    Повернути (c0, c1), кожен — масив із трьох чисел.
    """
    if mode == "bbox":
        # TODO 2a
        return blk.max(axis=0), blk.min(axis=0)
    else:
        # TODO 2b
        L = blk @ np.array([0.299, 0.587, 0.114])
        return blk[L.argmax()], blk[L.argmin()]


def encode_block(blk, mode="bbox"):
    """Стиснути один блок. Повертає (палітра (4,3), індекси (16,)).

    Порядок дій:
      1. c0, c1 = pick_endpoints(blk, mode)
      2. квантувати обидва під RGB565: to565(c0), to565(c1)
      3. pal = make_palette(c0, c1)
      4. для кожного з 16 пікселів знайти найближчий колір палітри

    Підказка для кроку 4 — без жодного циклу:
        d = ((blk[:, None, :] - pal[None, :, :]) ** 2).sum(-1)   # (16, 4)
        idx = d.argmin(axis=1)                                   # (16,)
    Запис blk[:, None, :] додає порожню вісь, щоб кожен піксель
    порівнявся з кожним кольором палітри.
    """
    # TODO 3
    c0, c1 = pick_endpoints(blk, mode)
    pal = make_palette(to565(c0), to565(c1))
    d = ((blk[:, None, :] - pal[None, :, :]) ** 2).sum(-1)   # (16, 4)
    return pal, d.argmin(axis=1)


# ════════════════════════════════════════════════════════════════════
#  ГОТОВЕ: кодування всього зображення
# ════════════════════════════════════════════════════════════════════

def bc1(img, mode="bbox"):
    h, w, _ = img.shape
    out = np.zeros((img.shape[0] // 4 * img.shape[1] // 4, 16, 3))
    for i, blk in enumerate(blocks_of(img)):
        pal, idx = encode_block(blk, mode)
        out[i] = pal[idx]
    return unblocks(out, h, w)


# ════════════════════════════════════════════════════════════════════
#  ТЕСТИ
# ════════════════════════════════════════════════════════════════════

def check(label, ok, detail=""):
    print(f"   [{'OK ' if ok else 'НІ '}] {label}" + ("" if ok or not detail else f"   — {detail}"))
    if not ok:
        print("\n   Зупиняюся. Виправте це місце й запустіть знову.")
        raise SystemExit(1)


def self_test():
    print("\n1. Палітра з чотирьох кольорів")
    pal = make_palette(np.array([255., 255., 255.]), np.array([0., 0., 0.]))
    check("форма (4, 3)", np.shape(pal) == (4, 3), str(np.shape(pal)))
    check("колір 0 — це перший опорний", np.allclose(pal[0], 255))
    check("колір 1 — це другий опорний", np.allclose(pal[1], 0))
    check("колір 2 на третині шляху", np.allclose(pal[2], 170, atol=1), f"у вас {pal[2][0]:.0f}")
    check("колір 3 на двох третинах", np.allclose(pal[3], 85, atol=1), f"у вас {pal[3][0]:.0f}")

    print("\n2. Вибір опорних кольорів")
    blk = np.zeros((16, 3))
    blk[:8] = [255, 0, 0]
    blk[8:] = [0, 255, 0]
    c0, c1 = pick_endpoints(blk, "bbox")
    check("коробка дає жовтий і чорний",
          np.allclose(sorted([c0.sum(), c1.sum()]), [0, 510]),
          f"{c0.round(0)} і {c1.round(0)}")
    c0, c1 = pick_endpoints(blk, "pixel")
    got = sorted([tuple(c0), tuple(c1)])
    check("реальні пікселі дають червоний і зелений",
          got == [(0., 255., 0.), (255., 0., 0.)], f"{c0.round(0)} і {c1.round(0)}")

    print("\n3. Кодування одного блоку")
    pal, idx = encode_block(blk, "pixel")
    check("індексів рівно 16", len(idx) == 16, str(len(idx)))
    check("усі індекси в діапазоні 0…3", idx.min() >= 0 and idx.max() <= 3)
    err = np.abs(pal[idx] - blk).max()
    check("два чисті кольори відтворено точно", err < 1, f"похибка {err:.0f}")
    pal, idx = encode_block(blk, "bbox")
    err = np.abs(pal[idx] - blk).max()
    check("а коробка на тому самому блоці помиляється", err > 100, f"похибка {err:.0f}")

    print("\n4. Ціле зображення")
    out = bc1(TESTS["фотографія"])
    check("розмір збігся", out.shape == TESTS["фотографія"].shape, str(out.shape))
    mx, mn, r, p = stats(TESTS["фотографія"], out)
    check("RMSE на фотографії близько 5,7", abs(r - 5.71) < 0.5, f"у вас {r:.2f}")
    check("PSNR близько 33", abs(p - 33.0) < 0.8, f"у вас {p:.1f}")

    print("\n   Усі тести пройдено. Можна братися за завдання 3 і 4.\n")


def run_all():
    print(f"   {'зображення':<14}{'макс':>7}{'середня':>10}{'RMSE':>8}{'PSNR':>8}")
    for name, im in TESTS.items():
        out = bc1(im)
        mx, mn, r, p = stats(im, out)
        print(f"   {name:<14}{mx:>7.0f}{mn:>10.2f}{r:>8.2f}{p:>8.1f}")
        save(out, f"bc1_{name}.png")
    ang = angle_error(TESTS["нормал-мапа"], bc1(TESTS["нормал-мапа"]))
    print(f"\n   нормал-мапа: кутова похибка {ang.mean():.2f}° у середньому, "
          f"{ang.max():.2f}° максимум")
    print(f"   пікселів із похибкою більше за 2°: {(ang > 2).mean() * 100:.1f}%")


def compare_methods():
    """Завдання 3: два способи вибору опорних кольорів на всіх зображеннях."""
    print("\nЗавдання 3. Два способи вибору опорних кольорів (RMSE)")
    print(f"   {'зображення':<14}{'коробка':>9}{'пікселі':>9}")
    for name, im in TESTS.items():
        a = bc1(im, "bbox")
        b = bc1(im, "pixel")
        print(f"   {name:<14}{stats(im, a)[2]:>9.2f}{stats(im, b)[2]:>9.2f}")
    nm = TESTS["нормал-мапа"]
    for mode, label in (("bbox", "коробка"), ("pixel", "пікселі")):
        ang = angle_error(nm, bc1(nm, mode))
        print(f"   нормал-мапа, {label}: кут {ang.mean():.2f}° у середньому, "
              f"{ang.max():.2f}° максимум")


if __name__ == "__main__":
    try:
        self_test()
    except NotImplementedError as e:
        print(f"\n   Місце {e} ще не зроблено. Знайдіть його у файлі за словом TODO,")
        print("   заповніть за підказкою поруч — і запустіть файл знову.\n")
        raise SystemExit(1)
    print("Завдання 4. Усі чотири зображення")
    run_all()
    compare_methods()
