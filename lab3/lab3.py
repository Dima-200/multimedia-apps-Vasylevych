"""Лабораторна 3 — Ентропійне стиснення (Python + NumPy; Pillow лише для відкриття фото й PNG)."""
import io
import numpy as np
from PIL import Image

W, H = 512, 384
ORIGINAL = W * H            # 196 608 байтів


# ---------- Підготовка зображень ----------
grad = np.tile(np.linspace(0, 255, 512).astype(np.uint8), (384, 1))
rng = np.random.default_rng(1)
noise = rng.integers(0, 256, (384, 512)).astype(np.uint8)
# скріншот: збережіть знімок екрана як screen.png поруч зі скриптом
scr = np.array(Image.open("screen.png").convert("L").resize((512, 384)))
# фотографія: photo.jpg поруч зі скриптом
photo = np.array(Image.open("photo.jpg").convert("L").resize((512, 384)))

images = {"градієнт": grad, "білий шум": noise,
          "скріншот": scr, "фотографія": photo}


# ---------- Завдання 1: ентропія ----------
def entropy(img):
    values, counts = np.unique(img, return_counts=True)   # які значення є і скільки разів
    p = counts / counts.sum()                              # частки, у сумі 1
    return float(-(p * np.log2(p)).sum())


# ---------- Завдання 2: RLE ----------
def rle_size(img):
    data = img.ravel()          # розгорнули у плоский масив
    n = len(data)
    i = 0
    out = 0                     # скільки байтів займе результат
    while i < n:
        j = i
        # 1. скільки однакових байтів іде підряд, починаючи з i
        # 2. серія не довша за 255 (лічильник — один байт)
        while j + 1 < n and data[j + 1] == data[i] and (j + 1 - i) < 255:
            j += 1
        out += 2                # 3. одна пара = 2 байти (лічильник + значення)
        i = j + 1               # 4. переходимо до кінця знайденої серії
    return out


# ---------- Завдання 3: Хаффман (заготовка) ----------
def huffman_codes(freq):
    """freq — словник {значення пікселя: скільки разів трапилось}
       повертає   {значення пікселя: код рядком із нулів та одиниць}"""
    if len(freq) == 1:
        return {list(freq)[0]: "0"}
    nodes = [[n, {v: ""}] for v, n in freq.items()]
    while len(nodes) > 1:
        nodes.sort(key=lambda node: node[0])
        a = nodes.pop(0)
        b = nodes.pop(0)
        for v in a[1]:  a[1][v] = "0" + a[1][v]
        for v in b[1]:  b[1][v] = "1" + b[1][v]
        nodes.append([a[0] + b[0], {**a[1], **b[1]}])
    return nodes[0][1]


def huffman_stats(img):
    values, counts = np.unique(img, return_counts=True)
    freq = {int(v): int(c) for v, c in zip(values, counts)}
    codes = huffman_codes(freq)
    total = sum(freq.values())
    avg = sum(freq[v] * len(codes[v]) for v in freq) / total
    lengths = [len(c) for c in codes.values()]
    return avg, min(lengths), max(lengths)


# ---------- Завдання 4: PNG ----------
def png_size(img):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG", optimize=True)
    return len(buf.getvalue())


# ---------- Запуск ----------
if __name__ == "__main__":
    print("=== Завдання 1: ентропія ===")
    ent = {}
    for name, im in images.items():
        ent[name] = entropy(im)
        print(f"{name:<12} {ent[name]:.2f} біта на піксель")

    print("\n=== Завдання 2: RLE ===")
    rle = {}
    for name, im in images.items():
        rle[name] = rle_size(im) / ORIGINAL * 100
        print(f"{name:<12} {rle[name]:.1f}%")

    print("\n=== Завдання 3: Хаффман ===")
    huff = {}
    for name, im in images.items():
        avg, lo, hi = huffman_stats(im)
        huff[name] = avg
        print(f"{name:<12} ентропія {ent[name]:.2f}  Хаффман {avg:.2f}  коди від {lo} до {hi} біт")

    print("\n=== Завдання 4: PNG ===")
    png = {}
    for name, im in images.items():
        png[name] = png_size(im) / ORIGINAL * 100
        print(f"{name:<12} {png[name]:.2f}%")

    print("\n=== Таблиця 1 ===")
    lines = ["| Зображення | Ентропія | Хаффман | RLE, % | PNG, % |",
             "| --- | --- | --- | --- | --- |"]
    for name in images:
        lines.append(f"| {name.capitalize()} | {ent[name]:.2f} | {huff[name]:.2f} "
                     f"| {rle[name]:.1f} | {png[name]:.1f} |".replace(".", ","))
    table = "\n".join(lines)
    print(table)
    with open("table.txt", "w", encoding="utf-8") as f:
        f.write(table + "\n")

    # зберігаємо тестові зображення для архіву
    for name, fn in [("градієнт", "gradient.png"), ("білий шум", "noise.png"),
                     ("скріншот", "screenshot_gray.png"), ("фотографія", "photo_gray.png")]:
        Image.fromarray(images[name]).save(fn)
