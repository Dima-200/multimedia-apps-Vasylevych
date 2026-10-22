
import numpy as np
from collections import Counter
from PIL import Image


grad = np.tile(np.linspace(0, 255, 512).astype(np.uint8), (384, 1))
rng = np.random.default_rng(1)
noise = rng.integers(0, 256, (384, 512)).astype(np.uint8)
scr = np.array(Image.open("screen.png").convert("L").resize((512, 384)))
photo = np.array(Image.open("photo.jpg").convert("L").resize((512, 384)))

images = {"градієнт": grad, "білий шум": noise,
          "скріншот": scr, "фотографія": photo}


def entropy(img):
    values, counts = np.unique(img, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def filter_row(cur, prev, f):
    a = np.concatenate(([0], cur[:-1])).astype(np.uint8)  
    b = prev                                                
    if f == 0:
        pred = np.zeros_like(cur)
    elif f == 1:
        pred = a
    else:
        pred = b
    return ((cur.astype(int) - pred.astype(int)) % 256).astype(np.uint8)


def filter_image(img, f):
    out = np.zeros_like(img)
    prev = np.zeros(img.shape[1], dtype=np.uint8)
    for y in range(img.shape[0]):
        out[y] = filter_row(img[y], prev, f)
        prev = img[y]           
    return out


def unfilter_row(enc, prev, f):
    n = len(enc)
    out = np.zeros(n, dtype=np.uint8)
    for x in range(n):
        a = out[x - 1] if x > 0 else 0
        b = prev[x]
        pred = 0 if f == 0 else (a if f == 1 else b)
        out[x] = (int(enc[x]) + int(pred)) % 256
    return out


def unfilter_image(enc, f):
    out = np.zeros_like(enc)
    prev = np.zeros(enc.shape[1], dtype=np.uint8)
    for y in range(enc.shape[0]):
        out[y] = unfilter_row(enc[y], prev, f)
        prev = out[y]           
    return out


def sum_abs(row):
    r = row.astype(int)
    signed = np.where(r > 127, r - 256, r)
    return int(np.abs(signed).sum())


def filter_adaptive(img):
    out = np.zeros_like(img)
    chosen = []
    prev = np.zeros(img.shape[1], dtype=np.uint8)
    for y in range(img.shape[0]):
        best_f, best_row, best_sum = 0, None, None
        for f in (0, 1, 2):
            row = filter_row(img[y], prev, f)
            s = sum_abs(row)
            if best_sum is None or s < best_sum:
                best_f, best_row, best_sum = f, row, s
        out[y] = best_row
        chosen.append(best_f)
        prev = img[y]
    return out, chosen


def unfilter_adaptive(enc, chosen):
    out = np.zeros_like(enc)
    prev = np.zeros(enc.shape[1], dtype=np.uint8)
    for y in range(enc.shape[0]):
        out[y] = unfilter_row(enc[y], prev, chosen[y])
        prev = out[y]
    return out


if __name__ == "__main__":
    print("=== Перевірка завдання 1 ===")
    row = np.array([40, 42, 44, 46, 48, 50, 52, 54], dtype=np.uint8)
    zeros = np.zeros(8, dtype=np.uint8)
    print(filter_row(row, zeros, 1))   
    print(filter_row(row, zeros, 0)) 
    print(filter_row(row, row, 2))    

    names = {0: "None", 1: "Sub", 2: "Up"}
    summary = []
    lines = ["| Зображення | До | None | Sub | Up | Адаптивний |",
             "| --- | --- | --- | --- | --- | --- |"]

    print("\n=== Завдання 2-4: ентропія, зворотне перетворення, адаптивний вибір ===")
    for title, img in images.items():
        e0 = entropy(img)
        ents = {}
        for f in (0, 1, 2):
            enc = filter_image(img, f)
            ok = np.array_equal(img, unfilter_image(enc, f))
            print(f"{title:<12} фільтр {names[f]:<5} відновлення: {ok}")
            ents[f] = entropy(enc)
        enc_a, chosen = filter_adaptive(img)
        ok = np.array_equal(img, unfilter_adaptive(enc_a, chosen))
        ea = entropy(enc_a)
        print(f"{title:<12} адаптивний відновлення: {ok}  вибір: {dict(Counter(chosen))}")

        lines.append(f"| {title.capitalize()} | {e0:.2f} | {ents[0]:.2f} | "
                     f"{ents[1]:.2f} | {ents[2]:.2f} | {ea:.2f} |".replace(".", ","))

        cands = {names[f]: ents[f] for f in (0, 1, 2)}
        cands["Адаптивний"] = ea
        best = min(cands, key=cands.get)
        gain = (e0 - cands[best]) / e0 * 100
        summary.append((title, best, e0, cands[best], gain))

    table1 = "\n".join(lines)
    print("\n=== Таблиця 1: ентропія, біт на піксель ===")
    print(table1)

    lines2 = ["| Тип зображення | Оптимальний фільтр | Ентропія до / після | Виграш, % |",
              "| --- | --- | --- | --- |"]
    for t, b, e0, e1, g in summary:
        lines2.append(f"| {t.capitalize()} | {b} | {e0:.2f} / {e1:.2f} | {g:.1f} |".replace(".", ","))
    table2 = "\n".join(lines2)
    print("\n=== Таблиця 2: підсумкова ===")
    print(table2)

    with open("table4.txt", "w", encoding="utf-8") as f:
        f.write(table1 + "\n\n" + table2 + "\n")