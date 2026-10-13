"""Лабораторна 2 — Гама і колірні простори (Python + NumPy, без cv2/skimage/PIL)."""
import struct
import sys
import numpy as np

def read_ppm(path):
    """Читає P6 PPM, повертає масив (h, w, 3) uint8. Підставте свій зчитувач з лаби 1."""
    with open(path, "rb") as f:
        data = f.read()
    pos, tokens = 0, []
    while len(tokens) < 4:
        while data[pos:pos + 1].isspace():
            pos += 1
        if data[pos:pos + 1] == b"#":
            while data[pos:pos + 1] != b"\n":
                pos += 1
            continue
        start = pos
        while not data[pos:pos + 1].isspace():
            pos += 1
        tokens.append(data[start:pos])
    pos += 1
    w, h = int(tokens[1]), int(tokens[2])
    return np.frombuffer(data, dtype=np.uint8, count=w * h * 3, offset=pos).reshape(h, w, 3)


def write_bmp(path, img):
    """Пише 24-бітний BMP. Приймає (h, w) сіре або (h, w, 3) RGB, uint8."""
    img = np.asarray(img)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    h, w, _ = img.shape
    row = w * 3
    pad = (4 - row % 4) % 4
    bgr = img[::-1, :, ::-1] 
    rows = np.zeros((h, row + pad), dtype=np.uint8)
    rows[:, :row] = bgr.reshape(h, row)
    pix = rows.tobytes()
    header = b"BM" + struct.pack("<IHHI", 54 + len(pix), 0, 0, 54)
    dib = struct.pack("<IiiHHIIiiII", 40, w, h, 1, 24, 0, len(pix), 2835, 2835, 0, 0)
    with open(path, "wb") as f:
        f.write(header + dib + pix)


def to_u8(a):
    return np.clip(np.rint(a), 0, 255).astype(np.uint8)


def to_light(code):
    """Код пікселя 0…255  ->  фізичне світло 0…1"""
    c = np.asarray(code, dtype=np.float64) / 255
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def to_code(light):
    """Фізичне світло 0…1  ->  код пікселя 0…255"""
    l = np.asarray(light, dtype=np.float64)
    s = np.where(l <= 0.0031308, l * 12.92, 1.055 * l ** (1 / 2.4) - 0.055)
    return s * 255


def task1():
    print("=== Завдання 1 ===")
    mid = (to_light(0) + to_light(255)) / 2
    print(round(float(to_code(mid))))                   
    print(round(float(to_light(128)) * 100, 1))         
    print(round(float(to_light(64)) * 100, 1)) 
    print(round(float(to_light(192)) * 100, 1))  
    print(to_light(np.array([0, 128, 255]))) 
    print(to_light(255), to_code(1.0))


def make_board(size=512, cell=8):
    yy, xx = np.indices((size, size))
    return (((yy // cell + xx // cell) % 2) * 255).astype(np.uint8)


def box_average(img, k):
    """Зменшує зображення у k разів, усереднюючи квадрати k на k"""
    h, w = img.shape
    return img.reshape(h // k, k, w // k, k).mean(axis=(1, 3))


def task2():
    print("\n=== Завдання 2 ===")
    board = make_board(512, 8)
    K = 16                                             

    naive = box_average(board.astype(np.float64), K)

    light = to_light(board)                              
    avg_light = box_average(light, K)                    
    correct = to_code(avg_light)                       

    print("наївно:   ", round(naive.mean(), 1))
    print("правильно:", round(correct.mean(), 1))
    print("різниця:  ", round(correct.mean() - naive.mean(), 1))

    res = {
        "Оригінальна шахівниця": board,
        "Зменшена наївно в sRGB": naive,
        "Зменшена в лінійному просторі": correct,
    }
    rows = []
    for name, img in res.items():
        rows.append((name, float(np.mean(img)), float(to_light(img).mean()) * 100))
    for n, c, l in rows:
        print(f"{n:32s} код {c:6.1f}   світло {l:5.1f}%")

    write_bmp("board_original.bmp", board)
    write_bmp("board_naive.bmp", to_u8(naive))
    write_bmp("board_correct.bmp", to_u8(correct))
    return rows


def rgb_to_ycbcr(img):
    x = img.astype(np.float64)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return y, cb, cr


def ycbcr_to_rgb(y, cb, cr):
    r = y + 1.402 * (cr - 128)
    g = y - 0.344136 * (cb - 128) - 0.714136 * (cr - 128)
    b = y + 1.772 * (cb - 128)
    return to_u8(np.stack([r, g, b], axis=-1))


def task3(img):
    print("\n=== Завдання 3 ===")
    y, cb, cr = rgb_to_ycbcr(img)
    back = ycbcr_to_rgb(y, cb, cr)
    print("макс. різниця:", np.abs(back.astype(int) - img.astype(int)).max())  
    gray = np.stack([y, y, y], axis=-1).clip(0, 255).astype(np.uint8)
    naive_gray = np.repeat(img.astype(np.float64).mean(axis=-1, keepdims=True), 3, axis=-1)
    write_bmp("photo_Y.bmp", gray)
    write_bmp("photo_naive_gray.bmp", to_u8(naive_gray))


def rgb_to_hsv(img):
    x = img.astype(np.float64) / 255
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx = x.max(-1); mn = x.min(-1); d = mx - mn
    h = np.zeros_like(mx)
    m = (d != 0) & (mx == r); h[m] = (60 * ((g - b)[m] / d[m])) % 360
    m = (d != 0) & (mx == g); h[m] = 60 * ((b - r)[m] / d[m]) + 120
    m = (d != 0) & (mx == b); h[m] = 60 * ((r - g)[m] / d[m]) + 240
    s = np.where(mx == 0, 0, d / np.where(mx == 0, 1, mx))
    return h, s, mx


def hsv_to_rgb(h, s, v):
    h = np.asarray(h, dtype=np.float64) % 360
    i = (h // 60).astype(int) % 6                     
    f = h / 60 - np.floor(h / 60)
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return to_u8(np.stack([r, g, b], axis=-1) * 255)


def task4(img):
    print("\n=== Завдання 4 ===")
    h, s, v = rgb_to_hsv(img)
    same = hsv_to_rgb(h, s, v)
    print("туди й назад, макс. різниця:", np.abs(same.astype(int) - img.astype(int)).max())
    write_bmp("hsv_s03.bmp", hsv_to_rgb(h, s * 0.3, v))
    write_bmp("hsv_s2.bmp", hsv_to_rgb(h, np.minimum(s * 2, 1), v))
    write_bmp("hsv_h60.bmp", hsv_to_rgb((h + 60) % 360, s, v))

def task5(rows):
    print("\n=== Завдання 5: таблиця ===")
    lines = ["| Зображення | Середній код | Середнє світло |", "| --- | --- | --- |"]
    for n, c, l in rows:
        lines.append(f"| {n} | {c:.1f} | {l:.1f}% |".replace(".", ","))
    text = "\n".join(lines)
    print(text)
    with open("report_table.txt", "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    task1()
    rows = task2()
    photo_path = sys.argv[1] if len(sys.argv) > 1 else None
    if photo_path:
        photo = read_ppm(photo_path)
        task3(photo)
        task4(photo)
    else:
        print("\n(Завдання 3-4 пропущено: передайте PPM-фото: python lab2.py photo.ppm)")
    task5(rows)
