import struct
import numpy as np


# ---------- PPM ----------
def read_ppm(path):
    with open(path, "rb") as f:
        data = f.read()
    pos = 0
    fields = []
    while len(fields) < 4:
        while data[pos:pos + 1].isspace():
            pos += 1
        if data[pos:pos + 1] == b"#":
            while data[pos:pos + 1] not in (b"\n", b""):
                pos += 1
            continue
        start = pos
        while not data[pos:pos + 1].isspace():
            pos += 1
        fields.append(data[start:pos])
    if fields[0] != b"P6":
        raise ValueError("Це не PPM формату P6")
    width, height, maxval = int(fields[1]), int(fields[2]), int(fields[3])
    if maxval != 255:
        raise ValueError("Підтримується лише maxval = 255")
    pos += 1
    need = width * height * 3
    if len(data) - pos < need:
        raise ValueError("Файл обірваний: пікселів менше, ніж треба")
    pixels = np.frombuffer(data, dtype=np.uint8, count=need, offset=pos)
    return pixels.reshape(height, width, 3)


def write_ppm(path, img):
    height, width, _ = img.shape
    with open(path, "wb") as f:
        f.write(b"P6\n%d %d\n255\n" % (width, height))
        f.write(np.ascontiguousarray(img).tobytes())


# ---------- BMP 24 біт ----------
def write_bmp24(path, img):
    height, width, _ = img.shape
    stride = ((width * 24 + 31) // 32) * 4
    padding = stride - width * 3
    offbits = 54
    filesize = offbits + stride * height

    out = bytearray()
    out += struct.pack("<2sIHHI", b"BM", filesize, 0, 0, offbits)
    out += struct.pack("<IiiHHIIiiII", 40, width, height, 1, 24,
                       0, stride * height, 2835, 2835, 0, 0)
    for y in range(height - 1, -1, -1):          # рядки знизу вгору
        for x in range(width):
            r, g, b = img[y][x]
            out += bytes((b, g, r))              # порядок B, G, R
        out += b"\x00" * padding
    with open(path, "wb") as f:
        f.write(out)


def read_bmp24(path):
    with open(path, "rb") as f:
        data = f.read()
    offbits = struct.unpack_from("<I", data, 10)[0]
    width = struct.unpack_from("<i", data, 18)[0]
    height = struct.unpack_from("<i", data, 22)[0]
    bits = struct.unpack_from("<H", data, 28)[0]
    if bits != 24:
        raise ValueError("Це не 24-бітний BMP")
    stride = ((width * bits + 31) // 32) * 4
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        start = offbits + y * stride             # y-й рядок рахуючи знизу
        for x in range(width):
            b = data[start + x * 3 + 0]
            g = data[start + x * 3 + 1]
            r = data[start + x * 3 + 2]
            img[height - 1 - y, x] = (r, g, b)
    return img


# ---------- BMP 8 біт, палітра 6x6x6 ----------
def build_palette():
    palette = bytearray()
    for i in range(256):
        if i < 216:
            r = (i // 36) * 51
            g = ((i // 6) % 6) * 51
            b = (i % 6) * 51
        else:
            r = g = b = 0
        palette += bytes((b, g, r, 0))
    return palette


def color_index(r, g, b):
    return 36 * (int(r) // 51) + 6 * (int(g) // 51) + (int(b) // 51)


def write_bmp8(path, img):
    height, width, _ = img.shape
    stride = ((width * 8 + 31) // 32) * 4
    padding = stride - width
    offbits = 54 + 1024                          # 1078
    filesize = offbits + stride * height

    out = bytearray()
    out += struct.pack("<2sIHHI", b"BM", filesize, 0, 0, offbits)
    out += struct.pack("<IiiHHIIiiII", 40, width, height, 1, 8,
                       0, stride * height, 2835, 2835, 256, 0)
    out += build_palette()
    for y in range(height - 1, -1, -1):
        for x in range(width):
            r, g, b = img[y][x]
            out.append(color_index(r, g, b))
        out += b"\x00" * padding
    with open(path, "wb") as f:
        f.write(out)


def read_bmp8(path):
    with open(path, "rb") as f:
        data = f.read()
    offbits = struct.unpack_from("<I", data, 10)[0]
    width = struct.unpack_from("<i", data, 18)[0]
    height = struct.unpack_from("<i", data, 22)[0]
    bits = struct.unpack_from("<H", data, 28)[0]
    if bits != 8:
        raise ValueError("Це не 8-бітний BMP")
    stride = ((width * 8 + 31) // 32) * 4
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        start = offbits + y * stride
        for x in range(width):
            i = data[start + x]
            p = 54 + i * 4                       # запис палітри: B, G, R, 0
            img[height - 1 - y, x] = (data[p + 2], data[p + 1], data[p])
    return img


def read_bmp(path):
    with open(path, "rb") as f:
        head = f.read(30)
    bits = struct.unpack_from("<H", head, 28)[0]
    return read_bmp8(path) if bits == 8 else read_bmp24(path)
