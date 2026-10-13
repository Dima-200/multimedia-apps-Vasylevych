import os
import numpy as np
from imgio import *

# Завдання 1
img = np.zeros((4, 6, 3), dtype=np.uint8)
img[0:2, 0:3] = (255, 0, 0)
img[0:2, 3:6] = (0, 255, 0)
img[2:4, 0:3] = (0, 0, 255)
img[2:4, 3:6] = (255, 255, 255)
with open("test.ppm", "wb") as f:
    f.write(b"P6\n6 4\n255\n")
    f.write(img.tobytes())

a = read_ppm("test.ppm")
print(a.shape, a[0, 0], a[3, 5])

# Завдання 2-3
write_bmp24("test.bmp", a)
print("розмір test.bmp:", os.path.getsize("test.bmp"))
b = read_bmp24("test.bmp")
print("array_equal:", np.array_equal(a, b))

# Завдання 4
write_bmp8("test8.bmp", a)
print("розмір test8.bmp:", os.path.getsize("test8.bmp"),
      "| ppm:", os.path.getsize("test.ppm"))
print("array_equal (8 біт):", np.array_equal(a, read_bmp8("test8.bmp")))
