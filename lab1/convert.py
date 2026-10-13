import sys
from imgio import read_ppm, write_ppm, read_bmp, write_bmp24, write_bmp8

if len(sys.argv) < 3:
    print("Використання: python convert.py вхід вихід [--8]")
    sys.exit(1)

src, dst = sys.argv[1], sys.argv[2]
palette = "--8" in sys.argv

if src.endswith(".ppm") and dst.endswith(".bmp"):
    img = read_ppm(src)
    (write_bmp8 if palette else write_bmp24)(dst, img)
elif src.endswith(".bmp") and dst.endswith(".ppm"):
    write_ppm(dst, read_bmp(src))
else:
    print("Не знаю, як перетворити ці формати")
