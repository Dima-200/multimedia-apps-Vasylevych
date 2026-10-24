import time
import numpy as np
from PIL import Image
from skimage import data
import cv2


def conv2(img, K):
    kh, kw = K.shape
    ph, pw = kh // 2, kw // 2
    pad = np.pad(img, ((ph, ph), (pw, pw)), mode='edge')
    out = np.zeros_like(img)
    for i in range(kh):
        for j in range(kw):
            out += K[i, j] * pad[i:i + img.shape[0], j:j + img.shape[1]]
    return out


gray = np.array(Image.fromarray(data.astronaut()).convert('L')).astype(float)


def save(arr, path):
    Image.fromarray(np.clip(arr, 0, 255).round().astype(np.uint8)).save(path)
    print('   збережено:', path)


def gauss1d(sigma, n):
    x = np.arange(n) - n // 2
    k = np.exp(-x ** 2 / (2 * sigma ** 2))
    return k / k.sum()             


def conv_sep(img, k):
    n = len(k) // 2
    pad = np.pad(img, ((0, 0), (n, n)), mode='edge')
    tmp = np.zeros_like(img)
    for j, v in enumerate(k):
        tmp += v * pad[:, j:j + img.shape[1]]
    pad = np.pad(tmp, ((n, n), (0, 0)), mode='edge')
    out = np.zeros_like(img)
    for i, v in enumerate(k):
        out += v * pad[i:i + img.shape[0], :]
    return out


def best(f, r=5):
    ts = []
    for _ in range(r):
        t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    return min(ts) * 1000               # мілісекунди


if __name__ == "__main__":
    print("=== Завдання 1: перевірка ===")
    avg = conv2(gray, np.ones((3, 3)) / 9)
    print(f"   середня яскравість: оригінал {gray.mean():.2f}, після ядра 3x3 {avg.mean():.2f}")

    print("\n=== Завдання 2: три фільтри ===")
    save(gray, 'original.png')

    k = gauss1d(1.6, 7)
    BLUR = np.outer(k, k)
    blur = conv2(gray, BLUR)
    save(blur, 'blur.png')

    SX = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], float)
    SY = SX.T
    gx, gy = conv2(gray, SX), conv2(gray, SY)
    edges = np.hypot(gx, gy)
    edges = edges / edges.max() * 255  
    save(edges, 'sobel.png')

    detail = gray - blur         
    sharp = np.clip(gray + 1.2 * detail, 0, 255)
    save(sharp, 'sharp.png')

    print("\n=== Завдання 3: роздільне розмиття й заміри ===")
    print(f"{'ядро':>7}{'звичайне, мс':>14}{'роздільне, мс':>15}{'виграш':>9}{'розбіжність':>14}")
    for n in (3, 5, 9, 15, 31):
        k = gauss1d(max(n / 5, 0.6), n)
        K = np.outer(k, k)
        diff = np.abs(conv2(gray, K) - conv_sep(gray, k)).max()
        ta = best(lambda: conv2(gray, K))
        tb = best(lambda: conv_sep(gray, k))
        print(f"{n:>3}x{n:<3}{ta:14.1f}{tb:15.1f}{ta / tb:8.1f}x{diff:14.1e}")

    print("\n=== Завдання 4: порівняння з OpenCV ===")
    n, sigma = 15, 3.0
    k = gauss1d(sigma, n)
    mine = conv_sep(gray, k)
    lib = cv2.GaussianBlur(gray, (n, n), sigma, borderType=cv2.BORDER_REPLICATE)
    print('розбіжність:', np.abs(mine - lib).max())
    print('ваш час:   %.1f мс' % best(lambda: conv_sep(gray, k)))
    print('OpenCV:    %.1f мс' % best(lambda: cv2.GaussianBlur(gray, (n, n), sigma,
                                                               borderType=cv2.BORDER_REPLICATE)))
