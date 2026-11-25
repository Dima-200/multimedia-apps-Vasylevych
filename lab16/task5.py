from lab16 import *

jobs = [("H.264", "libx264", [18, 23, 28]),
        ("H.265", "libx265", [20, 25, 30]),
        ("AV1", "libsvtav1", [25, 35, 45])]

results = {}
for name, codec, qs in jobs:
    pts = []
    for q in qs:
        path, dt, size = encode(f"{codec}_{q}.mp4", codec=codec, crf=q)
        psnr, ssim = measure(path)
        pts.append((size / 1024, psnr))
        print(f"{name:<7}{q:>4}{size/1024:>8.0f} КБ{psnr:>8.2f}{dt:>7.1f}с")
    results[name] = pts

fig, ax = plt.subplots(figsize=(6.5, 4.2))
for name, pts in results.items():
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=name)
ax.axhline(39.5, color="gray", ls="--", label="39,5 дБ")
ax.set_xscale("log")
ax.set_xlabel("розмір, КБ")
ax.set_ylabel("PSNR, дБ")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig("codecs.png", dpi=130)
print("збережено: codecs.png")

# розмір при 39,5 дБ: лінійна інтерполяція у log-розмірі
base = None
for name, pts in results.items():
    pts = sorted(pts, key=lambda p: p[1])
    ys = [p[1] for p in pts]
    xs = [np.log(p[0]) for p in pts]
    if ys[0] <= 39.5 <= ys[-1]:
        s = float(np.exp(np.interp(39.5, ys, xs)))
        base = base or s
        print(f"{name}: ~{s:.0f} КБ при 39,5 дБ, {100*(1-s/base):+.0f}% до H.264")
    else:
        print(f"{name}: 39,5 дБ поза діапазоном виміряних точок, додайте crf")
