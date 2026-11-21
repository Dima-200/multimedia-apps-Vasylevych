"""
Завдання 4 (власне відео) + бонус «блок 8 замість 16».
Лежить поруч із lab15.py. Запуск:  python lab15_video.py my.mp4
"""

import sys
import lab15
from lab15 import video_to_frames, gray, compensate, entropy, draw_vectors


def task4(path, limit=40, step=1):
    fr = video_to_frames(path, limit=limit)
    res = []
    for i in range(0, len(fr) - 1, step):
        a, b = gray(fr[i]), gray(fr[i + 1])
        _, _, s0, s1 = compensate(b, a)
        res.append((s0 / max(s1, 1), i))
        print(f"   кадри {i}->{i+1}: виграш {res[-1][0]:.1f}×")

    best, worst = max(res), min(res)
    print("\nнайбільший виграш:", best, "\nнайменший виграш:", worst)

    for tag, (g, i) in (("best", best), ("worst", worst)):
        a, b = gray(fr[i]), gray(fr[i + 1])
        pred, vecs, s0, s1 = compensate(b, a)
        print(f"{tag}: виграш {s0/s1:.1f}×, ентропія {entropy(b-a):.2f} → {entropy(b-pred):.2f}")
        draw_vectors(fr[i + 1], vecs, f"my_vectors_{tag}.png")


def bonus_block8(path):
    """Порівняння розміру блоку 16 і 8: біти на вектори і на залишок."""
    fr = video_to_frames(path, limit=32)
    a, b = gray(fr[30]), gray(fr[31])
    for bs in (16, 8):
        lab15.BS = bs               # compensate і search_block читають BS із модуля lab15
        pred, vecs, s0, s1 = compensate(b, a)
        bits_vec = len(vecs) * 2 * 6                       # ~6 біт на компоненту
        bits_res = entropy(b - pred) * lab15.H * lab15.W
        print(f"BS={bs}: векторів {len(vecs)}, виграш {s0/s1:.1f}×, "
              f"{bits_vec/8/1024:.2f} КБ векторів, {bits_res/8/1024:.2f} КБ залишку, "
              f"разом {(bits_vec+bits_res)/8/1024:.2f} КБ")
    lab15.BS = 16                   # повернути як було


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "my.mp4"
    task4(p)
    # bonus_block8(p)   # розкоментуйте, якщо лишився час
