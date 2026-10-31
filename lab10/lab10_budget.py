"""Лабораторна 10 — Unity: бюджет текстур. Калькулятор пам'яті й план."""

import math

BPP = {'RGBA32': 32, 'ASTC 4×4': 8, 'ASTC 6×6': 128 / 36, 'ASTC 8×8': 2}
BUDGET = 24.0      


def mem(side, fmt, mips=True):
    b = side * side * BPP[fmt] / 8
    if mips:
        b *= 4 / 3
    return b / 2 ** 20         


BLOCK = {'ASTC 4×4': 4, 'ASTC 6×6': 6, 'ASTC 8×8': 8}


def mem_blocks(side, fmt):
    """Як рахує Unity: на кожному рівні міпмап блоки добудовуються до повних
    (ASTC: блок будь-якого розміру = 16 байтів). Для RGBA32 те саме, що mem()."""
    if fmt == 'RGBA32':
        return mem(side, fmt)
    bw = BLOCK[fmt]
    total, w = 0, side
    while True:
        total += math.ceil(w / bw) ** 2 * 16
        if w == 1:
            break
        w = max(1, w // 2)
    return total / 2 ** 20


scene = [('Ground_Albedo', 2048), ('Ground_Normal', 2048),
         ('Building_Albedo', 2048), ('Building_Normal', 2048),
         ('Props_Albedo', 1024), ('Props_Normal', 1024),
         ('Skybox', 2048),
         ('Character_Albedo', 1024), ('Character_Normal', 1024),
         ('UI_Atlas', 2048), ('Decals', 1024), ('Noise', 512)]

before = sum(mem(s, 'RGBA32') for _, s in scene)
print('Завдання 1. Без стиснення: %.2f МБ (бюджет %.0f МБ, перевищення у %.1f раза)'
      % (before, BUDGET, before / BUDGET))
print('   без міпмап було б: %.2f МБ' % sum(mem(s, 'RGBA32', mips=False) for _, s in scene))

plan = {
    'Ground_Albedo':    (2048, 'ASTC 6×6', 'звичайний колір, великий об\'єкт під ногами: роздільність лишаємо'),
    'Ground_Normal':    (1024, 'ASTC 4×4', 'нормалі: формат не чіпаємо, економимо роздільністю'),
    'Building_Albedo':  (2048, 'ASTC 6×6', 'звичайний колір, будівлі видно зблизька'),
    'Building_Normal':  (1024, 'ASTC 4×4', 'нормалі: економимо роздільністю'),
    'Props_Albedo':     (1024, 'ASTC 6×6', 'звичайний колір дрібних об\'єктів'),
    'Props_Normal':     (1024, 'ASTC 4×4', 'нормалі: формат 4×4 і роздільність лишаємо, щоб не втратити дрібні деталі'),
    'Skybox':           (1024, 'ASTC 6×6', 'далеке тло: гравець не підходить, роздільність вдвічі менше'),
    'Character_Albedo': (1024, 'ASTC 6×6', 'звичайний колір, персонаж біля камери: роздільність лишаємо'),
    'Character_Normal': (1024, 'ASTC 4×4', 'нормалі персонажа: формат і роздільність не чіпаємо'),
    'UI_Atlas':         (2048, 'ASTC 4×4', 'інтерфейс: у блоках по два кольори, формат дешевший за зменшення'),
    'Decals':           (1024, 'ASTC 6×6', 'наклейки: колір з прозорістю, ASTC 6×6 підтримує альфу й дає малу похибку'),
    'Noise':            (512,  'ASTC 8×8', 'службова текстура, якість ні на що не впливає'),
}

print('\nЗавдання 2. План')
print('%-18s%8s%8s  %-10s%9s%10s%13s' % ('текстура', 'було', 'стало', 'формат', 'було МБ', 'стало МБ', 'зекономлено'))
total_after = 0
total_unity = 0
rows = []
for name, side in scene:
    s2, fmt, why = plan[name]
    b, a = mem(side, 'RGBA32'), mem(s2, fmt)
    total_after += a
    total_unity += mem_blocks(s2, fmt)
    rows.append((name, side, s2, fmt, b, a, why))
    print('%-18s%8d%8d  %-10s%9.2f%10.2f%13.2f' % (name, side, s2, fmt, b, a, b - a))

print('\nРазом після плану: %.2f МБ за формулою, %.2f МБ як порахує Unity (блоки добудовуються до повних)'
      % (total_after, total_unity))
print('Вкладаємося в бюджет %.0f МБ: %s (запас %.2f МБ)'
      % (BUDGET, 'так' if total_after <= BUDGET else 'ні', BUDGET - total_after))
ui = mem(2048, 'ASTC 4×4')
print('Найдорожча текстура: UI_Atlas, %.2f МБ (%.0f%% плану)' % (ui, ui / total_after * 100))

print('\nІнтерфейс: два способи здешевити вчетверо')
print('   1024², без стиснення: %.2f МБ' % mem(1024, 'RGBA32'))
print('   2048², ASTC 4×4:      %.2f МБ' % mem(2048, 'ASTC 4×4'))

print('\nЗапитання 7. Бюджет 12 МБ: що ще відрізати')
print('   зараз %.2f МБ, треба мінімум відрізати %.2f МБ' % (total_after, total_after - 12))