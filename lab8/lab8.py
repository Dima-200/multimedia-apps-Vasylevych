import numpy as np
from PIL import Image
from skimage import data

gray = np.array(Image.fromarray(data.astronaut()).convert('L'))


def to_light(c):                    
    c = c / 255.0
    return np.where(c <= 0.04045, c / 12.92,
                    ((c + 0.055) / 1.055) ** 2.4)


def to_code(L):                     
    c = np.where(L <= 0.0031308, L * 12.92,
                 1.055 * L ** (1 / 2.4) - 0.055)
    return np.clip(c * 255, 0, 255)

def half_naive(img):
    h, w = img.shape
    return img.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))


def half_linear(img):
    L = to_light(img)
    h, w = L.shape
    return to_code(L.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3)))


def checker(N=512, cell=4, ang=18):
    y, x = np.indices((N, N)).astype(float) - N / 2
    a = np.deg2rad(ang)
    xr = x * np.cos(a) - y * np.sin(a)
    yr = x * np.sin(a) + y * np.cos(a)
    return (((np.floor(xr / cell) + np.floor(yr / cell)) % 2) * 255)


if __name__ == "__main__":
    print("=== Завдання 1: збільшення ===")
    small = gray[340:356, 372:388]                
    im = Image.fromarray(small)
    im.resize((320, 320), Image.NEAREST).save('up_nearest.png')
    im.resize((320, 320), Image.BILINEAR).save('up_bilinear.png')
    im.resize((320, 320), Image.LANCZOS).save('up_lanczos.png')
    a = np.array(Image.open('up_nearest.png')).astype(int)
    b = np.array(Image.open('up_bilinear.png')).astype(int)
    c = np.array(Image.open('up_lanczos.png')).astype(int)
    print('   nearest ≠ bilinear:', not np.array_equal(a, b),
          '  bilinear ≠ lanczos:', not np.array_equal(b, c))
    print('   у nearest блок 20×20 однорідний:', len(np.unique(a[:20, :20])) == 1)

    print("\n=== Завдання 2: зменшення й аліасинг ===")
    ch = checker()
    k = 4
    nearest = ch[::k, ::k]                     
    area = ch.reshape(512 // k, k, 512 // k, k).mean(axis=(1, 3))
    for name, im2 in (('ds_nearest', nearest), ('ds_area', area)):
        Image.fromarray(im2.astype(np.uint8)).save(name + '.png')
        print('   ', name, 'середнє %.1f  розкид %.1f' % (im2.mean(), im2.std()))

    print("\n=== Завдання 3: ланцюжок міпмап ===")
    g = gray.astype(float)
    level = 0
    while g.shape[0] >= 2:
        Image.fromarray(g.astype(np.uint8)).save(f'mip_{level}.png')
        print('   ', level, g.shape[0], 'середня яскравість %.1f' % g.mean())
        g = half_naive(g)
        level += 1
    print('   рівнів:', level)

    print("\n=== Завдання 4: скільки коштує гама ===")
    cb = np.indices((256, 256)).sum(axis=0) % 2 * 255.0
    print('   наївно:    %.1f' % half_naive(cb).mean())
    print('   правильно: %.1f' % half_linear(cb).mean())

    print('\n   рівень  розмір  наївно  правильно  різниця')
    gn = gray.astype(float)
    gl = gray.astype(float)
    print('   %6d %7d %7.1f %9.1f %+8.1f' % (0, 512, gn.mean(), gl.mean(), 0))
    for lv in range(1, 6):
        gn, gl = half_naive(gn), half_linear(gl)
        print('   %6d %7d %7.1f %9.1f %+8.1f'
              % (lv, gn.shape[0], gn.mean(), gl.mean(), gl.mean() - gn.mean()))
        
    print("\n=== Додатково: Pillow ===")
    imc = Image.fromarray(ch.astype(np.uint8))
    imc.resize((128, 128), Image.NEAREST).save('lib_nearest.png')
    imc.resize((128, 128), Image.BOX).save('lib_box.png')
    imc.resize((128, 128), Image.LANCZOS).save('lib_lanczos.png')
    box = np.array(Image.open('lib_box.png')).astype(float)
    print('   макс розбіжність BOX і ds_area: %.1f' % np.abs(box - np.array(Image.open('ds_area.png')).astype(float)).max())
    cb8 = Image.fromarray(cb.astype(np.uint8))
    print('   BOX на шахівниці 1 піксель, середнє: %.1f' % np.array(cb8.resize((128, 128), Image.BOX)).mean())
