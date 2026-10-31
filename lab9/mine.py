from lab9 import *
im = np.array(Image.open("my.png").convert("RGB"))
h, w = im.shape[:2]
im = im[:h//4*4, :w//4*4]
out = bc1(im, "pixel")
print("RMSE: %.2f" % stats(im, out)[2], " макс: %.0f" % stats(im, out)[0])
save(out, "bc1_my.png")
save(np.abs(im.astype(float) - out) * 3, "diff_my.png")