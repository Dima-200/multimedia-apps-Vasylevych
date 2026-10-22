import os
from lab6 import *

img2 = load_gray("my.png")
os.makedirs("second", exist_ok=True)
os.chdir("second")     
save_gray(img2, "original.png")
build_curve(img2)