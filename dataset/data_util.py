from PIL import Image
import numpy as np


def pil_load_img(path):
    #print(path)
    image = Image.open(path).convert("RGB")
    image = np.array(image)
    return image
