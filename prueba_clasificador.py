from features_classifier import StateFeatures
from PIL import Image as PImage
import numpy as np


def loadImages(name, path="./labels/"):
    img = PImage.open(path + name)
    return img
    #return ([x for x in img if x.uid >= 1e8],
            #[x for x in img if x.uid < 1e8])


img = loadImages(name="0_181455427_20.png")
#img.show()


clasificador = StateFeatures()
print(clasificador)
features = clasificador.features(obses=np.array(img))

# print(features)
#
