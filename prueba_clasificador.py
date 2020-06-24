from features_classifier import StateFeatures, StateClassifier
from PIL import Image as PImage
import numpy as np
import torch


def loadImages(name, path="./labels/"):
    img = PImage.open(path + name)
    return img
    #return ([x for x in img if x.uid >= 1e8],
            #[x for x in img if x.uid < 1e8])


img = loadImages(name="0_181455427_20.png")
#img.show()


# clasificador = StateFeatures()
# print(clasificador)
# features = clasificador.features(obses=np.array(img))

device = torch.device('cuda')
clasificador = StateClassifier()
clasificador.load_state_dict(torch.load('save_classifier.pkl', map_location='cpu'))
clasificador.to(device)
#class_observation = np.array(observation)
class_observation = torch.from_numpy(np.array(img)[None]).to(device)
features = clasificador(class_observation).detach().cpu().numpy()[0]#Change the boolean array to a binary array (length 11)
features = features > 0
features = [0 if features[i]==False else 1 for i in range(len(features))]
print(features)

# print(features)
#
