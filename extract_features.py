from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.applications.vgg16 import preprocess_input
import numpy as np
import sys
import os
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
base_model = VGG16(weights='imagenet', include_top=True)
out = base_model.get_layer("fc2").output
model = Model(inputs=base_model.input, outputs=out)


def save_feature(save_path, feature):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print("[+]Save extracted feature to file : ", save_path)
    np.save(save_path, feature)


def extract_features(src):
    with open(src, "r") as file:
        for i, line in enumerate(file):
            img_path = line[:-1]
            print("[+] Read image  : ", img_path, " id : ", i)
            if os.path.isfile(img_path) and img_path.find(".jpg") != -1:
                save_path = img_path.replace("gender_dataset_face", "train_x")

                img = image.load_img(img_path, target_size=(224, 224))
                img_data = image.img_to_array(img)
                img_data = np.expand_dims(img_data, axis=0)
                img_data = preprocess_input(img_data)

                print("[+] Extract feature from image : ", img_path)
                feature = model.predict(img_data)

                print(img_data.shape)
                print(feature[0])

                #print(np.argmax(feature))
                save_feature(save_path, feature)

if __name__ == "__main__":
    src = sys.argv[1]
    print(src)
    extract_features(src)


