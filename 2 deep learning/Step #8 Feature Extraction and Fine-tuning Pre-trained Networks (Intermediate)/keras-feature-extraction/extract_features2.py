# USAGE
# python extract_features.py

# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from pyimagesearch import config
from imutils import paths
import numpy as np
import pickle
import random
import os

# load the ResNet50 network and initialize the label encoder
print("[INFO] loading network...")
model = ResNet50(weights="imagenet", include_top=False)
le = None

# loop over the data splits
for split in (config.TRAIN , config.TEST , config.VAL):
        # grab all image paths in the current split
        print("[INFO] processing '{} split'...".format(split))
        p = os.path.sep.join([config.BASE_PATH, split])
        imagePaths = list(paths.list_images(p))
        # randomly shuffle the image paths and then extract the class
        # labels from the file paths
        random.shuffle(imagePaths)
        labels = [p.split(os.path.sep)[-2] for p in imagePaths]
        labels.append("food")
        # if the label encoder is None, create it
        if le is None:
                le = LabelEncoder()
                le.fit(labels)
        # open the output CSV file for writing
        csvPath = os.path.sep.join([config.BASE_CSV_PATH,"{}2.csv".format(split)])
        csv = open(csvPath, "w")
        # loop over the images in batches
        for i,imagePath in enumerate(imagePaths):
                print("[INFO] processing image {}/{}".format(i + 1,len(imagePaths)))
                image = load_img(imagePath, target_size=(224, 224))
                image = img_to_array(image)
                image = np.expand_dims(image, axis=0)
                image = preprocess_input(image)
                features = model.predict(image)
                features = features.reshape((7 * 7 * 2048))
                label = le.transform(labels[i:i+1])
                label = label[0]
                vec = ",".join([str(v) for v in features])
                csv.write("{},{}\n".format(label, vec))
        print("done")
        csv.close()

# serialize the label encoder to disk
f = open("output/le2.cpickle", "wb")
f.write(pickle.dumps(le))
f.close()
