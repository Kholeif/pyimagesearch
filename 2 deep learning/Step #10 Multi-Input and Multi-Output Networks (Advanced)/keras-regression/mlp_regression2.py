from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from pyimagesearch import models
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import locale
import os

def load_house_attributes(inputPath):
	cols = ["bedrooms", "bathrooms", "area", "zipcode", "price"]
	df = pd.read_csv(inputPath, sep=" ", header=None, names=cols)
	zipcodes = df["zipcode"].value_counts().keys().tolist()
	counts = df["zipcode"].value_counts().tolist()
	for (zipcode, count) in zip(zipcodes, counts):
		if count < 25:
			idxs = df[df["zipcode"] == zipcode].index
			df.drop(idxs, inplace=True)
	return df

def process_house_attributes(df):
	continuous = ["bedrooms", "bathrooms", "area", "price"]
	cs = MinMaxScaler().fit(df[continuous])
	trainContinuous = cs.transform(df[continuous])

	zipBinarizer = LabelBinarizer().fit(df["zipcode"])
	trainCategorical = zipBinarizer.transform(df["zipcode"])
    
	df2 = np.hstack([trainCategorical, trainContinuous])

	return df2

print("[INFO] loading house attributes...")
inputPath = os.path.sep.join(["Houses-dataset/Houses Dataset/" , "HousesInfo.txt"])
df = load_house_attributes(inputPath)

print("[INFO] processing data...")
df2 = process_house_attributes(df)

x=df2[:,:-1]
y=df2[:,-1]

print("[INFO] constructing training/testing split...")
(trainX, testX, trainY, testY) = train_test_split(x, y, test_size=0.25, random_state=42)

model = models.create_mlp(trainX.shape[1], regress=True)
opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

print("[INFO] training model...")
model.fit(x=trainX, y=trainY, 
	validation_data=(testX, testY),
	epochs=300, batch_size=8)

print("[INFO] predicting house prices...")
preds = model.predict(testX)

diff = preds.flatten() - testY
percentDiff = (diff / testY) * 100
absPercentDiff = np.abs(percentDiff)

mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)

locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
print("[INFO] avg. house price: {}, std house price: {}".format(
	locale.currency(df["price"].mean(), grouping=True),
	locale.currency(df["price"].std(), grouping=True)))
print("[INFO] mean: {:.2f}%, std: {:.2f}%".format(mean, std))