import pandas as pd
import numpy as np
import os
import pickle

from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv("./data/processed/train_processed.csv")

x_train = train_data.drop(columns=['Potability'],axis=1)
y_train = train_data['Potability']

clf = RandomForestClassifier()
clf.fit(x_train,y_train)

pickle.dump(clf,open("model.pkl","wb"))

