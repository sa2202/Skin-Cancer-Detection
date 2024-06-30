# -*- coding: utf-8 -*-
"""BEST_32_32

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/14pvAUmgD2UYH0rhbK0Ozo66bXfW3Gti_
"""

import pandas as pd

from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

skin_df =pd.read_csv('/content/drive/MyDrive/hmnist_32_32_RGB.csv')
skin_df.head()

import os
import time
import shutil
import itertools

# import data handling tools
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt
# import Deep learning Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras import regularizers
from tensorflow.keras.metrics import categorical_crossentropy

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

print ('modules loaded')

Label = skin_df["label"]
Data = skin_df.drop(columns=["label"])

skin_df["label"].value_counts()

from imblearn.over_sampling import RandomOverSampler

# Assuming Data is a DataFrame
oversample = RandomOverSampler()
Data_array, Label = oversample.fit_resample(Data.to_numpy().reshape(-1, 32*32*3), Label)

# Reshape the array back to the original shape
Data = Data_array.reshape(-1, 32, 32, 3)
print('Shape of Data:', Data.shape)

Label = np.array(Label)
Label

classes = {
    4: ('nv', 'melanocytic nevi', 0),  # 0 for benign
    6: ('mel', 'melanoma', 1),         # 1 for malignant
    2: ('bkl', 'benign keratosis-like lesions', 0),
    1: ('bcc', 'basal cell carcinoma', 1),
    5: ('vasc', 'pyogenic granulomas and hemorrhage', 0),
    0: ('akiec', 'Actinic keratoses and intraepithelial carcinomae', 1),
    3: ('df', 'dermatofibroma', 0)
}

from sklearn.model_selection import train_test_split

X_train , X_test , y_train , y_test = train_test_split(Data , Label , test_size = 0.25 , random_state = 49)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

datagen = ImageDataGenerator(rescale=(1./255)
                             ,rotation_range=10
                             ,zoom_range = 0.1
                             ,width_shift_range=0.1
                             ,height_shift_range=0.1)

testgen = ImageDataGenerator(rescale=(1./255))

from keras.callbacks import ReduceLROnPlateau

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy'
                                            , patience = 2
                                            , verbose=1
                                            ,factor=0.5
                                            , min_lr=0.00001)

model = keras.models.Sequential()

# Create Model Structure
model.add(keras.layers.Input(shape=[32, 32, 3]))
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(keras.layers.MaxPooling2D())
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(keras.layers.MaxPooling2D())
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(keras.layers.MaxPooling2D())
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(keras.layers.MaxPooling2D())

model.add(keras.layers.Flatten())
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.Dense(units=256, activation='relu', kernel_initializer='he_normal'))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Dense(units=128, activation='relu', kernel_initializer='he_normal'))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Dense(units=64, activation='relu', kernel_initializer='he_normal'))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Dense(units=32, activation='relu', kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.L1L2()))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Dense(units=7, activation='softmax', kernel_initializer='glorot_uniform', name='classifier'))
model.compile(Adamax(learning_rate= 0.001), loss= 'categorical_crossentropy', metrics= ['accuracy'])

model.summary()

tf.keras.utils.plot_model(model, show_shapes = True, show_dtype = True, show_layer_names = True, rankdir="TB", expand_nested = True, dpi = 100 ,to_file='model_best.png')

!pip install visualkeras
import visualkeras0

from PIL import ImageFont
font = ImageFont.load_default()
visualkeras.layered_view(model, legend=True, font=font,to_file='output.png')

history = model.fit(X_train ,
                    y_train ,
                    epochs=25 ,
                    batch_size=128,
                    validation_data=(X_test , y_test) ,
                    callbacks=[learning_rate_reduction])





from tensorflow.keras.models import load_model

# Load the pre-trained model
cnn_model = load_model('best_model_64.keras')

import joblib

joblib.dump(cnn_model, 'cnn_model_skin.joblib')

from sklearn.ensemble import RandomForestClassifier

loaded_cnn_model = joblib.load('cnn_model_skin.joblib')

X_train_features = loaded_cnn_model.predict(X_train)
X_test_features = loaded_cnn_model.predict(X_test)

X_train_features_flat = X_train_features.reshape(X_train_features.shape[0], -1)
X_test_features_flat = X_test_features.reshape(X_test_features.shape[0], -1)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=0)
rf_classifier.fit(X_train_features_flat, y_train)

y_pred = rf_classifier.predict(X_test_features_flat)
y_pred

train_score = cnn_model.evaluate(X_train, y_train, verbose= 1)
test_score = cnn_model.evaluate(X_test, y_test, verbose= 1)

print("Train Loss: ", train_score[0])
print("Train Accuracy: ", train_score[1])
print('-' * 20)
print("Test Loss: ", test_score[0])
print("Test Accuracy: ", test_score[1])

y_true = np.array(y_test)

y_pred = np.argmax(y_pred , axis=1)
y_true = np.argmax(y_true , axis=1)

from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# Assuming X_train_features and X_test_features are your extracted features
# and y_train, y_test are your corresponding labels

# Initialize the Decision Tree classifier
tree_classifier = DecisionTreeClassifier()

# Train the classifier
tree_classifier.fit(X_train_features, y_train)

# Make predictions
y_pred_tree = tree_classifier.predict(X_test_features)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred_tree)
print("Accuracy:", accuracy)

# Generate classification report
report = classification_report(y_test, y_pred_tree)
print("Classification Report:\n", report)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# Initialize the KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)

# Train the classifier
knn_classifier.fit(X_train_features, y_train)

# Make predictions
y_pred_knn = knn_classifier.predict(X_test_features)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred_knn)
print("Accuracy:", accuracy)

# Generate classification report
report = classification_report(y_test, y_pred_knn)
print("Classification Report:\n", report)

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# Assuming X_train_features and X_test_features are your extracted features
# and y_train, y_test are your corresponding multi-labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train_features, y_train, test_size=0.2, random_state=42)

# Initialize the base classifier
base_classifier = ExtraTreesClassifier(n_estimators=100, random_state=0)

# Initialize the MultiOutputClassifier
multi_output_classifier = MultiOutputClassifier(base_classifier)

# Train the classifier
multi_output_classifier.fit(X_train, y_train)

# Make predictions
y_pred_multi_output = multi_output_classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred_multi_output)
print("Accuracy:", accuracy)

# Generate classification report
report = classification_report(y_test, y_pred_multi_output)
print("Classification Report:\n", report)

classes_labels = []
for key in classes.keys():
    classes_labels.append(key)

print(classes_labels)

cm = cm = confusion_matrix(y_true, y_pred, labels=classes_labels)

plt.figure(figsize= (10, 10))
plt.imshow(cm, interpolation= 'nearest', cmap= plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation= 45)
plt.yticks(tick_marks, classes)


thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j], horizontalalignment= 'center', color= 'white' if cm[i, j] > thresh else 'black')

plt.tight_layout()
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

plt.show()

import matplotlib.pyplot as plt

# Define the training history dictionary
history = {
    'loss': [1.1349, 0.4640, 0.2622, 0.1833, 0.1375, 0.1030, 0.0813, 0.0704, 0.0533, 0.0208, 0.0094, 0.0082, 0.0076, 0.0067, 0.0038, 0.0028, 0.0022, 0.0013, 0.0016, 0.0012, 0.0011, 0.000952, 0.0010, 0.0010, 0.0008942],
    'val_loss': [0.9485, 0.4387, 0.2512, 0.2144, 0.1336, 0.1434, 0.1100, 0.2377, 0.1177, 0.0533, 0.0659, 0.0457, 0.0549, 0.0773, 0.0546, 0.0572, 0.0510, 0.0521, 0.0495, 0.0506, 0.0504, 0.0516, 0.0506, 0.0500, 0.0525],
    'accuracy': [0.5947, 0.8463, 0.9137, 0.9395, 0.9541, 0.9654, 0.9727, 0.9759, 0.9820, 0.9938, 0.9976, 0.9979, 0.9981, 0.9981, 0.9993, 0.9994, 0.9997, 0.9999, 0.9998, 0.9999, 0.9999, 0.9999, 0.9999, 0.9999, 0.9999],
    'val_accuracy': [0.6360, 0.8539, 0.9151, 0.9169, 0.9568, 0.9490, 0.9627, 0.9254, 0.9588, 0.9842, 0.9824, 0.9882, 0.9853, 0.9821, 0.9866, 0.9868, 0.9881, 0.9867, 0.9884, 0.9876, 0.9883, 0.9878, 0.9886, 0.9882, 0.9876]
}

# Extract relevant metrics from the history dictionary
tr_loss = history['loss']
val_loss = history['val_loss']
tr_acc = history['accuracy']
val_acc = history['val_accuracy']
epochs = range(1, len(tr_loss) + 1)

# Plotting
plt.figure(figsize=(15, 6))

# Plotting Training and Validation Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, tr_loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'g', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plotting Training and Validation Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, tr_acc, 'r', label='Training Accuracy')
plt.plot(epochs, val_acc, 'g', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()