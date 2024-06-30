#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from tensorflow.keras.utils import to_categorical
# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

print ('modules loaded')


# In[2]:


skin_df =pd.read_csv(r"C:\Users\Aman Sethia\Desktop\dataverse_files\Project_dataset\hmnist_32_32_RGB.csv")
skin_df.head()


# In[3]:


Label = skin_df["label"]
Data = skin_df.drop(columns=["label"])


# In[4]:


skin_df["label"].value_counts()


# In[5]:


from imblearn.over_sampling import RandomOverSampler

# Assuming Data is a DataFrame
oversample = RandomOverSampler()
Data_array, Label = oversample.fit_resample(Data.to_numpy().reshape(-1, 32*32*3), Label)

# Reshape the array back to the original shape
Data = Data_array.reshape(-1, 32, 32, 3)
print('Shape of Data:', Data.shape)


# In[6]:


Label = np.array(Label)
Label


# In[8]:


classes = {
    4: ('nv', 'melanocytic nevi', 0),  # 0 for benign
    6: ('mel', 'melanoma', 1),         # 1 for malignant
    2: ('bkl', 'benign keratosis-like lesions', 0),
    1: ('bcc', 'basal cell carcinoma', 1),
    5: ('vasc', 'pyogenic granulomas and hemorrhage', 0),
    0: ('akiec', 'Actinic keratoses', 1),
    3: ('df', 'dermatofibroma', 0)
}


# In[9]:


from sklearn.model_selection import train_test_split


# In[10]:


X_train , X_test , y_train , y_test = train_test_split(Data , Label , test_size = 0.25 , random_state = 49)


# In[11]:


print("Number of images for training:", X_train.shape[0])
print("Number of images for testing:", X_test.shape[0])


# In[15]:


from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# In[16]:


from keras.applications.densenet import DenseNet201


# In[17]:


pre_trained_model = DenseNet201(input_shape=(32, 32, 3), include_top=False, weights="imagenet")


# In[18]:


for layer in pre_trained_model.layers:
    print(layer.name)
    layer.trainable = False
    
print(len(pre_trained_model.layers))


# In[19]:


last_layer = pre_trained_model.get_layer('relu')
print('last layer output shape:', last_layer.output_shape)
last_output = last_layer.output


# In[23]:


from keras import layers

from keras import Model
# Flatten the output layer to 1 dimension
x = layers.GlobalMaxPooling2D()(last_output)
# Add a fully connected layer with 512 hidden units and ReLU activation
x = layers.Dense(512, activation='relu')(x)
# Add a dropout rate of 0.7
x = layers.Dropout(0.5)(x)
# Add a final sigmoid layer for classification
x = layers.Dense(7, activation='softmax')(x)

# Configure and compile the model

model = Model(pre_trained_model.input, x)
optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-7,amsgrad=True)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])


# In[24]:


model.summary()


# In[27]:


from keras.callbacks import ReduceLROnPlateau

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy'
                                            , patience = 2
                                            , verbose=1
                                            ,factor=0.5
                                            , min_lr=0.00001)


# In[28]:


history = model.fit(X_train ,
                    y_train ,
                    epochs=3 ,
                    batch_size=128,
                    validation_data=(X_test , y_test) ,
                    callbacks=[learning_rate_reduction]
)


# In[29]:


for layer in pre_trained_model.layers:
    layer.trainable = True


# In[30]:


optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-7,amsgrad=True)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['acc'])


# In[31]:


from keras.callbacks import ReduceLROnPlateau

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy'
                                            , patience = 2
                                            , verbose=1
                                            ,factor=0.5
                                            , min_lr=0.00001)


# In[32]:


history = model.fit(X_train ,
                    y_train ,
                    epochs=25 ,
                    batch_size=128,
                    validation_data=(X_test , y_test) ,
                    callbacks=[learning_rate_reduction]
)


# In[33]:


model.save("densenet_model.keras")


# In[34]:


from tensorflow.keras.models import load_model

# Load the pre-trained model
cnn_model = load_model('densenet_model.keras')


# In[35]:


import joblib


# In[36]:


joblib.dump(cnn_model, 'cnn_model_skin.joblib')


# In[37]:


from sklearn.ensemble import RandomForestClassifier


# In[38]:


loaded_cnn_model = joblib.load('cnn_model_skin.joblib')


# In[39]:


X_train_features = loaded_cnn_model.predict(X_train)
X_test_features = loaded_cnn_model.predict(X_test)


# In[40]:


X_train_features_flat = X_train_features.reshape(X_train_features.shape[0], -1)
X_test_features_flat = X_test_features.reshape(X_test_features.shape[0], -1)


# In[41]:


from sklearn.ensemble import RandomForestClassifier


# In[42]:


rf_classifier = RandomForestClassifier(n_estimators=100, random_state=0)
rf_classifier.fit(X_train_features_flat, y_train)


# In[58]:


y_pred = rf_classifier.predict(X_test_features_flat)
y_pred


# In[60]:


def plot_training(hist):
    tr_acc = hist.history['acc']
    tr_loss = hist.history['loss']
    val_acc = hist.history['val_acc']
    val_loss = hist.history['val_loss']
    index_loss = np.argmin(val_loss)
    val_lowest = val_loss[index_loss]
    index_acc = np.argmax(val_acc)
    acc_highest = val_acc[index_acc]

    plt.figure(figsize= (20, 8))
    plt.style.use('fivethirtyeight')
    Epochs = [i+1 for i in range(len(tr_acc))]
    loss_label = f'best epoch= {str(index_loss + 1)}'
    acc_label = f'best epoch= {str(index_acc + 1)}'

    plt.subplot(1, 2, 1)
    plt.plot(Epochs, tr_loss, 'r', label= 'Training loss')
    plt.plot(Epochs, val_loss, 'g', label= 'Validation loss')
    plt.scatter(index_loss + 1, val_lowest, s= 150, c= 'blue', label= loss_label)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(Epochs, tr_acc, 'r', label= 'Training Accuracy')
    plt.plot(Epochs, val_acc, 'g', label= 'Validation Accuracy')
    plt.scatter(index_acc + 1 , acc_highest, s= 150, c= 'blue', label= acc_label)
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout
    plt.show()


# In[61]:


plot_training(history)


# In[56]:


train_score = model.evaluate(X_train, y_train, verbose= 1)
test_score = model.evaluate(X_test, y_test, verbose= 1)

print("Train Loss: ", train_score[0])
print("Train Accuracy: ", train_score[1])
print('-' * 20)
print("Test Loss: ", test_score[0])
print("Test Accuracy: ", test_score[1])


# In[62]:


y_true = np.array(y_test)

y_pred = np.argmax(y_pred , axis=1)
y_true = np.argmax(y_true , axis=1)


# In[63]:


from sklearn.metrics import accuracy_score, classification_report


# In[64]:


accuracy = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", report)


# In[65]:


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


# In[66]:


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


# In[67]:


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


# In[ ]:




