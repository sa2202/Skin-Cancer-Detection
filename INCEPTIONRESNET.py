#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical  # convert to one-hot-encoding

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt


# In[2]:


skin_df =pd.read_parquet(r"C:\Users\Aman Sethia\Downloads\output_92_92.parquet", engine='pyarrow')


# In[3]:


class_mapping = {
    'Melanocytic_nevi': 4,
    'melanoma': 6,
    'Benign_keratosis-like_lesions': 2,
    'Basal_cell_carcinoma': 1,
    'Vascular_lesions': 5,
    'Actinic_keratoses': 0,
    'Dermatofibroma': 3,
    4: 4, 6: 6, 2: 2, 1: 1, 5: 5, 0: 0, 3: 3
}

# Replace values in the "label" column
skin_df['label'] = skin_df['label'].replace(class_mapping)


# In[4]:


skin_df.head()


# In[5]:


X = skin_df.drop("label", axis=1).values
label = skin_df["label"].values
X.shape, label.shape


# In[6]:


skin_df["label"].value_counts()


# In[7]:


classes = {
    4: ('nv', 'melanocytic nevi', 0),  # 0 for benign
    6: ('mel', 'melanoma', 1),         # 1 for malignant
    2: ('bkl', 'benign keratosis-like lesions', 0),
    1: ('bcc', 'basal cell carcinoma', 1),
    5: ('vasc', 'pyogenic granulomas and hemorrhage', 0),
    0: ('akiec', 'Actinic keratoses and intraepithelial carcinomae', 1),
    3: ('df', 'dermatofibroma', 0)
}


# In[8]:





# In[9]:





# In[53]:


X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.25, random_state=1)


# In[54]:


X_train = X_train.reshape(X_train.shape[0], *(92, 92, 3))

X_test = X_test.reshape(X_test.shape[0], *(92, 92, 3))


# In[55]:


X_train.shape, X_val.shape, X_test.shape


# In[56]:


y_train = to_categorical(y_train)

y_test = to_categorical(y_test)


# In[57]:


y_train.shape, y_val.shape


# In[58]:


from keras.applications.inception_resnet_v2 import InceptionResNetV2


# In[59]:


import keras.backend as K


# In[60]:


pre_trained_model = InceptionResNetV2(input_shape=(92, 92, 3), include_top=False, weights="imagenet")


# In[61]:


for layer in pre_trained_model.layers:
    print(layer.name)
    if hasattr(layer, 'moving_mean') and hasattr(layer, 'moving_variance'):
        layer.trainable = True
        K.eval(K.update(layer.moving_mean, K.zeros_like(layer.moving_mean)))
        K.eval(K.update(layer.moving_variance, K.zeros_like(layer.moving_variance)))
    else:
        layer.trainable = False

print(len(pre_trained_model.layers))


# In[62]:


last_layer = pre_trained_model.get_layer('conv_7b_ac')
print('last layer output shape:', last_layer.output_shape)
last_output = last_layer.output


# In[63]:


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
optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=True)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])


# In[65]:


history = model.fit(X_train ,
                    y_train ,
                    epochs=3 ,
                    batch_size=128,
                    validation_data=(X_test , y_test) ,
                    )


# In[66]:


pre_trained_model.layers[617].name


# In[67]:


for layer in pre_trained_model.layers[618:]:
    layer.trainable = True


# In[68]:


optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-7,amsgrad=False)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['acc'])


# In[69]:


from keras.callbacks import ReduceLROnPlateau

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy'
                                            , patience = 2
                                            , verbose=1
                                            ,factor=0.5
                                            , min_lr=0.00001)


# In[70]:


model.summary()


# In[71]:


history = model.fit(X_train ,
                    y_train ,
                    epochs=25 ,
                    batch_size=128,
                    validation_data=(X_test , y_test) ,
                    callbacks=[learning_rate_reduction])


# In[72]:


model.save("resinc_model.keras")


# In[73]:


from tensorflow.keras.models import load_model


# In[74]:


cnn_model = load_model('resinc_model.keras')


# In[75]:


import joblib


# In[76]:


joblib.dump(cnn_model, 'resnic1_skin.joblib')


# In[77]:


from sklearn.ensemble import RandomForestClassifier


# In[78]:


loaded_cnn_model = joblib.load('resnic1_skin.joblib')


# In[79]:


X_train_features = loaded_cnn_model.predict(X_train)
X_test_features = loaded_cnn_model.predict(X_test)


# In[80]:


X_train_features_flat = X_train_features.reshape(X_train_features.shape[0], -1)
X_test_features_flat = X_test_features.reshape(X_test_features.shape[0], -1)


# In[81]:


from sklearn.ensemble import RandomForestClassifier


# In[82]:


rf_classifier = RandomForestClassifier(n_estimators=100, random_state=0)
rf_classifier.fit(X_train_features_flat, y_train)


# In[83]:


y_pred = rf_classifier.predict(X_test_features_flat)
y_pred


# In[84]:


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


# In[85]:


train_score =model.evaluate(X_train, y_train, verbose= 1)
test_score = model.evaluate(X_test, y_test, verbose= 1)

print("Train Loss: ", train_score[0])
print("Train Accuracy: ", train_score[1])
print('-' * 20)
print("Test Loss: ", test_score[0])
print("Test Accuracy: ", test_score[1])


# In[86]:


y_true = np.array(y_test)

y_pred = np.argmax(y_pred , axis=1)
y_true = np.argmax(y_true , axis=1)


# In[87]:


from sklearn.metrics import accuracy_score, classification_report


# In[88]:


accuracy = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred)


# In[89]:


print("Accuracy:", accuracy)
print("Classification Report:\n", report)


# In[90]:


classes_labels = []
for key in classes.keys():
    classes_labels.append(key)

print(classes_labels)


# In[91]:


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools
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


# In[54]:


class_names = [classes[i][0] for i in range(len(classes))]
num_images_to_visualize = 100
random_indices = np.random.choice(len(X_test), num_images_to_visualize, replace=False)

correct_predictions = 0

for idx in random_indices:
    img = X_test[idx]
    true_label, true_danger = classes[y_true[idx]][0], classes[y_true[idx]][2]
    pred_label, pred_danger = classes[y_pred[idx]][0], classes[y_pred[idx]][2]

    plt.imshow(img)
    plt.title(f"True: {true_label} ({'Benign' if true_danger == 0 else 'Malignant'})\n"
              f"Predicted: {pred_label} ({'Benign' if pred_danger == 0 else 'Malignant'})")
    plt.show()

    # Check if the prediction is correct
    if true_label == pred_label:
        correct_predictions += 1

accuracy = correct_predictions / num_images_to_visualize
print(f"Accuracy on visualized images: {accuracy * 100:.2f}%")
print(f"Correct Predictions: {correct_predictions}")


# In[92]:


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


# In[93]:


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


# In[94]:


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




