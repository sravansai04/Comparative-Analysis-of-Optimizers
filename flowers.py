# -*- coding: utf-8 -*-

import PIL.Image as Image
import matplotlib.pylab as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import os 
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,Model
from keras.layers import Dense,Conv2D, MaxPooling2D, BatchNormalization,Dropout, Flatten,Activation,concatenate,Input,AlphaDropout
from keras.optimizers import Adam
from keras.utils import to_categorical
import tensorflow as tf
import random as rn
import cv2                  
from tqdm import tqdm
from random import shuffle  
import itertools
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras import backend as K


if not os.path.exists("outputs/flowerDataset"):
    os.makedirs("outputs/flowerDataset")

if not os.path.exists("Dataset"):
    os.makedirs("Dataset")
import gdown

url = "https://drive.google.com/uc?id=1Gb-_LbLxBJyrblL5wDE1k_HTX5Y6ubTw"
output = "flowers.zip"
gdown.download(url,output, quiet=False)





import zipfile
with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall("Dataset/")
x=[]
y=[]
size=150
path='Dataset/flowers/'
for ftype in os.listdir(path):
    for img in tqdm(os.listdir(path+ftype)):
        label=ftype
        path1 = os.path.join(path+ftype,img)
        img = plt.imread(path1)
        img = cv2.resize(img, (size,size))
        x.append(np.array(img))
        y.append(str(label))

from tensorflow.keras.utils import to_categorical
encoder=LabelEncoder()
y=encoder.fit_transform(y)
y=to_categorical(y,5)
x=np.array(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

y_train

"""# VGG16"""
print("\n")
print("---------------------------------Creating VGG16 Model---------------------------------")

from tensorflow.keras.applications.vgg16 import VGG16
#from tensorflow.keras.applications.vgg16 import preprocess_input

base_model = VGG16(weights="imagenet", include_top=False, input_shape=x_train[0].shape)

## will not train base mode
base_model.trainable = False

base_model.summary()

"""## Adam"""
print("\n")
print("--------------------------------- VGG16 Model with Adam Optimizer---------------------------------")

#add our layers on top of this model
from tensorflow.keras import layers, models

flatten_layer = layers.Flatten()
dense_layer_1 = layers.Dense(50, activation='relu')
dense_layer_2 = layers.Dense(20, activation='relu')
prediction_layer = layers.Dense(5, activation='softmax')


model = models.Sequential([
    base_model,
    flatten_layer,
    dense_layer_1,
    dense_layer_2,
    prediction_layer
])

from tensorflow.keras.callbacks import EarlyStopping


beta1 = 0.9
beta2 = 0.99
epsilonE = 1e-06
vggadam=model
adam1 = tf.keras.optimizers.Adam(
    learning_rate = 0.001,
    beta_1 = beta1,
    beta_2 = beta2,
    epsilon = epsilonE,
    amsgrad=False,
 
)
res=vggadam.compile(
    optimizer=adam1,
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

es = EarlyStopping(monitor='val_accuracy', mode='max', patience=5,  restore_best_weights=True)

historyvgg = vggadam.fit(x_train, y_train, epochs=50, validation_split=0.2, batch_size=32, callbacks=[es])

vggadamval=vggadam.evaluate(x_test,y_test)
print("\n")
print("VGG16 Model - Adam Optimizer Accuracy - ",vggadamval[1]*100)


"""## Adagrad"""
print("\n")
print("--------------------------------- VGG16 Model with Adagrad Optimizer---------------------------------")

vggada=model
beta1 = 0.9
beta2 = 0.99
epsilonE = 1e-06
adagrad1 = tf.keras.optimizers.Adagrad(
    learning_rate = 0.001,
    initial_accumulator_value = 0.1,
    epsilon = epsilonE,
)

vggada.compile(
    optimizer=adagrad1,
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)
historyada = vggada.fit(x_train, y_train, epochs=50, validation_split=0.2, batch_size=32, callbacks=[es])

vggadaval=vggada.evaluate(x_test,y_test)
print("\n")
print("VGG16 Model - Adagrad Optimizer Accuracy - ",vggadaval[1]*100)


"""## RMSprop"""
print("\n")
print("--------------------------------- VGG16 Model with RMSProp Optimizer---------------------------------")

# RMSProp
incrms = model
rmsprop1 = tf.keras.optimizers.RMSprop(
    learning_rate = 0.001,
    rho = 0.9,
    momentum = 0.0,
    epsilon = epsilonE,
    centered=False,
)
incrms.compile(
    optimizer=rmsprop1,
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)
historyrms = incrms.fit(x_train, y_train, epochs=50, validation_split=0.2, batch_size=32, callbacks=[es])

incrmsval=incrms.evaluate(x_test, y_test)
print("\n")
print("VGG16 Model - RMSProp Optimizer Accuracy - ",incrmsval[1]*100)


"""## SGD"""
print("\n")
print("--------------------------------- VGG16 Model with SGD Optimizer---------------------------------")

ressgd = model
momentum1 = tf.keras.optimizers.SGD(
    learning_rate = 0.01,
    momentum = beta1,
    nesterov = False,
)

ressgd.compile(
    optimizer=momentum1,
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)
historysgd = ressgd.fit(x_train, y_train, epochs=50, validation_split=0.2, batch_size=32, callbacks=[es])

ressgdval=ressgd.evaluate(x_test, y_test)
print("\n")
print("VGG16 Model - SGD Optimizer Accuracy - ",ressgdval[1]*100)


"""## Plots"""

import seaborn as sns
classifiers = [vggadamval , vggadaval ,incrmsval ,ressgdval]
cv_results_res = []
for i in classifiers :
    cv_results_res.append(i[1])
cv_results_res = pd.DataFrame({"Evaluate":cv_results_res,"Network":["resAdam","resAdagrad","resRMS",
"resSGD"]})

cv_results_res

# use seaborn to create a barplot
sns.set(style="whitegrid")
ax = sns.barplot(x="Network", y="Evaluate", data=cv_results_res)

# add labels to the plot
ax.set_title("Evaluation Scores for Different Optimizers for VGG16")
ax.set_xlabel("Network")
ax.set_ylabel("Accuracy")
plt.savefig("outputs/flowerDataset/vgg16_optimizer_evaluation.png")
# display the plot
plt.show()

# Plot training & validation loss values
plt.plot(historyvgg.history['loss'])
plt.plot(historyada.history['loss'])
plt.plot(historyrms.history['loss'])
plt.plot(historysgd.history['loss'])
plt.title('VGG16 Model Training Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Adam', 'Adagrad','RMSprop','SGD'], loc='upper right')
plt.savefig("outputs/flowerDataset/vgg16_training_loss.png")
plt.show()

# Plot training & validation loss values
plt.plot(historyvgg.history['val_loss'])
plt.plot(historyada.history['val_loss'])
plt.plot(historyrms.history['val_loss'])
plt.plot(historysgd.history['val_loss'])
plt.title('VGG16 Model Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Adam', 'Adagrad','RMSprop','SGD'], loc='upper right')
plt.savefig("outputs/flowerDataset/vgg16_validation_loss.png")
plt.show()

# Plot training & validation loss values
plt.plot(historyvgg.history['accuracy'])
plt.plot(historyada.history['accuracy'])
plt.plot(historyrms.history['accuracy'])
plt.plot(historysgd.history['accuracy'])
plt.title('VGG16 Model Training Accuracy')
plt.ylabel('accuracy')
plt.xlabel('Epoch')
plt.legend(['Adam', 'Adagrad','RMSprop','SGD'], loc='lower right')
plt.savefig('outputs/flowerDataset/vgg16_training_accuracy.png')
plt.show()

# Plot training & validation loss values
plt.plot(historyvgg.history['val_accuracy'])
plt.plot(historyada.history['val_accuracy'])
plt.plot(historyrms.history['val_accuracy'])
plt.plot(historysgd.history['val_accuracy'])
plt.title('VGG16 Model Validation Accuracy')
plt.ylabel('accuracy')
plt.xlabel('Epoch')
plt.legend(['Adam', 'Adagrad','RMSprop','SGD'], loc='lower right')
plt.savefig('outputs/flowerDataset/vgg16_validation_accuracy.png')
plt.show()

import seaborn as sns
import pandas as pd
optimizer =['Adam', 'Adagrad','RMSprop','SGD']
histories =[historyvgg,historyada,historyrms,historysgd]
acc_data=[]
for i in range(len(optimizer)):
    acc = histories[i].history['val_accuracy']
    acc_data.append(acc)
  
fig, ax = plt.subplots()
ax.boxplot(acc_data, labels=[str(i) for i in optimizer])
ax.set_ylabel('Accuracy')
ax.set_xlabel('Optimizers')
ax.set_title('VGG16 Model - Validation Accuracy by Optimizer')
plt.savefig('outputs/flowerDataset/vgg16_validation_accuracy_whiskerplot.png')
plt.show()

"""# ResNet50"""
print("\n")
print("--------------------------------- ResNet50 Model --------------------------------")
resmodel = tf.keras.applications.ResNet50(input_shape=(150,150,3),weights='imagenet', include_top=False)

resmodel.trainable = False

from tensorflow.keras import layers, models

flatten_layer = layers.Flatten()
dense_layer_1 = layers.Dense(50, activation='relu')
dense_layer_2 = layers.Dense(20, activation='relu')
prediction_layer = layers.Dense(5, activation='softmax')


model = models.Sequential([
    resmodel,
    flatten_layer,
    dense_layer_1,
    dense_layer_2,
    prediction_layer
])

beta1 = 0.9
beta2 = 0.99
epsilonE = 1e-06

model.summary()

"""## Adam"""
print("\n")
print("--------------------------------- ResNet50 Model with Adam Optimizer---------------------------------")
from tensorflow.keras.callbacks import EarlyStopping

resadam=model
adam1 = tf.keras.optimizers.Adam(
    learning_rate = 0.001,
    beta_1 = beta1,
    beta_2 = beta2,
    epsilon = epsilonE,
    amsgrad=False,
 
)
resad=resadam.compile(
    optimizer=adam1,
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

es = EarlyStopping(monitor='val_accuracy', mode='max', patience=5,  restore_best_weights=True)

historyres = resadam.fit(x_train, y_train, epochs=50, validation_split=0.2, batch_size=32, callbacks=[es])

resadamval=resadam.evaluate(x_test,y_test)
print("\n")
print("ResNet50 Model - Adam Optimizer Accuracy - ", resadamval[1]*100)
resadamval

"""## Adagrad"""
print("\n")
print("--------------------------------- ResNet50 Model with Adagrad Optimizer---------------------------------")
resada=model
adagrad1 = tf.keras.optimizers.Adagrad(
    learning_rate = 0.001,
    initial_accumulator_value = 0.1,
    epsilon = epsilonE,
)

resada.compile(
    optimizer=adagrad1,
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)
historyadares = resada.fit(x_train, y_train, epochs=50, validation_split=0.2, batch_size=32, callbacks=[es])

resadaval=resada.evaluate(x_test,y_test)
print("\n")
print("ResNet50 Model - Adagrad Optimizer Accuracy - ", resadaval[1]*100)
resadaval

"""## RMSProp"""
print("\n")
print("--------------------------------- ResNet50 Model with RMSProp Optimizer---------------------------------")
# RMSProp
resrms = model
rmsprop1 = tf.keras.optimizers.RMSprop(
    learning_rate = 0.001,
    rho = 0.9,
    momentum = 0.0,
    epsilon = epsilonE,
    centered=False,
)
resrms.compile(
    optimizer=rmsprop1,
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)
historyrmsres = resrms.fit(x_train, y_train, epochs=50, validation_split=0.2, batch_size=32, callbacks=[es])

resrmsval=resrms.evaluate(x_test, y_test)
print("\n")
print("ResNet50 Model - RMSProp Optimizer Accuracy - ", resrmsval[1]*100)
resrmsval

"""## SGD"""
print("\n")
print("--------------------------------- ResNet50 Model with SGD Optimizer---------------------------------")
ressgdres = model
momentum1 = tf.keras.optimizers.SGD(
    learning_rate = 0.01,
    momentum = beta1,
    nesterov = False,
)

ressgdres.compile(
    optimizer=momentum1,
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)
historysgdres = ressgdres.fit(x_train, y_train, epochs=50, validation_split=0.2, batch_size=32, callbacks=[es])

ressgdresval=ressgdres.evaluate(x_test, y_test)
print("\n")
print("ResNet50 Model - SGD Optimizer Accuracy - ", ressgdresval[1]*100)
ressgdresval

"""## Plots"""

import seaborn as sns
classifiers = [resadamval , resadaval ,resrmsval ,ressgdresval]
cv_results_res = []
for i in classifiers :
    cv_results_res.append(i[1])
cv_results_res = pd.DataFrame({"Evaluate":cv_results_res,"Network":["resAdam","resAdagrad","resRMS",
"resSGD"]})

cv_results_res

# use seaborn to create a barplot
sns.set(style="whitegrid")
ax = sns.barplot(x="Network", y="Evaluate", data=cv_results_res)

# add labels to the plot
ax.set_title("Evaluation Scores for Different Optimizers for ResNet50")
ax.set_xlabel("Network")
ax.set_ylabel("Accuracy")
plt.savefig("outputs/flowerDataset/resnet50_evaluation_scores.png")
# display the plot
plt.show()



"""### Loss Plots"""

# Plot training & validation loss values
plt.plot(historyres.history['loss'])
plt.plot(historyadares.history['loss'])
plt.plot(historyrmsres.history['loss'])
plt.plot(historysgdres.history['loss'])
plt.title('ResNet50 Model Training Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Adam', 'Adagrad','RMSprop','SGD'], loc='upper right')
plt.savefig("outputs/flowerDataset/resnet50_training_loss.png")
plt.show()

# Plot training & validation loss values
plt.plot(historyres.history['val_loss'])
plt.plot(historyadares.history['val_loss'])
plt.plot(historyrmsres.history['val_loss'])
plt.plot(historysgdres.history['val_loss'])
plt.title('ResNet50 Model Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Adam', 'Adagrad','RMSprop','SGD'], loc='upper right')
plt.savefig("outputs/flowerDataset/resnet50_validation_loss.png")

plt.show()

# Plot training & validation loss values
plt.plot(historyres.history['accuracy'])
plt.plot(historyadares.history['accuracy'])
plt.plot(historyrmsres.history['accuracy'])
plt.plot(historysgdres.history['accuracy'])
plt.title('ResNet50 Model Training Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Adam', 'Adagrad','RMSprop','SGD'], loc='lower right')
plt.savefig("outputs/flowerDataset/resnet50_training_accuracy.png")
plt.show()

# Plot training & validation loss values
plt.plot(historyres.history['val_accuracy'])
plt.plot(historyadares.history['val_accuracy'])
plt.plot(historyrmsres.history['val_accuracy'])
plt.plot(historysgdres.history['val_accuracy'])
plt.title('ResNet50 Model Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Adam', 'Adagrad','RMSprop','SGD'], loc='lower right')
plt.savefig("outputs/flowerDataset/resnet50_validation_accuracy.png")
plt.show()

import seaborn as sns
import pandas as pd
optimizer =['Adam', 'Adagrad','RMSprop','SGD']
histories =[historyres,historyadares,historyrmsres,historysgdres]
acc_data=[]
for i in range(len(optimizer)):
    acc = histories[i].history['val_accuracy']
    acc_data.append(acc)
  
fig, ax = plt.subplots()
ax.boxplot(acc_data, labels=[str(i) for i in optimizer])
ax.set_ylabel('Accuracy')
ax.set_xlabel('Optimizers')
ax.set_title('ResNet50 Model - Validation Accuracy by Optimizer')
plt.savefig("outputs/flowerDataset/resnet50_validation_accuracy_whiskerplot.png")
plt.show()

"""# InceptionV3"""
print("\n")
print("--------------------------------- InceptionResNetV2 Model ---------------------------------")
inceptionModel = tf.keras.applications.InceptionResNetV2(input_shape=(150,150,3),weights='imagenet', include_top=False)

inceptionModel.trainable = False

from tensorflow.keras import layers, models

flatten_layer = layers.Flatten()
dense_layer_1 = layers.Dense(50, activation='relu')
dense_layer_2 = layers.Dense(20, activation='relu')
prediction_layer = layers.Dense(5, activation='softmax')


model_incept = models.Sequential([
    inceptionModel,
    flatten_layer,
    dense_layer_1,
    dense_layer_2,
    prediction_layer
])

model_incept.summary()

"""## Adam"""
print("\n")
print("--------------------------------- InceptionResNetV2 Model with Adam Optimizer---------------------------------")
from tensorflow.keras.callbacks import EarlyStopping

incepadam=model_incept
adam1 = tf.keras.optimizers.Adam(
    learning_rate = 0.001,
    beta_1 = beta1,
    beta_2 = beta2,
    epsilon = epsilonE,
    amsgrad=False,
 
)
incepada=incepadam.compile(
    optimizer=adam1,
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)
es = EarlyStopping(monitor='val_accuracy', mode='max', patience=5,  restore_best_weights=True)
historyincepadam = incepadam.fit(x_train, y_train, epochs=50, validation_split=0.2, batch_size=32, callbacks=[es])

incepadamval=incepadam.evaluate(x_test,y_test)
print("\n")
print("InceptionResNetV2 Model - Adam Optimizer Accuracy - ", incepadamval[1]*100)
incepadamval

"""## Adagrad"""
print("\n")
print("--------------------------------- InceptionResNetV2 Model with Adagrad Optimizer---------------------------------")
incepada=model_incept
adagrad1 = tf.keras.optimizers.Adagrad(
    learning_rate = 0.001,
    initial_accumulator_value = 0.1,
    epsilon = epsilonE,
)

incepada.compile(
    optimizer=adagrad1,
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)
historyadares = incepada.fit(x_train, y_train, epochs=50, validation_split=0.2, batch_size=32, callbacks=[es])

incepadaval=incepada.evaluate(x_test,y_test)
print("\n")
print("InceptionResNetV2 Model - Adagrad Optimizer Accuracy - ", incepadaval[1]*100)
incepadaval

"""## RMSProp"""
print("\n")
print("--------------------------------- InceptionResNetV2 Model with RMSProp Optimizer---------------------------------")
# RMSProp
inceprms = model_incept
rmsprop1 = tf.keras.optimizers.RMSprop(
    learning_rate = 0.001,
    rho = 0.9,
    momentum = 0.0,
    epsilon = epsilonE,
    centered=False,
)
inceprms.compile(
    optimizer=rmsprop1,
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)
historyinceprms= inceprms.fit(x_train, y_train, epochs=50, validation_split=0.2, batch_size=32, callbacks=[es])

resincepval=inceprms.evaluate(x_test, y_test)
print("\n")
print("InceptionResNetV2 Model - RMSProp Optimizer Accuracy - ", resincepval[1]*100)
resincepval

"""## SGD"""
print("\n")
print("--------------------------------- InceptionResNetV2 Model with SGD Optimizer---------------------------------")

incepsgd = model_incept
momentum1 = tf.keras.optimizers.SGD(
    learning_rate = 0.01,
    momentum = beta1,
    nesterov = False,
)

incepsgd.compile(
    optimizer=momentum1,
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)
historyincepsgd = incepsgd.fit(x_train, y_train, epochs=50, validation_split=0.2, batch_size=32, callbacks=[es])

incepsgdval=incepsgd.evaluate(x_test, y_test)
print("\n")
print("InceptionResNetV2 Model - SGD Optimizer Accuracy - ", incepsgdval[1]*100)
incepsgdval

"""## Plots"""



import seaborn as sns
classifiers = [incepadamval , incepadaval ,resincepval ,incepsgdval]
cv_results_res = []
for i in classifiers :
    cv_results_res.append(i[1])
cv_results_res = pd.DataFrame({"Evaluate":cv_results_res,"Network":["resAdam","resAdagrad","resRMS",
"resSGD"]})

cv_results_res

# use seaborn to create a barplot
sns.set(style="whitegrid")
ax = sns.barplot(x="Network", y="Evaluate", data=cv_results_res)

# add labels to the plot
ax.set_title("Evaluation Scores for Different Optimizers for InceptionResNetV2")
ax.set_xlabel("Network")
ax.set_ylabel("Accuracy")

# display the plot
plt.savefig("outputs/flowerDataset/inceptionresnet50_evaluation_scores.png")
plt.show()

"""### Loss Plots"""

# Plot training & validation loss values
plt.plot(historyincepadam.history['loss'])
plt.plot(historyadares.history['loss'])
plt.plot(historyinceprms.history['loss'])
plt.plot(historyincepsgd.history['loss'])
plt.title('InceptionResNetV2 Model Training loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Adam', 'Adagrad','RMSprop','SGD'], loc='upper right')
plt.savefig("outputs/flowerDataset/inceptionresnet50_training_loss.png")
plt.show()

# Plot training & validation loss values
plt.plot(historyincepadam.history['val_loss'])
plt.plot(historyadares.history['val_loss'])
plt.plot(historyinceprms.history['val_loss'])
plt.plot(historyincepsgd.history['val_loss'])
plt.title('InceptionResNetV2 Model Validation loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Adam', 'Adagrad','RMSprop','SGD'], loc='upper right')
plt.savefig("outputs/flowerDataset/inceptionresnet50_validation_loss.png")
plt.show()

# Plot training & validation loss values
plt.plot(historyincepadam.history['accuracy'])
plt.plot(historyadares.history['accuracy'])
plt.plot(historyinceprms.history['accuracy'])
plt.plot(historyincepsgd.history['accuracy'])
plt.title('InceptionResNetV2 Model Training Accuracy')
plt.ylabel('accuracy')
plt.xlabel('Epoch')
plt.legend(['Adam', 'Adagrad','RMSprop','SGD'], loc='lower right')
plt.savefig("outputs/flowerDataset/inceptionresnet50_training_accuracy.png")
plt.show()

# Plot training & validation loss values
plt.plot(historyincepadam.history['val_accuracy'])
plt.plot(historyadares.history['val_accuracy'])
plt.plot(historyinceprms.history['val_accuracy'])
plt.plot(historyincepsgd.history['val_accuracy'])
plt.title('InceptionResNetV2 Model Validation Accuracy')
plt.ylabel('accuracy')
plt.xlabel('Epoch')
plt.legend(['Adam', 'Adagrad','RMSprop','SGD'], loc='lower right')
plt.savefig("outputs/flowerDataset/inceptionresnet50_validation_accuracy.png")
plt.show()

import seaborn as sns
import pandas as pd
optimizer =['Adam', 'Adagrad','RMSprop','SGD']
histories =[historyincepadam,historyadares,historyinceprms,historyincepsgd]
acc_data=[]
for i in range(len(optimizer)):
    acc = histories[i].history['val_accuracy']
    acc_data.append(acc)
  
fig, ax = plt.subplots()
ax.boxplot(acc_data, labels=[str(i) for i in optimizer])
ax.set_ylabel('Accuracy')
ax.set_xlabel('Optimizers')
ax.set_title('InceptionResNetV2 Model - Validation Accuracy by Optimizer')
plt.savefig("outputs/flowerDataset/inceptionresnet50_val_acc_whiskerplot.png")
plt.show()

"""# Xception"""
print("\n")
print("--------------------------------- Xception Model ---------------------------------")
xceptionModel = tf.keras.applications.Xception(input_shape=(150,150,3),weights='imagenet', include_top=False)

xceptionModel.trainable = False

from tensorflow.keras import layers, models

flatten_layer = layers.Flatten()
dense_layer_1 = layers.Dense(50, activation='relu')
dense_layer_2 = layers.Dense(20, activation='relu')
prediction_layer = layers.Dense(5, activation='softmax')


model_xception = models.Sequential([
    xceptionModel,
    flatten_layer,
    dense_layer_1,
    dense_layer_2,
    prediction_layer
])

model_xception.summary()

"""## Adam"""
print("\n")
print("--------------------------------- Xception Model with Adam Optimizer---------------------------------")
xcepadam=model_xception
adam1 = tf.keras.optimizers.Adam(
    learning_rate = 0.001,
    beta_1 = beta1,
    beta_2 = beta2,
    epsilon = epsilonE,
    amsgrad=False,
 
)
xcepadam.compile(
    optimizer=adam1,
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)
es = EarlyStopping(monitor='val_accuracy', mode='max', patience=5,  restore_best_weights=True)
historyxcepadam = xcepadam.fit(x_train, y_train, epochs=50, validation_split=0.2, batch_size=32, callbacks=[es])

xcepadamval=xcepadam.evaluate(x_test,y_test)
print("\n")
print("Xception Model - Adam Optimizer Accuracy - ", xcepadamval[1]*100)
xcepadamval

"""## Adagrad"""
print("\n")
print("--------------------------------- Xception Model with Adagrad Optimizer---------------------------------")
xcepada=model_xception
adagrad1 = tf.keras.optimizers.Adagrad(
    learning_rate = 0.001,
    initial_accumulator_value = 0.1,
    epsilon = epsilonE,
)

xcepada.compile(
    optimizer=adagrad1,
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)
historyxcepada = xcepada.fit(x_train, y_train, epochs=50, validation_split=0.2, batch_size=32, callbacks=[es])

xcepadaval=xcepada.evaluate(x_test,y_test)
print("\n")
print("Xception Model - Adagrad Optimizer Accuracy - ", xcepadaval[1]*100)
xcepadaval

"""## RMSProp"""
print("\n")
print("--------------------------------- Xception Model with RMSProp Optimizer---------------------------------")
# RMSProp
xceprms = model_xception
rmsprop1 = tf.keras.optimizers.RMSprop(
    learning_rate = 0.001,
    rho = 0.9,
    momentum = 0.0,
    epsilon = epsilonE,
    centered=False,
)
xceprms.compile(
    optimizer=rmsprop1,
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)
historyxceprms= xceprms.fit(x_train, y_train, epochs=50, validation_split=0.2, batch_size=32, callbacks=[es])

xceprmsval=xceprms.evaluate(x_test,y_test)
print("\n")
print("Xception Model - RMSProp Optimizer Accuracy - ", xceprmsval[1]*100)
xceprmsval

"""## SGD"""
print("\n")
print("--------------------------------- Xception Model with SGD Optimizer---------------------------------")
xcepsgd = model_xception
momentum1 = tf.keras.optimizers.SGD(
    learning_rate = 0.01,
    momentum = beta1,
    nesterov = False,
)

xcepsgd.compile(
    optimizer=momentum1,
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)
historyxcepsgd = xcepsgd.fit(x_train, y_train, epochs=50, validation_split=0.2, batch_size=32, callbacks=[es])

xcepsgdval=xcepsgd.evaluate(x_test,y_test)
print("\n")
print("Xception Model - SGD Optimizer Accuracy - ", xcepsgdval[1]*100)
xcepsgdval

"""## Plots"""

import seaborn as sns
classifiers = [xcepadamval , xcepadaval ,xceprmsval ,xcepsgdval]
cv_results_res = []
for i in classifiers :
    cv_results_res.append(i[1])
cv_results_res = pd.DataFrame({"Evaluate":cv_results_res,"Network":["resAdam","resAdagrad","resRMS",
"resSGD"]})

cv_results_res

# use seaborn to create a barplot
sns.set(style="whitegrid")
ax = sns.barplot(x="Network", y="Evaluate", data=cv_results_res)

# add labels to the plot
ax.set_title("Evaluation Scores for Different Optimizers for Xception")
ax.set_xlabel("Network")
ax.set_ylabel("Accuracy")

# display the plot
plt.savefig("outputs/flowerDataset/xception_optimizer_evaluation.png")
plt.show()

"""### Loss Plots"""

# Plot training & validation loss values
plt.plot(historyxcepadam.history['loss'])
plt.plot(historyxcepada.history['loss'])
plt.plot(historyxceprms.history['loss'])
plt.plot(historyxcepsgd.history['loss'])
plt.title('Xception Model Training Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Adam', 'Adagrad','RMSprop','SGD'], loc='upper right')
plt.savefig("outputs/flowerDataset/xception_training_loss.png")
plt.show()

# Plot training & validation loss values
plt.plot(historyxcepadam.history['val_loss'])
plt.plot(historyxcepada.history['val_loss'])
plt.plot(historyxceprms.history['val_loss'])
plt.plot(historyxcepsgd.history['val_loss'])
plt.title('Xception Model Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Adam', 'Adagrad','RMSprop','SGD'], loc='upper right')
plt.savefig("outputs/flowerDataset/xception_validation_loss.png")
plt.show()

# Plot training & validation loss values
plt.plot(historyxcepadam.history['accuracy'])
plt.plot(historyxcepada.history['accuracy'])
plt.plot(historyxceprms.history['accuracy'])
plt.plot(historyxcepsgd.history['accuracy'])
plt.title('Xception Model Training Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Adam', 'Adagrad','RMSprop','SGD'], loc='lower right')
plt.savefig("outputs/flowerDataset/xception_training_accuracy.png")
plt.show()

# Plot training & validation loss values
plt.plot(historyxcepadam.history['val_accuracy'])
plt.plot(historyxcepada.history['val_accuracy'])
plt.plot(historyxceprms.history['val_accuracy'])
plt.plot(historyxcepsgd.history['val_accuracy'])
plt.title('Xception Model Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Adam', 'Adagrad','RMSprop','SGD'], loc='lower right')
plt.savefig("outputs/flowerDataset/xception_validation_accuracy.png")
plt.show()

import seaborn as sns
import pandas as pd
optimizer =['Adam', 'Adagrad','RMSprop','SGD']
histories =[historyxcepadam,historyxcepada,historyxceprms,historyxcepsgd]
acc_data=[]
for i in range(len(optimizer)):
    acc = histories[i].history['val_accuracy']
    acc_data.append(acc)
  
fig, ax = plt.subplots()
ax.boxplot(acc_data, labels=[str(i) for i in optimizer])
x.set_ylabel('Accuracy')
ax.set_xlabel('Optimizers')
ax.set_title('Xception Model - Validation Accuracy by Optimizer')
plt.savefig("outputs/flowerDataset/xception_val_acc_whiskerplot.png")
plt.show()

