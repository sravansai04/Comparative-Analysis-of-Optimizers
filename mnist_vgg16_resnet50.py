# -*- coding: utf-8 -*-
"""MNIST_VGG16_ResNet50.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1zK3rOQohuquLdt_utmq9JPPsOkyb6qGy
"""
import os
import tensorflow as tf
# tf.test.gpu_device_name()

import gdown
print("---------------------------------Started Downloading Dataset---------------------------------")

url = "https://drive.google.com/u/0/uc?id=1Y2dYLTfyQn5ua0TUmpLkmhSr-6V6y7BT&export=download"
output = "file_name.zip"

gdown.download(url, output, quiet=False)

import zipfile
with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall()
print("---------------------------------Completed Downloading Dataset---------------------------------")

import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import tensorflow.keras.layers as Layers
import tensorflow.keras.activations as Actications
import tensorflow.keras.models as Models
import tensorflow.keras.optimizers as Optimizer
import tensorflow.keras.utils as Utils
from tensorflow.keras.callbacks import ReduceLROnPlateau , EarlyStopping
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train = pd.read_csv("digit-recognizer/train.csv")
# test = pd.read_csv("test.csv")

train.head()

X = (train.drop(columns=['label'],axis=1).values / 255.0).reshape(-1,28,28)
y = train['label']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
X_train.shape , X_test.shape

#convert data to rgb
X_train=tf.image.grayscale_to_rgb(tf.expand_dims(X_train, axis=3)) 
X_test=tf.image.grayscale_to_rgb(tf.expand_dims(X_test, axis=3))

#resize to minimum size of (32x32)
X_train=tf.image.resize_with_pad(X_train,32,32)
X_test=tf.image.resize_with_pad(X_test,32,32)

vggmodel = VGG16(input_shape=(32,32,3),weights='imagenet', include_top=False)
#freeze the first 3 blocks
for layer in vggmodel.layers[:11]:
    layer.trainable = False

top_model = vggmodel.output
top_model = Layers.Flatten()(top_model)
top_model = Layers.Dense(512, activation='relu')(top_model)
top_model = Layers.Dropout(0.5)(top_model)
top_model = Layers.Dense(64, activation='relu')(top_model)
top_model = Layers.Dropout(0.2)(top_model)

output_layer = Layers.Dense(10, activation='softmax')(top_model)


print("\n")

if not os.path.exists("outputs/mnistDataset"):
    os.makedirs("outputs/mnistDataset")

if not os.path.exists("mnist_models"):
    os.makedirs("mnist_models")
    

"""# VGG16"""
print("---------------------------------Creating VGG16 Model---------------------------------")
vggmodel = Models.Model(inputs=vggmodel.input, outputs=output_layer)
vggmodel.summary()

"""## Adam"""
print("\n")
print("--------------------------------- VGG16 Model with Adam Optimizer---------------------------------")
beta1 = 0.9
beta2 = 0.99
epsilonE = 1e-06
adam1 = tf.keras.optimizers.Adam(
    learning_rate = 0.001,
    beta_1 = beta1,
    beta_2 = beta2,
    epsilon = epsilonE,
    amsgrad=False,
 
)

y_train=keras.utils.to_categorical(y_train,num_classes=10)

y_test=keras.utils.to_categorical(y_test,num_classes=10)

vggAdam = vggmodel
from keras import callbacks

vggAdam.compile(optimizer = adam1, loss = "categorical_crossentropy", metrics=["accuracy"])
earlystopping = callbacks.EarlyStopping(monitor ="val_loss", 
                                        mode ="min", patience = 3, 
                                        restore_best_weights = True)
vggadamhis = vggAdam.fit(X_train,y_train,validation_split=0.3,epochs=50,callbacks=[earlystopping],batch_size=128)
vggadamval=vggAdam.evaluate(X_test, y_test)
vggAdam.save("mnist_models/vggadam_model.h5")
print("VGG16 Model - Adam Optimizer Accuracy - ",vggadamval[1]*100 )



"""## AdaGrad"""
print("\n")
print("--------------------------------- VGG16 Model with Adagrad Optimizer---------------------------------")
# Adagrad
adagrad1 = tf.keras.optimizers.Adagrad(
    learning_rate = 0.001,
    initial_accumulator_value = 0.1,
    epsilon = epsilonE,
)
vggada = vggmodel
vggada.compile(optimizer = adagrad1, loss = "categorical_crossentropy", metrics=["accuracy"])
vggadahis = vggada.fit(X_train,y_train,validation_split=0.3,epochs=50,callbacks=[earlystopping],batch_size=128)
vggadaval=vggada.evaluate(X_test, y_test)
print("\n")
print("VGG16 Model - Adagrad Optimizer Accuracy - ",vggadaval[1]*100 )
vggadaval

vggada.save("mnist_models/vggada_model.h5")


"""## RMSProp"""
print("\n")
print("--------------------------------- VGG16 Model with RMSProp Optimizer---------------------------------")
# RMSProp
rmsprop1 = tf.keras.optimizers.RMSprop(
    learning_rate = 0.001,
    rho = 0.9,
    momentum = 0.0,
    epsilon = epsilonE,
    centered=False,
)

vggrms = vggmodel
vggrms.compile(optimizer = rmsprop1, loss = "categorical_crossentropy", metrics=["accuracy"])
vggrmshis = vggrms.fit(X_train,y_train,validation_split=0.3,epochs=50,callbacks=[earlystopping],batch_size=128)
vggrmsval=vggrms.evaluate(X_test, y_test)
print("\n")
print("VGG16 Model - RMSProp Optimizer Accuracy - ",vggrmsval[1]*100 )
vggrmsval
vggrms.save("mnist_models/vggrms_model.h5")


"""## SGD"""
print("\n")
print("--------------------------------- VGG16 Model with SGD Optimizer---------------------------------")
# Momentum
momentum1 = tf.keras.optimizers.SGD(
    learning_rate = 0.01,
    momentum = beta1,
    nesterov = False,
)

vggsgd = vggmodel
vggsgd.compile(optimizer = momentum1, loss = "categorical_crossentropy", metrics=["accuracy"])
vggsgdhis = vggsgd.fit(X_train,y_train,validation_split=0.3,epochs=50,callbacks=[earlystopping],batch_size=128)
vggsgdval=vggsgd.evaluate(X_test, y_test)
print("\n")
print("VGG16 Model - SGD Optimizer Accuracy - ",vggsgdval[1]*100 )
vggsgdval

vggsgd.save("mnist_models/vggsgd_model.h5")


"""## VGG16 Plots"""

import seaborn as sns
classifiers = [vggadamval , vggadaval ,vggrmsval ,vggsgdval]
cv_results = []
for i in classifiers :
    cv_results.append(i[1])
cv_res = pd.DataFrame({"Evaluate":cv_results,"Network":["Adam","Adagrad","RMSprop",
"SGD"]})

cv_res

# use seaborn to create a barplot
sns.set(style="whitegrid")
ax = sns.barplot(x="Network", y="Evaluate", data=cv_res)
ax.set_ylim([0, 1.0])
# add labels to the plot
ax.set_title("Evaluation Scores for Different Optimizers for VGG16")
ax.set_xlabel("Network")
ax.set_ylabel("Accuracy")

# display the plot
plt.savefig("outputs/mnistDataset/vgg16_optimizer_evaluation.png")
plt.show()

"""## VGG16 Loss Plots"""

# Plot training & validation loss values
plt.plot(vggadamhis.history['loss'])
plt.plot(vggadahis.history['loss'])
plt.plot(vggrmshis.history['loss'])
plt.plot(vggsgdhis.history['loss'])
plt.title('VGG16 Model Training Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Adam', 'Adagrad','RMSprop','SGD'], loc='upper right')
plt.savefig("outputs/mnistDataset/vgg16_training_loss.png")
plt.show()

# Plot training & validation loss values
plt.plot(vggadamhis.history['val_loss'])
plt.plot(vggadahis.history['val_loss'])
plt.plot(vggrmshis.history['val_loss'])
plt.plot(vggsgdhis.history['val_loss'])
plt.title('VGG16 Model Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Adam', 'Adagrad','RMSprop','SGD'], loc='upper right')
plt.savefig("outputs/mnistDataset/vgg16_validation_loss.png")
plt.show()

plt.plot(vggadamhis.history['accuracy'])
plt.plot(vggadahis.history['accuracy'])
plt.plot(vggrmshis.history['accuracy'])
plt.plot(vggsgdhis.history['accuracy'])
plt.title('VGG16 Model Training Accuracy')
plt.ylabel('accuracy')
plt.xlabel('Epoch')
plt.legend(['Adam', 'Adagrad','RMSprop','SGD'], loc='lower right')
plt.savefig('outputs/mnistDataset/vgg16_training_accuracy.png')
plt.show()

plt.plot(vggadamhis.history['val_accuracy'])
plt.plot(vggadahis.history['val_accuracy'])
plt.plot(vggrmshis.history['val_accuracy'])
plt.plot(vggsgdhis.history['val_accuracy'])
plt.title('VGG16 Model Validation Accuracy')
plt.ylabel('accuracy')
plt.xlabel('Epoch')
plt.legend(['Adam', 'Adagrad','RMSprop','SGD'], loc='lower right')
plt.savefig('outputs/mnistDataset/vgg16_validation_accuracy.png')
plt.show()

import seaborn as sns
import pandas as pd
optimizer =['Adam', 'Adagrad','RMSprop','SGD']
histories =[vggadamhis,vggadahis,vggrmshis,vggsgdhis]
acc_data=[]
for i in range(len(optimizer)):
    acc = histories[i].history['val_accuracy']
    acc_data.append(acc)
  
fig, ax = plt.subplots()
ax.boxplot(acc_data, labels=[str(i) for i in optimizer])
ax.set_ylabel('Accuracy')
ax.set_xlabel('Optimizers')
ax.set_title('VGG16 Model - Validation Accuracy by Optimizer')
plt.savefig('outputs/mnistDataset/vgg16_validation_accuracy_whiskerplot.png')
plt.show()



"""# Resnet50"""
print("\n")
print("--------------------------------- ResNet50 Model --------------------------------")
resmodel = tf.keras.applications.ResNet50(input_shape=(32,32,3),weights='imagenet', include_top=False)

resmodel.summary()

#freeze the first 3 blocks
for layer in resmodel.layers[:11]:
    layer.trainable = False

top_model = resmodel.output
top_model = Layers.Flatten()(top_model)
top_model = Layers.Dense(512, activation='relu')(top_model)
top_model = Layers.Dropout(0.5)(top_model)
top_model = Layers.Dense(64, activation='relu')(top_model)
top_model = Layers.Dropout(0.2)(top_model)

output_layer = Layers.Dense(10, activation='softmax')(top_model)

resmodel = Models.Model(inputs=resmodel.input, outputs=output_layer)
resmodel.summary()

"""## Adagrad"""
print("\n")
print("--------------------------------- ResNet50 Model with Adagrad Optimizer---------------------------------")
resada = resmodel
adagrad1 = tf.keras.optimizers.Adagrad(
    learning_rate = 0.001,
    initial_accumulator_value = 0.1,
    epsilon = epsilonE,
)
resada.compile(optimizer = adagrad1, loss = "categorical_crossentropy", metrics=["accuracy"])
resadahis = resada.fit(X_train,y_train,validation_split=0.3,epochs=50,callbacks=[earlystopping],batch_size=128)
resadaval=resada.evaluate(X_test, y_test)
print("\n")
print("ResNet50 Model - Adagrad Optimizer Accuracy - ", resadaval[1]*100)
resadaval


resada.save("mnist_models/resada_model.h5")


"""## Adam"""
print("\n")
print("--------------------------------- ResNet50 Model with Adam Optimizer---------------------------------")
adam1 = tf.keras.optimizers.Adam(
    learning_rate = 0.001,
    beta_1 = beta1,
    beta_2 = beta2,
    epsilon = epsilonE,
    amsgrad=False,
 
)

resadam = resmodel
resadam.compile(optimizer = adam1, loss = "categorical_crossentropy", metrics=["accuracy"])
resadamhis = resadam.fit(X_train,y_train,validation_split=0.3,epochs=50,callbacks=[earlystopping],batch_size=128)
resadamval=resadam.evaluate(X_test, y_test)

print(resadamval)



resadam.save("mnist_models/resadam_model.h5")


"""## RmsProp"""
print("\n")
print("--------------------------------- ResNet50 Model with RMSProp Optimizer---------------------------------")
# RMSProp
rmsprop1 = tf.keras.optimizers.RMSprop(
    learning_rate = 0.001,
    rho = 0.9,
    momentum = 0.0,
    epsilon = epsilonE,
    centered=False,
)

resrms = resmodel
resrms.compile(optimizer = rmsprop1, loss = "categorical_crossentropy", metrics=["accuracy"])
resrmshis = resrms.fit(X_train,y_train,validation_split=0.3,epochs=50,callbacks=[earlystopping],batch_size=128)
resrmsval=resrms.evaluate(X_test, y_test)
print("\n")
print("ResNet50 Model - RMSProp Optimizer Accuracy - ", resrmsval[1]*100)
resrmsval
resrms.save("mnist_models/resrms_model.h5")



"""## SGD"""
print("\n")
print("--------------------------------- ResNet50 Model with SGD Optimizer---------------------------------")
ressgd = resmodel
momentum1 = tf.keras.optimizers.SGD(
    learning_rate = 0.01,
    momentum = beta1,
    nesterov = False,
)

ressgd.compile(optimizer = momentum1, loss = "categorical_crossentropy", metrics=["accuracy"])
ressgdhis = ressgd.fit(X_train,y_train,validation_split=0.3,epochs=50,callbacks=[earlystopping],batch_size=128)
ressgdval=ressgd.evaluate(X_test, y_test)
print("\n")
print("ResNet50 Model - SGD Optimizer Accuracy - ", ressgdval[1]*100)
ressgdval
ressgd.save("mnist_models/ressgd_model.h5")



"""## ResNet50 Plots"""

import seaborn as sns
classifiers = [resadaval , resadamval ,resrmsval ,ressgdval]
cv_results_res = []
for i in classifiers :
    cv_results_res.append(i[1])
cv_results_res = pd.DataFrame({"Evaluate":cv_results_res,"Network":["Adam","Adagrad","RMSprop",
"SGD"]})

# # use seaborn to create a barplot
# sns.set(style="whitegrid")
# ax = sns.barplot(x="Network", y="Evaluate", data=cv_results_res)

# # add labels to the plot
# ax.set_title("Evaluation Scores for Different Optimizers for ResNet50")
# ax.set_xlabel("Optimizer")
# ax.set_ylabel("Accuracy")

# # display the plot
# plt.show()

# use seaborn to create a barplot
sns.set(style="whitegrid")
ax = sns.barplot(x="Network", y="Evaluate", data=cv_results_res)
ax.set_ylim([0, 1.0])

ax.set_title("Evaluation Scores for Different Optimizers for ResNet50")
ax.set_xlabel("Network")
ax.set_ylabel("Accuracy")
plt.savefig("outputs/mnistDataset/resnet50_evaluation_scores.png")
# display the plot
plt.show()

"""## ResNet Loss Plots"""

# Plot training & validation loss values
plt.plot(resadamhis.history['loss'])
plt.plot(resadahis.history['loss'])
plt.plot(resrmshis.history['loss'])
plt.plot(ressgdhis.history['loss']) 
plt.title('ResNet50 Model Training Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Adam', 'Adagrad','RMSprop','SGD'], loc='upper right')
plt.savefig("outputs/mnistDataset/resnet50_training_loss.png")
plt.show()

# Plot training & validation loss values
plt.plot(resadamhis.history['val_loss'])
plt.plot(resadahis.history['val_loss'])
plt.plot(resrmshis.history['val_loss'])
plt.plot(ressgdhis.history['val_loss'])
plt.title('ResNet50 Model Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Adam', 'Adagrad','RMSprop','SGD'], loc='upper right')
plt.savefig("outputs/mnistDataset/resnet50_validation_loss.png")
plt.show()

# Plot training & validation loss values
plt.plot(resadamhis.history['accuracy'])
plt.plot(resadahis.history['accuracy'])
plt.plot(resrmshis.history['accuracy'])
plt.plot(ressgdhis.history['accuracy'])
plt.title('ResNet50 Model Training Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Adam', 'Adagrad','RMSprop','SGD'], loc='lower right')
plt.savefig("outputs/mnistDataset/resnet50_training_accuracy.png")
plt.show()

# Plot training & validation loss values
plt.plot(resadamhis.history['val_accuracy'])
plt.plot(resadahis.history['val_accuracy'])
plt.plot(resrmshis.history['val_accuracy'])
plt.plot(ressgdhis.history['val_accuracy'])
plt.title('ResNet50 Model Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Adam', 'Adagrad','RMSprop','SGD'], loc='lower right')
plt.savefig("outputs/mnistDataset/resnet50_validation_accuracy.png")
plt.show()

import seaborn as sns
import pandas as pd
optimizer =['Adam', 'Adagrad','RMSprop','SGD']
histories =[resadamhis,resadahis,resrmshis,ressgdhis]
# Get the validation accuracies for each optimizer
acc_data = []
for i in range(len(optimizer)):
    acc = histories[i].history['val_accuracy']
    acc_data.append(pd.DataFrame({'Optimizer': str(optimizer[i]), 'Accuracy': acc}))

# Concatenate the dataframes and draw the density plot
acc_df = pd.concat(acc_data)
sns.kdeplot(data=acc_df, x='Accuracy', hue='Optimizer', fill=True)

acc_data=[]
for i in range(len(optimizer)):
    acc = histories[i].history['val_accuracy']
    acc_data.append(acc)
  
fig, ax = plt.subplots()
ax.boxplot(acc_data, labels=[str(i) for i in optimizer])
ax.set_ylabel('Accuracy')
ax.set_xlabel('Optimizers')
ax.set_title('ResNet50 Model - Validation Accuracy by Optimizer')
plt.savefig("outputs/mnistDataset/resnet50_validation_accuracy_whiskerplot.png")
plt.show()

