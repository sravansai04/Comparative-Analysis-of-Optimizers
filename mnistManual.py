
from PIL import Image
import numpy as np
import tensorflow as tf
actualV=[]
predV=[]
def predictDigit(path, model,name):
 # print(type(name))
  # Load the image
  image = Image.open(path)
  # Convert the image to grayscale
  image = image.convert('L')
  # Resize the image to 32x32 pixels
  image = image.resize((32, 32))
  image_array = np.array(image)
  image_array = tf.image.grayscale_to_rgb(tf.expand_dims(image_array, axis=2))
  image_array = tf.expand_dims(image_array, axis=0)
  hi=model.predict(image_array)
  # Get the predicted class label
  predicted_class = np.argmax(hi)
  # Print the input label and the predicted label
  # print("Input label: ", name)
  # print("Predicted label: ", predicted_class)
  actualV.append(name)
  predV.append(predicted_class)

import os
hf={"7":7,"1":1,"2":2,"3":3,"4":4,"5":5,"6":6,"8":8,"9":9}
p="images/images/"
a=os.listdir(p)
a = [x for x in a if x != ".DS_Store"]
print(a[0])
modelPath = "mnist_models/"
allModels = os.listdir(modelPath)
print(allModels)
# Load the saved model
from keras.models import load_model
for i in allModels:
  ji=i.split('.')[0]
  ji=ji.split("_")[0]
  print(f"------------------------------Prediction using {ji} ----------------------------")
  loaded_model = load_model(modelPath+i)
  print("Loaded Model Successfully")
  for i in range(len(a)):
    nam=a[i].split(".")[0]
    nam=nam.split("_")[-1]
    predictDigit(p+a[i], loaded_model,nam)
#  print(actualV)
#  print(predV)

  num_correct = 0
  for i in range(len(actualV)):
      if int(actualV[i]) == int(predV[i]):
          num_correct += 1

  # Calculate the accuracy as a percentage
  accuracy = num_correct / len(actualV) * 100

  # Print the accuracy
  print(f"{ji}","   Accuracy: {:.2f}%".format(accuracy))

