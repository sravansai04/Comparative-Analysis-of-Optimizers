import pkg_resources
import subprocess

# List of required packages
packages = ['gdown','pillow', 'matplotlib', 'tensorflow', 'numpy', 'pandas', 'scikit-learn', 'keras', 'opencv-python', 'tqdm', 'seaborn']

# Iterate through the packages and check if they are already installed
for package in packages:
    try:
        pkg_resources.get_distribution(package)
        print(f"{package} is already installed.")
    except pkg_resources.DistributionNotFound:
        # If the package is not found, install it
        print(f"{package} is not installed. Installing...")
        subprocess.call(['pip', 'install', package])


import gdown
import zipfile
import os


if not os.path.exists("outputs"):
    os.makedirs("outputs")



# with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
#     zip_ref.extractall(output_dir_path)

import subprocess
print("\n")
print("\n")
print("-------------------------Dataset 1 - Water Level Detection-------------------------------")
subprocess.run(["python","waterBotteles.py"])
print("\n")
print("\n")
print("-------------------------Dataset 2 - ChestXrays - Pneunomia Detection-------------------------------")
subprocess.run(["python","chestxrays.py"])
print("\n")
print("\n")
print("-------------------------Dataset 3 - Flower Classification -------------------------------")
print("\n")
print("\n")
subprocess.run(["python","flowers.py"])
print("-------------------------Dataset 4 - MNIST Dataset -------------------------------")
print("\n")
print("\n")
subprocess.run(["python","mnist_vgg16_resnet50.py"])
subprocess.run(["python","mnist_inceptionresnetv2.py"])
subprocess.run(["python","mnist_xception.py"])
print("\n")
print("\n")
print("-------------------------MNIST Manual Dataset -------------------------------")
print("\n")
print("\n")
subprocess.run(["python","mnistManual.py"])
print("\n")
print("\n")
print("<-------------------------  COMPLETED  ------------------------------->")