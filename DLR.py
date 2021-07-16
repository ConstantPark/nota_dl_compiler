import os
import numpy as np
from dlr import DLRModel

# Load the compiled model
input_shape = {'data': [1, 3, 224, 224]} # A single RGB 224x224 image
output_shape = [1, 1000]                 # The probability for each one of the 1,000 classes
device = 'cpu'                           # Go, Raspberry Pi, go!
model = DLRModel('resnet50', input_shape, output_shape, device)

# Load names for ImageNet classes
synset_path = os.path.join(model_path, 'synset.txt')
with open(synset_path, 'r') as f:
    synset = eval(f.read())

# Load an image stored as a numpy array
image = np.load('dog.npy').astype(np.float32)
print(image.shape)
input_data = {'data': image}

# Predict 
out = model.run(input_data)
top1 = np.argmax(out[0])
prob = np.max(out)
print("Class: %s, probability: %f" % (synset[top1], prob))