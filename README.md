## Classifying fruits and vegetables via transfer learning

![alt text](https://raw.githubusercontent.com/weechien/fruits-360/master/fruits.jpg "Cover photo")


Today, we will explore on an existing image classifier - the Resnet18 model, and try to apply transfer learning to classify 131 types of fruits and vegetables.

Let’s cover this article briefly:

1. We will be using the [PyTorch](https://pytorch.org/) framework to train our model.
2. We will also be using [Kaggle](http://kaggle.com/) to write our code and train the model using a GPU. You will require an account for Kaggle.
3. The dataset will be from the [Fruit-Images-Dataset](https://github.com/Horea94/Fruit-Images-Dataset).
  * Total number of images is 90,483.
  * Total number of training images is 67,692 (one fruit or vegetable per image).
  * Total number of testing images is 22,688 (one fruit or vegetable per image).
4. You can get all the code and training steps from [Jovian](https://jovian.ml/weechien/assignment-5-fruits-360).
5. I will only use function calls here so as to avoid cluttering the page. You may refer to the link above for the function definitions.


We will be exploring the dataset to have a better understanding of it.


In the notebook, let's first clone the dataset from GitHub.
```markdown
!git clone https://github.com/Horea94/Fruit-Images-Dataset
```
![alt text](https://raw.githubusercontent.com/weechien/fruits-360/master/clone.JPG "Clone from GitHub")


Import all the required libraries.
```markdown
import os
import torch
import matplotlib.pyplot as plt

from torch import nn
from torch import optim
from os.path import join
from torchvision import models
from tqdm.notebook import tqdm
from torchvision import transforms
from torch.nn import functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset, DataLoader, random_split
```


Let's also check the type of device we are running.
Make sure to enable GPU on Kaggle.
```markdown
device = get_default_device()
print(device)
```
![alt text](https://raw.githubusercontent.com/weechien/fruits-360/master/device.JPG "Device type")


Next, we will import and load the datasets into pytorch.
We will be splitting the existing test dataset into 50% validation and 50% test.
```markdown 
# Dataset folders and loaders
DATA_DIR = join(os.getcwd(), 'Fruit-Images-Dataset')
TRAIN_DIR = join(DATA_DIR, 'Training')
TEST_DIR = join(DATA_DIR, 'Test')

data_transformer = transforms.Compose([transforms.ToTensor()])

test_dataset_split = ImageFolder(TEST_DIR, transform=data_transformer)
n = len(test_dataset_split)
n_validation = int(validation_pct_of_test * n)

train_dataset = ImageFolder(TRAIN_DIR, transform=data_transformer)
validation_dataset = Subset(test_dataset_split, range(n_validation))
test_dataset = Subset(test_dataset_split, range(n_validation, n))

train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=2, pin_memory=True)
validation_dataloader = DataLoader(validation_dataset, batch_size, shuffle=True, num_workers=2, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True, num_workers=2, pin_memory=True)

train_dataloader = DeviceDataLoader(train_dataloader, device)
validation_dataloader = DeviceDataLoader(validation_dataloader, device)
test_dataloader = DeviceDataLoader(test_dataloader, device)
```



The training dataset overview is shown below.
```markdown
# Information on the dataset

print('Number of training dataset:', len(train_dataset))
print('Number of validation dataset:', len(validation_dataset))
print('Number of testing dataset:', len(test_dataset))
print('Number of classes:',len(train_dataset.classes))
[print(f'{idx}： {cls}') for idx, cls in enumerate(train_dataset.classes)]
```
![alt text](https://raw.githubusercontent.com/weechien/fruits-360/master/dataset_preview.JPG "Dataset preview")
![alt text](https://raw.githubusercontent.com/weechien/fruits-360/master/classes.JPG "Dataset preview")


Let's take a peek at the individual training images
```markdown
# Information on a single data

images, labels = next(iter(train_dataloader))
plot_img(images, labels)
print('Image shape:', images[0].shape)
```
![alt text](https://raw.githubusercontent.com/weechien/fruits-360/master/training_images.JPG "Training images")

