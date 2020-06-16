## Classifying fruits and vegetables via transfer learning

![alt text](https://raw.githubusercontent.com/weechien/fruits-360/master/fruits.jpg "Cover photo")

Today, we will explore on an existing image classifier - the Resnet18 model, and try to apply transfer learning to classify 131 types of fruits and vegetables. Let’s cover this article briefly:

1. We will be using the [PyTorch](https://pytorch.org/) framework to train our model.
2. We will also be using [Kaggle](http://kaggle.com/) to write our code and train the model using a GPU. You will require an account for Kaggle.
3. The dataset will be from the [Fruit-Images-Dataset](https://github.com/Horea94/Fruit-Images-Dataset).
  * Total number of images is 90,483.
  * Total number of training images is 67,692 (one fruit or vegetable per image).
  * Total number of testing images is 22,688 (one fruit or vegetable per image).
4. You can get all the code and training steps from this [link](https://jovian.ml/weechien/assignment-5-fruits-360).
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

The classes shown is rather long, so we will have to scroll down to view the entire list.
Here is the full list of classes:
**0**： Apple Braeburn  **1**： Apple Crimson Snow  **2**： Apple Golden 1  **3**： Apple Golden 2  **4**： Apple Golden 3
**5**： Apple Granny Smith  **6**： Apple Pink Lady  **7**： Apple Red 1  **8**： Apple Red 2  **9**： Apple Red 3
**10**： Apple Red Delicious  **11**： Apple Red Yellow 1  **12**： Apple Red Yellow 2  **13**： Apricot  **14**： Avocado
**15**： Avocado ripe  **16**： Banana  **17**： Banana Lady Finger  **18**： Banana Red  **19**： Beetroot
**20**： Blueberry  **21**： Cactus fruit  **22**： Cantaloupe 1  **23**： Cantaloupe 2  **24**： Carambula
**25**： Cauliflower  **26**： Cherry 1  **27**： Cherry 2  **28**： Cherry Rainier  **29**： Cherry Wax Black
**30**： Cherry Wax Red  **31**： Cherry Wax Yellow  **32**： Chestnut  **33**： Clementine  **34**： Cocos
**35**： Corn  **36**： Corn Husk  **37**： Cucumber Ripe  **38**： Cucumber Ripe 2  **39**： Dates
**40**： Eggplant  **41**： Fig  **42**： Ginger Root  **43**： Granadilla  **44**： Grape Blue
**45**： Grape Pink  **46**： Grape White  **47**： Grape White 2  **48**： Grape White 3  **49**： Grape White 4
**50**： Grapefruit Pink  **51**： Grapefruit White  **52**： Guava  **53**： Hazelnut  **54**： Huckleberry
**55**： Kaki  **56**： Kiwi  **57**： Kohlrabi  **58**： Kumquats  **59**： Lemon
**60**： Lemon Meyer  **61**： Limes  **62**： Lychee  **63**： Mandarine  **64**： Mango
**65**： Mango Red  **66**： Mangostan  **67**： Maracuja  **68**： Melon Piel de Sapo  **69**： Mulberry
**70**： Nectarine  **71**： Nectarine Flat  **72**： Nut Forest  **73**： Nut Pecan  **74**： Onion Red
**75**： Onion Red Peeled  **76**： Onion White  **77**： Orange  **78**： Papaya  **79**： Passion Fruit
**80**： Peach  **81**： Peach 2  **82**： Peach Flat  **83**： Pear  **84**： Pear 2
**85**： Pear Abate  **86**： Pear Forelle  **87**： Pear Kaiser  **88**： Pear Monster  **89**： Pear Red
**90**： Pear Stone  **91**： Pear Williams  **92**： Pepino  **93**： Pepper Green  **94**： Pepper Orange
**95**： Pepper Red  **96**： Pepper Yellow  **97**： Physalis  **98**： Physalis with Husk  **99**： Pineapple
**100**： Pineapple Mini  **101**： Pitahaya Red  **102**： Plum  **103**： Plum 2  **104**： Plum 3
**105**： Pomegranate  **106**： Pomelo Sweetie  **107**： Potato Red  **108**： Potato Red Washed  **109**： Potato Sweet
**110**： Potato White  **111**： Quince  **112**： Rambutan  **113**： Raspberry  **114**： Redcurrant
**115**： Salak  **116**： Strawberry  **117**： Strawberry Wedge  **118**： Tamarillo  **119**： Tangelo
**120**： Tomato 1  **121**： Tomato 2  **122**： Tomato 3  **123**： Tomato 4  **124**： Tomato Cherry Red
**125**： Tomato Heart  **126**： Tomato Maroon  **127**： Tomato Yellow  **128**： Tomato not Ripened  **129**： Walnut
**130**： Watermelon


Let's take a peek at the individual training images
```markdown
# Information on a single data

images, labels = next(iter(train_dataloader))
plot_img(images, labels)
print('Image shape:', images[0].shape)
```
![alt text](https://raw.githubusercontent.com/weechien/fruits-360/master/training_images.JPG "Training images")

