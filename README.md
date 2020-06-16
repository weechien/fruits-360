## Classifying fruits and vegetables via transfer learning

![alt text](https://raw.githubusercontent.com/weechien/fruits-360/master/fruits.jpg "Cover photo")

<br />
Today, we will explore on an existing image classifier - the Resnet18 model, and try to apply transfer learning to classify 131 types of fruits and vegetables.

<br /><br />
Let’s cover this article briefly:
<br />

1. We will be using the [PyTorch](https://pytorch.org/) framework to train our model.
2. We will also be using [Kaggle](http://kaggle.com/) to write our code and train the model using a GPU.<br />
You will require an account for Kaggle.
3. The dataset will be from the [Fruit-Images-Dataset](https://github.com/Horea94/Fruit-Images-Dataset).
    * Total number of images is 90,483.
    * Total number of training images is 67,692 (one fruit or vegetable per image).
    * Total number of testing images is 22,688 (one fruit or vegetable per image).
4. You can get all the code and training steps from [Jovian](https://jovian.ml/weechien/assignment-5-fruits-360).
5. I will only use function calls here so as to avoid cluttering the page.<br />
You may refer to the link above for the function definitions.

<br />
We will be exploring the dataset to have a better understanding of it.
<br /><br />

In the notebook, let's first clone the dataset from GitHub.
```markdown
!git clone https://github.com/Horea94/Fruit-Images-Dataset
```
![alt text](https://raw.githubusercontent.com/weechien/fruits-360/master/clone.JPG "Clone from GitHub")
<br />

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
<br />

Let's also check the type of device we are running.<br />
Make sure to enable GPU on Kaggle.
```markdown
device = get_default_device()
print(device)
```
![alt text](https://raw.githubusercontent.com/weechien/fruits-360/master/device.JPG "Device type")
<br />

Next, we will import and load the datasets into pytorch.<br />
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
<br />

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
<br /><br />
![alt text](https://raw.githubusercontent.com/weechien/fruits-360/master/classes.JPG "Dataset classes")
<br /><br />

Let's take a peek at some of the training images.
```markdown
# Information on a single data

images, labels = next(iter(train_dataloader))
plot_img(images, labels)
print('Image shape:', images[0].shape)
```
![alt text](https://raw.githubusercontent.com/weechien/fruits-360/master/training_images.JPG "Training images")
<br />

Initialize the model, setup the hyperparameters, then start training
```markdown
# Set hyperparams, initialize model, then start training

num_epochs = 25
batch_size = 32
learning_rate = 1e-4
momentum = .9
opt_func = torch.optim.SGD

model = to_device(FruitModel(), device)

history = fit(num_epochs, learning_rate, model, train_dataloader, validation_dataloader, opt_func)
```
<br />
We are training our resnet-18 model for 25 epochs with a batch size of 32, and feeding it with the fruits and vegetables images.<br />
As the epochs progresses, the loss gradually goes down and accuracy goes up.<br />
Validation accuracy remains stable, which indicates that the model has yet to overfit over the 25 epochs.<br /><br />

<em>Epoch [0], train_loss: 4.6832, val_loss: 4.5394, train_acc: 25.00%, val_acc: 45.75%</em><br />
<em>Epoch [1], train_loss: 4.4252, val_loss: 4.3475, train_acc: 83.33%, val_acc: 80.10%</em><br />
<em>Epoch [2], train_loss: 4.2717, val_loss: 4.2281, train_acc: 83.33%, val_acc: 88.16%</em><br />
<em>Epoch [3], train_loss: 4.1752, val_loss: 4.1462, train_acc: 100.00%, val_acc: 91.95%</em><br />
<em>Epoch [4], train_loss: 4.1144, val_loss: 4.0996, train_acc: 100.00%, val_acc: 92.58%</em><br />
<em>Epoch [5], train_loss: 4.0740, val_loss: 4.0661, train_acc: 83.33%, val_acc: 94.09%</em><br />
<em>Epoch [6], train_loss: 4.0457, val_loss: 4.0409, train_acc: 100.00%, val_acc: 94.41%</em><br />
<em>Epoch [7], train_loss: 4.0248, val_loss: 4.0232, train_acc: 100.00%, val_acc: 95.17%</em><br />
<em>Epoch [8], train_loss: 4.0088, val_loss: 4.0109, train_acc: 91.67%, val_acc: 95.69%</em><br />
<em>Epoch [9], train_loss: 3.9958, val_loss: 4.0013, train_acc: 91.67%, val_acc: 95.76%</em><br />
<em>Epoch [10], train_loss: 3.9855, val_loss: 3.9911, train_acc: 100.00%, val_acc: 95.98%</em><br />
<em>Epoch [11], train_loss: 3.9771, val_loss: 3.9843, train_acc: 91.67%, val_acc: 95.81%</em><br />
<em>Epoch [12], train_loss: 3.9700, val_loss: 3.9790, train_acc: 91.67%, val_acc: 96.63%</em><br />
<em>Epoch [13], train_loss: 3.9641, val_loss: 3.9713, train_acc: 100.00%, val_acc: 96.59%</em><br />
<em>Epoch [14], train_loss: 3.9586, val_loss: 3.9679, train_acc: 100.00%, val_acc: 96.94%</em><br />
<em>Epoch [15], train_loss: 3.9540, val_loss: 3.9634, train_acc: 100.00%, val_acc: 96.99%</em><br />
<em>Epoch [16], train_loss: 3.9500, val_loss: 3.9594, train_acc: 100.00%, val_acc: 96.88%</em><br />
<em>Epoch [17], train_loss: 3.9466, val_loss: 3.9566, train_acc: 100.00%, val_acc: 97.05%</em><br />
<em>Epoch [18], train_loss: 3.9433, val_loss: 3.9536, train_acc: 100.00%, val_acc: 97.40%</em><br />
<em>Epoch [19], train_loss: 3.9404, val_loss: 3.9503, train_acc: 100.00%, val_acc: 97.56%</em><br />
<em>Epoch [20], train_loss: 3.9378, val_loss: 3.9476, train_acc: 91.67%, val_acc: 97.62%</em><br />
<em>Epoch [21], train_loss: 3.9354, val_loss: 3.9464, train_acc: 100.00%, val_acc: 97.86%</em><br />
<em>Epoch [22], train_loss: 3.9334, val_loss: 3.9441, train_acc: 100.00%, val_acc: 97.67%</em><br />
<em>Epoch [23], train_loss: 3.9314, val_loss: 3.9431, train_acc: 100.00%, val_acc: 97.77%</em><br />
<em>Epoch [24], train_loss: 3.9297, val_loss: 3.9404, train_acc: 91.67%, val_acc: 97.93%</em><br />
<br />

Let's visualize the data above in a graph.<br />
The training and validation accuracies are not far off from each other, with a validation accuracy of around 97%.
```markdown
# Plot train-validation accuracy

train_acc = [i['train_correct'] / i['train_total'] for i in history]
val_acc = [i['val_correct'] / i['val_total'] for i in history]

plot_chart('Train-Validation Accuracy', ['train', 'validation'], [train_acc, val_acc], 'number of epochs', 'accuracy')
```
![alt text](https://raw.githubusercontent.com/weechien/fruits-360/master/train_val_acc.JPG "Train-val accuracy")
<br /><br />

The training and validation losses are also stable.
```markdown
# Plot training-validation loss

train_loss = [i['train_loss'] for i in history]
val_loss = [i['val_loss'] for i in history]

plot_chart('Train-Validation Accuracy', ['train', 'validation'], [train_loss, val_loss], 'number of epochs', 'loss')
```
![alt text](https://raw.githubusercontent.com/weechien/fruits-360/master/train_val_loss.JPG "Train-val loss")
<br /><br />

Finally, let's run the model on the testing dataset.
```markdown
# Prediction on testing data

test_preds = predict_dl(test_dataloader, model)
print(f'Accuracy on test data: {test_preds:.2%}')
```
Accuracy on test data: 98.46%

## Conclusion and closing thoughts
test
