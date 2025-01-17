#first make sure you have these installed with pip install {}
#importing all the necessary libraries and tools.
#pytorch is the main one, importing nn to build neural networks
#optim is to optimize using gradients
#datasets and transforms for handling data
#dataloader to batch and shuffle data
#F contains useful funcitons like ReLU used to filter data

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

#Get the training datasets
transform = transforms.Compose([transforms.ToTensor()]) # Converts images into PyTorch tensors (numeric arrays).

Converts images into PyTorch tensors (numeric arrays).
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform) 
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

#load the datasets, we divide data into training and testingn with a batch size of 32 images and shuffled for better training.
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)

# Simple Neural Network Blueprint class
# This nn.Module is the "base class" from PyTorch

class simpleNN(nn.Module):
    def __init__(self):
        super(simpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128) # The machine takes a 28x28 image and passes in to 128 Neurons
        self.fc2 = nn.Linear(128, 64) # The output from the first layer is passed to 64 neurons.
        self.fc3 = nn.Linear(64, 10) # The output from the second layer is passed onto 10 neurons. (0-9) representing digits.
        
    def forward(self, x):
        x = x.view(-1,28*28) #28*28 is a grid, but machine flattens it to a long row of 784 numbers.
        x = F.relu(self.fc1(x)) # The ReLU function ensures that only positive outputs are passed to the next layer
        x = F.relu(self.fc2(x)) # The ReLU function ensures that only positive outputs are passed to the next layer
        x = self.fc3(x) # we dont use ReLU here because we need the raw data for CrossEntropyLoss later.
        return x


model = simpleNN() #creating an instance
optimizer = optim.SGD(model.parameters(), lr=0.01) #optimizing with a learning rate of 0.01
criterion = nn.CrossEntropyLoss() # loss function measuing difference

for epoch in range(5): # run 5 whole iterations through dataset
    for images, labels in train_loader: 
        optimizer.zero_grad() # clear any leftovers from previous iteration
        output = model(images) # pass the batch of images
        loss = criterion(output, labels) #how wrong predictions are
        loss.backward() # Calculates gradients, gradients tell the optimizer how to adjust each weight to reduce the error
        optimizer.step() # Updates the weights
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    
correct = 0
total = 0

with torch.no_grad(): #disable gradient calculation
    for images, labels in test_loader: #loop over test data
        output = model(images) # pass images through trained model
        _, predicted = torch.max(output, 1) # output is a list of logits for each class
        total += labels.size(0) 
        correct += (predicted == labels).sum().item() #compare predicts with labels

print(f'Accuracy: {100 * correct / total}%')

torch.save(model.state_dict(), 'mnist_model.pth')

#plotting and visualizing the data with matplot library
import matplotlib.pyplot as plt

images, labels = next(iter(test_loader))
output = model(images)
_, preds = torch.max(output, 1)

for i in range(6):  # Show 6 images
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i][0], cmap='gray')
    plt.title(f'Label: {labels[i]}, Pred: {preds[i]}')
    plt.axis('off')
plt.show()
