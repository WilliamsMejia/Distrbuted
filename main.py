import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import torch.optim as optim
import csv

# Define the CIFAR-10 classes
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Data preparation and transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batch_size = 32

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Initialize the model and move to device
net = resnet18(num_classes=10)
net.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, nesterov=True)

# Open CSV file to log training and validation accuracy
with open('training_validation_accuracy.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Training Accuracy', 'Validation Accuracy'])

    # Training loop
    for epoch in range(20):
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        net.train()  # Set model to training mode
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        # Calculate and store training accuracy for this epoch
        train_acc = 100 * correct_train / total_train
        
        # Validation phase
        correct_val = 0
        total_val = 0
        net.eval()  # Set model to evaluation mode
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        # Calculate and store validation accuracy for this epoch
        val_acc = 100 * correct_val / total_val

        # Write the accuracies to the CSV file
        writer.writerow([epoch + 1, train_acc, val_acc])

        print(f'Epoch {epoch + 1}, Train Accuracy: {train_acc:.2f}%, Validation Accuracy: {val_acc:.2f}%')

print('Finished Training')

# Test the network on the test data
dataiter = iter(testloader)
images, labels = next(dataiter)
print('GroundTruth:', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.to(device)).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')

# Save the trained model
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)
