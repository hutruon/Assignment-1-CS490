import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Layer 1
        self.conv1 = nn.Conv1d(4, 96, 11, stride=4, padding=5)  # Add padding
        self.relu1 = nn.ReLU()

        # Layer 2
        self.conv2 = nn.Conv1d(96, 96, 1, padding=1)  # Add padding
        self.relu2 = nn.ReLU()

        # Layer 3
        self.conv3 = nn.Conv1d(96, 96, 1, padding=1)  # Add padding
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(3, stride=2)
        self.drop3 = nn.Dropout(0.5)

        # Layer 4
        self.conv4 = nn.Conv1d(96, 192, 11, stride=4, padding=5)  # Add padding
        self.relu4 = nn.ReLU()

        # Layer 5
        self.conv5 = nn.Conv1d(192, 192, 1, padding=1)  # Add padding
        self.relu5 = nn.ReLU()

        # Layer 6
        self.conv6 = nn.Conv1d(192, 192, 1, padding=1)  # Add padding
        self.relu6 = nn.ReLU()
        self.pool6 = nn.MaxPool1d(3, stride=2)
        self.drop6 = nn.Dropout(0.5)

        # Layer 7
        self.conv7 = nn.Conv1d(192, 384, 3, stride=1, padding=1)  # Add padding
        self.relu7 = nn.ReLU()

        # Layer 8
        self.conv8 = nn.Conv1d(384, 384, 1, padding=1)  # Add padding
        self.relu8 = nn.ReLU()

        # Layer 9
        self.conv9 = nn.Conv1d(384, 384, 1, padding=1)  # Add padding
        self.relu9 = nn.ReLU()
        self.drop9 = nn.Dropout(0.5)

        # Layer 10
        self.conv10 = nn.Conv1d(384, 20, 3, stride=1, padding=1)  # Add padding
        self.relu10 = nn.ReLU()

        # Layer 11
        self.conv11 = nn.Conv1d(20, 20, 1, padding=1)  # Add padding
        self.relu11 = nn.ReLU()

        # Layer 12
        self.conv12 = nn.Conv1d(20, 20, 1, padding=1)  # Add padding
        self.relu12 = nn.ReLU()
        self.adapool = nn.AdaptiveAvgPool1d((1))

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.drop3(self.pool3(self.relu3(self.conv3(x))))
        x = self.relu4(self.conv4(x))
        x = self.relu5(self.conv5(x))
        x = self.drop6(self.pool6(self.relu6(self.conv6(x))))
        x = self.relu7(self.conv7(x))
        x = self.relu8(self.conv8(x))
        x = self.drop9(self.relu9(self.conv9(x)))
        x = self.relu10(self.conv10(x))
        x = self.relu11(self.conv11(x))
        x = self.adapool(self.relu12(self.conv12(x)))
        x = torch.flatten(x, 1)  # Flatten for potential further layers or a classifier
        return x


# Step 3: One-hot encoding function
def one_hot_encoder(sequence):
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
    return np.array([mapping.get(nucleotide, [0, 0, 0, 0]) for nucleotide in sequence])

# Function to read and encode data from a file
def read_data_and_encode(file_path):
    labels = []
    encoded_data = []
    with open(file_path, 'r') as file:
        for line in file:
            label, sequence = line.strip().split(' ', 1)
            labels.append(int(label))
            encoded_data.append(one_hot_encoder(sequence))
    return np.array(labels), np.array(encoded_data)

# Custom Dataset class
class DNADataset(Dataset):
    def __init__(self, sequences, labels):
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.data = torch.stack([torch.tensor(s, dtype=torch.float32) for s in sequences])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        # Reshape data to have 4 channels
        data = data.transpose(1, 0)
        return data, self.labels[idx]

# Main program
if __name__ == "__main__":
    # Step 4: Read and encode data
    labels, encoded_data = read_data_and_encode('test_data.txt')
    
    # Step 8: Instantiate the dataset
    test_dataset = DNADataset(encoded_data, labels)
    
    # Step 9: Create a DataLoader
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True)

    # Read and encode training data
    train_labels, train_encoded_data = read_data_and_encode('train_data.txt')

    # Instantiate the training dataset
    train_dataset = DNADataset(train_encoded_data, train_labels)

    # Create a DataLoader for the training data
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)  
    

# Instantiating the model and assigning an optimizer to the model and creating a loss function

model = Net().to(device)
optimizer=optim.Adam(params=model.parameters(),lr=0.001)
loss_fn = nn.CrossEntropyLoss()

def train(model,device,train_dataloader,optimizer,epochs):
    print("inside train")
    model.train()
    for batch_ids, (img, classes) in enumerate(train_dataloader):
        classes=classes.type(torch.LongTensor)
        img,classes=img.to(device),classes.to(device)
        torch.autograd.set_detect_anomaly(True)     
        optimizer.zero_grad()
        output=model(img)
        loss = loss_fn(output,classes)                
        
        loss.backward()
        optimizer.step()
    if(batch_ids +1) % 2 == 0:
        print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
            epochs, batch_ids* len(img), len(train_dataloader.dataset),
            100.*batch_ids / len(train_dataloader),loss.item()))

def test(model, device, test_dataloader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for img, classes in test_dataloader:
            img, classes = img.to(device), classes.to(device)
            output = model(img)
            test_loss += F.cross_entropy(output, classes, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True) 
            correct += pred.eq(classes.view_as(pred)).sum().item()

    test_loss /= len(test_dataloader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_dataloader.dataset),
        100. * correct / len(test_dataloader.dataset)))
    print('=' * 30)

if __name__=='__main__':
    seed=42
    EPOCHS=3
    
    for epoch in range(1,EPOCHS+1):
        train(model,device,train_dataloader,optimizer,epoch)
        test(model,device,test_dataloader)
