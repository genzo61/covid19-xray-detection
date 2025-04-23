"""
Sınıflandırma problemi çözümü covid-19 dataseti ile CNN'leri kullanarak
"""

#%% import libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
# load dataset

def get_data_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    # veri setini yükleme
    train_Set = datasets.ImageFolder(root='dataset/train',transform=transform)
    test_set = datasets.ImageFolder(root='dataset/test',transform=transform)
    # data loader
    train_loader = DataLoader(train_Set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

#%% visualize dataset
def imshow(img):
    img = img / 2 + 0.5
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()
def get_sample_img(train_loader):
    dataiter = iter(train_loader)
    images,labels = next(dataiter)
    return images,labels
def visualize(n):
    train_loader, test_loader = get_data_loaders()
    images, labels = get_sample_img(train_loader)
    plt.figure(figsize=(15, 5))
    for i in range(n):
        plt.subplot(1,n,i+1)
        imshow(images[i])
        plt.title(labels[i])
    plt.show()
# visualize(3)
#%% build CNN Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # (1, 224, 224) → (32, 224, 224)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # → (32, 112, 112)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # → (64, 112, 112)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # → (64, 56, 56)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # → (128, 56, 56)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # → (128, 28, 28)

        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(128 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, 2)  # 2 sınıf için çıkış

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
model = CNN().to(device)
# define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
#%% Training and testing

# GPU kontrolü
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Kullanılan cihaz: {device}')


def train(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    toplam_kayip = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        dogru_tahmin = 0
        toplam = 0

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            toplam += labels.size(0)
            dogru_tahmin += (predicted == labels).sum().item()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], '
                      f'Batch [{i + 1}/{len(train_loader)}], '
                      f'Kayıp: {loss.item():.4f}')

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * dogru_tahmin / toplam
        toplam_kayip.append(epoch_loss)

        print(f'\nEpoch [{epoch + 1}/{num_epochs}] tamamlandı:')
        print(f'Ortalama Kayıp: {epoch_loss:.4f}')
        print(f'Doğruluk: {epoch_acc:.2f}%')
        print('-' * 60)

    return toplam_kayip


def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():  # Gradient hesaplanmasını devre dışı bırak
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = test_loss / len(test_loader)
    accuracy = 100 * correct / total

    print(f'\nTest Sonuçları:')
    print(f'Ortalama Test Kaybı: {avg_loss:.4f}')
    print(f'Test Doğruluğu: {accuracy:.2f}%')

    return avg_loss, accuracy


# Model oluştur ve cihaza taşı
model = CNN().to(device)

# Loss ve optimizer tanımla
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Veri yükleyicileri
train_loader, test_loader = get_data_loaders(batch_size=64)

# Eğitimi başlat
print("Eğitim başlıyor...")
kayip_gecmisi = train(model, train_loader, criterion, optimizer, num_epochs=10)

# Test et
print("\nTest başlıyor...")
test_loss, test_acc = test(model, test_loader, criterion)

# Eğitim kaybı grafiğini çiz
plt.figure(figsize=(10, 5))
plt.plot(kayip_gecmisi, label='Eğitim Kaybı')
plt.title('Eğitim Süresince Kayıp Değişimi')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()
plt.grid(True)
plt.show()

# Modeli kaydet
torch.save(model.state_dict(), 'model.pth')
print("\nModel 'model.pth' olarak kaydedildi.")

