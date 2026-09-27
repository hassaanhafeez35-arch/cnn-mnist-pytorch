import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device using : ", device)

transform = transforms.ToTensor()

train_data = MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)


test_data = MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)


train_loader= DataLoader(
    train_data,
    shuffle=True,
    batch_size=64
)

test_loader = DataLoader(
    test_data,
    shuffle=False,
    batch_size=64
)


class SimpleCnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
           nn.Conv2d(1,8,kernel_size=3,padding=1) ,
           nn.ReLU(),
           nn.MaxPool2d(2),
           nn.Conv2d(8,16,kernel_size=3,padding=1),
           nn.ReLU(),
           nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16*7*7,64),
            nn.ReLU(),
            nn.Linear(64,10)

        )
    def forward(self , x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return(x)



model = SimpleCnn().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001 )


for epoch in range(3):
    model.train()
    total_loss =0
    for image , label in train_loader:
        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output = model(image)
        loss = loss_fn(output,label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch : {epoch +1}  , loss : {total_loss:.4f}")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for image , label in test_loader:
            image = image.to(device)
            label = label.to(device)

            output = model(image)
            prediction = output.argmax(dim = 1)

            correct += (prediction==label).sum().item()
            total += label.size(0)


accuracy = 100 * correct / total
print(f"Accuracy : {accuracy:.2f}")

torch.save(model.state_dict(),"simple_cnn_mnist.pth")
print("model saved")
    
index = 100
image, true_label = test_data[index]

image = image.unsqueeze(0).to(device)

with torch.no_grad():
    output= model(image)
    predicted_label = output.argmax(dim=1).item()


plt.imshow(image.cpu().squeeze(), cmap = "gray")
plt.title(f"Actual Label {true_label} | Prediction {prediction}")
plt.axis("off")
plt.show()


