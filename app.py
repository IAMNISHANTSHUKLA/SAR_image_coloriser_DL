import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from skimage import io

SAR_DATASET_URL = "https://mediatum.ub.tum.de/1474000"
print(f"Dataset link: {SAR_DATASET_URL}")

class SARDataset(Dataset):
    def __init__(self, sar_images, optical_images, transform=None):
        self.sar_images = sar_images
        self.optical_images = optical_images
        self.transform = transform

    def __len__(self):
        return len(self.sar_images)

    def __getitem__(self, idx):
        sar_image = self.sar_images[idx]
        optical_image = self.optical_images[idx]

        if self.transform:
            sar_image = self.transform(sar_image)
            optical_image = self.transform(optical_image)

        return sar_image, optical_image


def load_sar_optical_data(dataset_path):
    
    sar_images = [io.imread(os.path.join(dataset_path, 'sar', file)) for file in os.listdir(os.path.join(dataset_path, 'sar'))]
    optical_images = [io.imread(os.path.join(dataset_path, 'optical', file)) for file in os.listdir(os.path.join(dataset_path, 'optical'))]

    return np.array(sar_images), np.array(optical_images)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

dataset_path = 'path_to_your_dataset'

sar_images, optical_images = load_sar_optical_data(dataset_path)
sar_dataset = SARDataset(sar_images, optical_images, transform=transform)
dataloader = DataLoader(sar_dataset, batch_size=16, shuffle=True)


def build_keras_model():
    input_img = Input(shape=(256, 256, 1))

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(3, (3, 3), activation='tanh', padding='same')(x)

    model = Model(input_img, x)
    model.compile(optimizer=Adam(), loss='mse')

    return model


keras_model = build_keras_model()
keras_model.summary()


class PyTorchColorizationNet(nn.Module):
    def __init__(self):
        super(PyTorchColorizationNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


pytorch_model = PyTorchColorizationNet()
print(pytorch_model)



def train_keras_model(model, dataloader, epochs=10):
    for epoch in range(epochs):
        for sar, optical in dataloader:
            sar = np.expand_dims(sar.numpy(), axis=-1)  # Add channel dimension
            model.train_on_batch(sar, optical)
        print(f"Epoch {epoch + 1}/{epochs} completed for Keras model")

def train_pytorch_model(model, dataloader, epochs=10):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        for sar, optical in dataloader:
            sar, optical = sar.float(), optical.float()

            optimizer.zero_grad()
            outputs = model(sar)
            loss = criterion(outputs, optical)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs} completed for PyTorch model")


train_keras_model(keras_model, dataloader)
train_pytorch_model(pytorch_model, dataloader)


def visualize_results(keras_model, pytorch_model, test_data):
    keras_output = keras_model.predict(np.expand_dims(test_data.numpy(), axis=-1))
    pytorch_output = pytorch_model(test_data.float()).detach().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(test_data.squeeze(), cmap='gray')
    axes[0].set_title('Original SAR Image')

    axes[1].imshow(keras_output.squeeze())
    axes[1].set_title('Keras Model Output')

    axes[2].imshow(np.transpose(pytorch_output.squeeze(), (1, 2, 0)))
    axes[2].set_title('PyTorch Model Output')

    plt.show()


sample_sar, _ = next(iter(dataloader))
visualize_results(keras_model, pytorch_model, sample_sar[0])
