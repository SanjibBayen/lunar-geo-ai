import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# ------------------------------
# Configurations
# ------------------------------
PATCH_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4
DATA_DIR = 'patches'  # Assumes subfolders: landslides/ and non_landslides/
MODEL_PATH = 'models/landslide_cnn.pt'

# ------------------------------
# Custom Dataset
# ------------------------------
class FolderLandslideDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform or transforms.ToTensor()

        for label, subfolder in enumerate(['non_landslides', 'landslides']):
            folder_path = Path(root_dir) / subfolder
            for img_file in folder_path.glob("*.png"):
                self.samples.append((img_file, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('L')
        image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

# ------------------------------
# CNN Model
# ------------------------------
class LandslideCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128), nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.net(x)
        return self.classifier(x)

# ------------------------------
# Train Function
# ------------------------------
def train():
    transform = transforms.Compose([
        transforms.Resize((PATCH_SIZE, PATCH_SIZE)),
        transforms.ToTensor()
    ])

    dataset = FolderLandslideDataset(DATA_DIR, transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LandslideCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Training on {len(train_ds)} samples | Validating on {len(val_ds)} samples")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

        acc = correct / len(train_ds)
        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}, Accuracy = {acc:.4f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")

# ------------------------------
# Run
# ------------------------------
if __name__ == "__main__":
    train()
