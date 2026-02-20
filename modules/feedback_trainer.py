import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import logging
from typing import List, Tuple, Optional
from pathlib import Path


class PatchDataset(Dataset):
    def __init__(self, patches: List[np.ndarray], labels: List[int]):
        """Initialize dataset with patches and corresponding labels.
        
        Args:
            patches: List of image patches (H,W) or (H,W,C)
            labels: List of integer labels
        """
        if len(patches) != len(labels):
            raise ValueError("Patches and labels must have same length")
            
        self.patches = patches
        self.labels = torch.LongTensor(labels)  # Convert to tensor once
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # For grayscale
        ])

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        patch = self.patches[idx]
        label = self.labels[idx]
        
        # Handle single-channel and multi-channel cases
        if patch.ndim == 2:
            patch = np.expand_dims(patch, axis=-1)  # Add channel dim
        elif patch.ndim == 3 and patch.shape[2] > 1:
            patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
            patch = np.expand_dims(patch, axis=-1)
            
        patch = self.transform(patch.astype(np.float32) / 255.0)
        return patch, label


class FeedbackCNN(nn.Module):
    def __init__(self, input_size: int = 32):
        """CNN model for feedback learning.
        
        Args:
            input_size: Expected input size (assumes square input)
        """
        super(FeedbackCNN, self).__init__()
        self.input_size = input_size
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        # Calculate flattened size dynamically
        pool_size = input_size // 2
        self.fc_input_size = 32 * pool_size * pool_size
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.fc_input_size, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)
        )
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.fc(x)
        return x


class FeedbackTrainer:
    def __init__(self, model_path: str = 'models/feedback_cnn.pt'):
        """Initialize feedback training system.
        
        Args:
            model_path: Path to save/load model weights
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_path = Path(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model directory if needed
        os.makedirs(self.model_path.parent, exist_ok=True)
        
        self.model = FeedbackCNN(input_size=32).to(self.device)
        self._load_existing_model()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=2)

    def _load_existing_model(self):
        """Load existing model weights if available."""
        try:
            if self.model_path.exists():
                state_dict = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.logger.info(f"Loaded feedback model from {self.model_path}")
            else:
                self.logger.warning("No existing model found. Initialized fresh model.")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            self.logger.warning("Initializing new model instead")
            self.model = FeedbackCNN().to(self.device)

    def update_model(self, 
                    feedback_patches: List[np.ndarray], 
                    labels: List[int], 
                    epochs: int = 3,
                    batch_size: int = 16) -> float:
        """Train model with new feedback data.
        
        Args:
            feedback_patches: List of image patches
            labels: Corresponding labels (0 or 1)
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Final training loss
        """
        self.logger.info(f"Starting feedback training with {len(feedback_patches)} samples")
        
        # Create dataset and dataloader
        dataset = PatchDataset(feedback_patches, labels)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,
            pin_memory=True,
            num_workers=min(4, os.cpu_count() or 1)
        )

        self.model.train()
        best_loss = float('inf')
        
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

            epoch_loss = running_loss / len(dataloader)
            epoch_acc = 100 * correct / total
            self.scheduler.step(epoch_loss)
            
            self.logger.info(
                f"Epoch {epoch+1}/{epochs}, "
                f"Loss: {epoch_loss:.4f}, "
                f"Acc: {epoch_acc:.2f}%, "
                f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
            )
            
            # Save best model
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(self.model.state_dict(), self.model_path)
                self.logger.debug(f"Saved improved model (loss={best_loss:.4f})")

        return best_loss

    def predict(self, patch: np.ndarray) -> Tuple[int, np.ndarray]:
        """Make prediction on a single patch.
        
        Args:
            patch: Input image patch (H,W) or (H,W,C)
            
        Returns:
            Tuple of (predicted_class, class_probabilities)
        """
        self.model.eval()
        with torch.no_grad():
            # Preprocess input
            if patch.ndim == 2:
                patch = np.expand_dims(patch, axis=-1)
            elif patch.ndim == 3 and patch.shape[2] > 1:
                patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
                patch = np.expand_dims(patch, axis=-1)
                
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            input_tensor = transform(patch.astype(np.float32) / 255.0)
            input_tensor = input_tensor.unsqueeze(0).to(self.device)
            
            # Get prediction
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
            prediction = torch.argmax(output, dim=1).item()
            
            return prediction, probs