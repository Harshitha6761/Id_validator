"""
Training Script for College ID Validator
Fine-tunes a ResNet18 model on the generated synthetic ID card dataset.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from image_classifier import ImageClassifier
import numpy as np

# --- Configuration ---
DATASET_DIR = "test_dataset"
MODEL_OUTPUT_PATH = "model_weights.pkl"
NUM_EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 0.001
TRAIN_VAL_SPLIT = 0.8  # 80% for training, 20% for validation

# --- Dataset Class ---
class IDCardDataset(Dataset):
    """Custom PyTorch Dataset for loading ID card images."""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Could not open image {image_path}. Skipping. Error: {e}")
            # Return a dummy image and label if an image is corrupted
            return torch.zeros((3, 224, 224)), torch.tensor(0)

        # Apply transformations
        if self.transform:
            image = self.transform(image)
            
        # Get label
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return image, label

# --- Helper Functions ---
def load_dataset_paths_and_labels(dataset_dir):
    """Loads image file paths and their corresponding labels from the dataset directory."""
    
    dataset_file = os.path.join(dataset_dir, "complete_dataset.json")
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Dataset JSON file not found at {dataset_file}. Please generate the dataset first.")
        
    with open(dataset_file, "r") as f:
        dataset = json.load(f)
        
    image_paths = []
    labels = []
    
    # Define mapping from category to label index
    # As per ImageClassifier: 0=genuine, 1=suspicious, 2=fake
    label_map = {
        "genuine": 0,
        "suspicious": 1,
        "fake": 2
    }
    
    for category, samples in dataset.items():
        if category not in label_map:
            continue
            
        for sample in samples:
            image_filename = sample["filename"]
            # Correct the path to be relative to the script location
            image_path = os.path.join(dataset_dir, category, image_filename)
            
            if os.path.exists(image_path):
                image_paths.append(image_path)
                labels.append(label_map[category])
            else:
                print(f"Warning: Image file not found at {image_path}. Skipping.")

    return image_paths, labels

# --- Main Training Class ---
class ModelTrainer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize the model from the classifier script
        # This ensures we're using the exact same architecture
        self.model_wrapper = ImageClassifier(model_path=MODEL_OUTPUT_PATH)
        self.model = self.model_wrapper.model.to(self.device)
        
        # Define image transformations
        self.transform = self.model_wrapper.transform

    def prepare_data(self):
        """Load and prepare data loaders for training and validation."""
        
        image_paths, labels = load_dataset_paths_and_labels(DATASET_DIR)
        
        # Create full dataset
        full_dataset = IDCardDataset(image_paths, labels, transform=self.transform)
        
        # Split into training and validation sets
        num_samples = len(full_dataset)
        train_size = int(TRAIN_VAL_SPLIT * num_samples)
        val_size = num_samples - train_size
        
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        print(f"Dataset loaded: {num_samples} total samples")
        print(f"Training set: {len(train_dataset)} samples")
        print(f"Validation set: {len(val_dataset)} samples")
        
        # Create data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    def train(self):
        """Run the training and validation loop."""
        
        self.prepare_data()
        
        # Set up optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        
        best_val_accuracy = 0.0
        
        print("\nStarting model training...")
        print("="*30)
        
        for epoch in range(NUM_EPOCHS):
            # --- Training Phase ---
            self.model.train()
            total_train_loss = 0
            train_correct = 0
            
            for i, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
                
                # Calculate training accuracy
                _, predicted = torch.max(outputs.data, 1)
                train_correct += (predicted == labels).sum().item()

            avg_train_loss = total_train_loss / len(self.train_loader)
            train_accuracy = train_correct / len(self.train_loader.dataset)
            
            # --- Validation Phase ---
            self.model.eval()
            total_val_loss = 0
            val_correct = 0
            
            with torch.no_grad():
                for images, labels in self.val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    total_val_loss += loss.item()
                    
                    # Calculate validation accuracy
                    _, predicted = torch.max(outputs.data, 1)
                    val_correct += (predicted == labels).sum().item()

            avg_val_loss = total_val_loss / len(self.val_loader)
            val_accuracy = val_correct / len(self.val_loader.dataset)
            
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
                  f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2%} | "
                  f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2%}")

            # Save the best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                self.save_model_weights()
                print(f"   -> New best model saved with validation accuracy: {val_accuracy:.2%}")
        
        print("="*30)
        print("Training complete!")

    def save_model_weights(self):
        """Saves the model's state dictionary to the specified path."""
        torch.save(self.model.state_dict(), MODEL_OUTPUT_PATH)
        print(f"Model weights saved to {MODEL_OUTPUT_PATH}")

def main():
    """Main function to start the training process."""
    trainer = ModelTrainer()
    trainer.train()

if __name__ == "__main__":
    main() 