import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
import librosa # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
from pathlib import Path
import tqdm

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Parameters
SAMPLE_RATE = 16000  # 16kHz sample rate
N_MFCC = 40  # Number of MFCC features
MAX_LENGTH = 3  # Maximum audio length in seconds
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {DEVICE}")

# Audio Feature Extraction Functions
def extract_mfcc(file_path, n_mfcc=N_MFCC, max_len=MAX_LENGTH*SAMPLE_RATE):
    """Extract MFCC features from an audio file."""
    try:
        # Load audio file
        audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
        
        # Make sure audio is the right length
        if len(audio) > max_len:
            audio = audio[:max_len]
        else:
            # Pad with zeros if audio is too short
            padding = max_len - len(audio)
            audio = np.pad(audio, (0, padding), 'constant')
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=n_mfcc)
        
        # Normalize MFCC features
        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
        
        return mfcc
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Custom Dataset Class
class AudioDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels
        
        # Dictionary to map class labels to integers
        self.class_to_idx = {
            'Subhanallah': 0,
            'Alhamdulillah': 1,
            'Allahuakbar': 2
        }
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # Extract features
        mfcc = extract_mfcc(file_path)
        
        if mfcc is None:
            # Return a placeholder if processing failed
            mfcc = np.zeros((N_MFCC, MAX_LENGTH * SAMPLE_RATE // 512 + 1))
        
        # Convert to tensor and flatten
        mfcc_tensor = torch.tensor(mfcc.reshape(-1), dtype=torch.float32)
        label_tensor = torch.tensor(self.class_to_idx[label], dtype=torch.long)
        
        return mfcc_tensor, label_tensor

# Function to load and split the dataset
def prepare_dataset(data_dir):
    file_paths = []
    labels = []
    
    # Walk through the data directory
    for word in os.listdir(data_dir):
        word_dir = os.path.join(data_dir, word)
        if os.path.isdir(word_dir):
            for file_name in os.listdir(word_dir):
                if file_name.endswith('.wav'):
                    file_path = os.path.join(word_dir, file_name)
                    file_paths.append(file_path)
                    labels.append(word)
    
    # Split the dataset: 80% train, 20% test
    train_files, test_files, train_labels, test_labels = train_test_split(
        file_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Training samples: {len(train_files)}")
    print(f"Testing samples: {len(test_files)}")
    
    return train_files, test_files, train_labels, test_labels

# FeedForward Neural Network Model
class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, dropout_rate=0.5):
        super(FeedForwardNN, self).__init__()
        
        # Create a list to hold all layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_sizes[i+1]))
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], num_classes))
        
        # Sequential container
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, patience=10):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            progress_bar.set_postfix(loss=loss.item(), acc=100.*correct/total)
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = 100. * correct / total
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            progress_bar = tqdm.tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                progress_bar.set_postfix(loss=loss.item(), acc=100.*correct/total)
        
        epoch_val_loss = running_loss / len(val_loader.dataset)
        epoch_val_acc = 100. * correct / total
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%")
        print(f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")
        
        # Early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Load the best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses, train_accuracies, val_accuracies

# Evaluate the model
def evaluate_model(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate accuracy
    test_acc = accuracy_score(all_labels, all_preds) * 100
    
    # Create confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    return test_acc, conf_matrix, running_loss / len(test_loader.dataset)

# Plot confusion matrix
def plot_confusion_matrix(conf_matrix, class_names):
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

# Plot training and validation curves
def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot loss
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    
    # Plot accuracy
    ax2.plot(train_accuracies, label='Training Accuracy')
    ax2.plot(val_accuracies, label='Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

# Main execution
def main():
    # Set your data directory here
    data_dir = "dataset"
    
    # Create sample folders for demonstration if they don't exist
    for word in ['Subhanallah', 'Alhamdulillah', 'Allahuakbar']:
        os.makedirs(os.path.join(data_dir, word), exist_ok=True)
    
    # Check if dataset exists and has files
    has_files = False
    for word in ['Subhanallah', 'Alhamdulillah', 'Allahuakbar']:
        word_dir = os.path.join(data_dir, word)
        if os.path.exists(word_dir) and len(os.listdir(word_dir)) > 0:
            has_files = True
            break
    
    if not has_files:
        print("No audio files found. Please record audio files and place them in the dataset folders.")
        print("Expected directory structure:")
        print(f"{data_dir}/")
        print("  ├── Subhanallah/")
        print("  ├── Alhamdulillah/")
        print("  └── Allahuakbar/")
        return
    
    # Prepare dataset
    train_files, test_files, train_labels, test_labels = prepare_dataset(data_dir)
    
    # Create datasets and dataloaders
    train_dataset = AudioDataset(train_files, train_labels)
    test_dataset = AudioDataset(test_files, test_labels)
    
    # Calculate input size based on MFCC features
    sample_mfcc = extract_mfcc(train_files[0])
    input_size = sample_mfcc.size if sample_mfcc is not None else N_MFCC * (MAX_LENGTH * SAMPLE_RATE // 512 + 1)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create and train the model
    hidden_sizes = [512, 256, 128]  # Three hidden layers
    model = FeedForwardNN(input_size, hidden_sizes, num_classes=3).to(DEVICE)
    
    # Print model summary
    print(model)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Train the model
    model, train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, test_loader, criterion, optimizer, num_epochs=50
    )
    
    # Evaluate the model
    test_acc, conf_matrix, test_loss = evaluate_model(model, test_loader, criterion)
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Plot results
    class_names = ['Subhanallah', 'Alhamdulillah', 'Allahuakbar']
    plot_confusion_matrix(conf_matrix, class_names)
    plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies)
    
    # Save the model
    torch.save(model.state_dict(), 'islamic_phrases_classifier.pth')
    print("Model saved successfully!")

if __name__ == "__main__":
    main()