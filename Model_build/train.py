from transformer_build import HandGestureTransformer
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
import os
import matplotlib.pyplot as plt

# Load data
class HandGestureDataset(Dataset):
    def __init__(self, data_dir, max_frames=100):
        """
        Args:
            data_dir: Path to data directory 
            max_frames: Maximum number of frames
        """
        self.data_dir = Path(data_dir)
        self.max_frames = max_frames
        self.samples = []
        self.labels = []
        self.label_to_idx = {}
        self.idx_to_label = {}
        # Load all samples
        gesture_folders = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        for idx, gesture_folder in enumerate(gesture_folders):
            gesture_name = gesture_folder.name
            self.label_to_idx[gesture_name] = idx
            self.idx_to_label[idx] = gesture_name
            # Load all .npy files in this gesture folder
            npy_files = list(gesture_folder.glob('*.npy'))
            for npy_file in npy_files:
                self.samples.append(npy_file)
                self.labels.append(idx)
        print(f"Loaded {len(self.samples)} samples from {len(gesture_folders)} classes")
        print(f"Classes: {list(self.label_to_idx.keys())}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Load numpy file: shape (frames, 42, 3)
        data = np.load(self.samples[idx])
        label = self.labels[idx]
        # Handle variable length sequences
        num_frames = data.shape[0]
        if num_frames > self.max_frames:
            # Sample evenly from the sequence
            indices = np.linspace(0, num_frames - 1, self.max_frames, dtype=int)
            data = data[indices]
            mask = np.ones(self.max_frames, dtype=np.float32)
        else:
            # Pad
            padding = np.zeros((self.max_frames - num_frames, 42, 3))
            data = np.concatenate([data, padding], axis=0)
            mask = np.concatenate([
                np.ones(num_frames, dtype=np.float32),
                np.zeros(self.max_frames - num_frames, dtype=np.float32)
            ])
        # Flatten keypoints: (frames, 42, 3) -> (frames, 126)
        data = data.reshape(self.max_frames, -1)
        return torch.FloatTensor(data), torch.LongTensor([label])[0], torch.FloatTensor(mask)

# Training Function với gradient clipping
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc='Training')
    for data, labels, mask in pbar:
        data, labels, mask = data.to(device), labels.to(device), mask.to(device)
        
        optimizer.zero_grad()
        outputs = model(data, mask)
        loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        
        total_loss += loss.item()
        
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy

# Validation Function
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels, mask in tqdm(dataloader, desc='Validation'):
            data, labels, mask = data.to(device), labels.to(device), mask.to(device)
            
            outputs = model(data, mask)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy, all_preds, all_labels

# Plot metrics
def plot_metrics(train_losses, val_losses, train_accs, val_accs, save_path=None):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    plt.title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Train Acc', linewidth=2)
    plt.plot(epochs, val_accs, 'r-', label='Val Acc', linewidth=2)
    plt.title('Training vs Validation Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# Main
def main():
    # Configuration với regularization tăng cường
    config = {
        'data_dir': 'D:\Semester\Semester5\DPL302\Project\dataset\data_split',
        'max_frames': 100,
        'batch_size': 16,
        'd_model': 128,
        'nhead': 8,
        'num_layers': 2,
        'dropout': 0.2,  # TĂNG từ 0.2 lên 0.4
        'learning_rate': 0.0001,
        'weight_decay': 0.0001,  # THÊM weight decay
        'num_epochs': 30,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    print(f"Using device: {config['device']}")
    
    # Load datasets
    train_dataset = HandGestureDataset(
        os.path.join(config['data_dir'], 'train'),
        max_frames=config['max_frames']
    )
    val_dataset = HandGestureDataset(
        os.path.join(config['data_dir'], 'val'),
        max_frames=config['max_frames']
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # Initialize model
    num_classes = len(train_dataset.label_to_idx)
    model = HandGestureTransformer(
        input_dim=126,  # 42 keypoints * 3 coordinates
        d_model=config['d_model'],
        n_head=config['nhead'],
        num_layers=config['num_layers'],
        num_classes=num_classes,
        dropout=config['dropout'],
        max_frames=config['max_frames']
    ).to(config['device'])
    
    # Loss and optimizer với weight decay
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']  # THÊM weight decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-6
    )
    
    # Training loop với early stopping
    best_val_acc = 0
    best_val_loss = float('inf')
    early_stopping_patience = 7
    early_stopping_counter = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, config['device']
        )
        val_loss, val_acc, val_preds, val_labels = validate(
            model, val_loader, criterion, config['device']
        )
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        scheduler.step(val_loss)
        
        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            print(f"Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")
        
        # Save best model (dựa trên accuracy)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'label_to_idx': train_dataset.label_to_idx,
                'config': config
            }, 'best_model_3.pth')
            print(f"Best model saved with validation accuracy: {val_acc:.4f}")
        
        # Check early stopping
        if early_stopping_counter >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch+1}!")
            break
    
    # Plot results
    plot_metrics(train_losses, val_losses, train_accs, val_accs, save_path='train_val_metrics_improved.png')

def test_model(model_path, test_data_dir, device='cuda', batch_size=32):
    """
    Test model on test set
    """
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    label_to_idx = checkpoint['label_to_idx']
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    
    # Load test dataset
    test_dataset = HandGestureDataset(test_data_dir, max_frames=config['max_frames'])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Initialize model
    model = HandGestureTransformer(
        input_dim=126,
        d_model=config['d_model'],
        n_head=config['nhead'],
        num_layers=config['num_layers'],
        num_classes=len(label_to_idx),
        dropout=config['dropout'],
        max_frames=config['max_frames']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded successfully!")
    
    # Test
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels, mask in tqdm(test_loader, desc='Testing'):
            data, labels, mask = data.to(device), labels.to(device), mask.to(device)
            outputs = model(data, mask)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return accuracy, all_preds, all_labels

if __name__ == '__main__':
    main()
