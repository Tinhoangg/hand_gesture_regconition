import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformer_build import HandGestureTransformer  

# label
CLASS_NAMES = ['Accept', 'Buy', 'Call', 'Candy', 'Catch', 'Deaf', 'Everyone','Food', 'Give', 'Green', 
               'Help', 'Hungry', 'I','Learn', 'Light-blue','Like', 'Milk', 'Music', 'Name', 'Red',
               'Ship', 'Son', 'Thanks','Want', 'Water', 'Where', 'Women', 'Yellow', 'Yogurt','You']

DATA_DIR = "lsa64_keypoints"
MAX_FRAMES = 60
BATCH_SIZE = 16
EPOCHS = 40
LR = 1e-4
INPUT_DIM = 2*21*3  # 126
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DATASET
class SignDataset(Dataset):
    def __init__(self, file_list, labels, max_frames=MAX_FRAMES):
        self.files = file_list
        self.labels = labels
        self.max_frames = max_frames

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        kp = np.load(self.files[idx])  # (num_frames, 2, 21, 3)
        num_frames = kp.shape[0]
        kp = kp.reshape(num_frames, -1)  # (num_frames, 126)

        # pad or cut
        if num_frames < self.max_frames:
            pad = np.zeros((self.max_frames - num_frames, kp.shape[1]), dtype=np.float32)
            kp = np.vstack([kp, pad])
        elif num_frames > self.max_frames:
            kp = kp[:self.max_frames]

        label = self.labels[idx]
        return torch.tensor(kp, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# LOAD FILES
file_list, labels = [], []
for idx, c in enumerate(CLASS_NAMES):
    class_folder = os.path.join(DATA_DIR, c)
    if not os.path.exists(class_folder):
        continue
    files = [os.path.join(class_folder, f) for f in os.listdir(class_folder) if f.endswith(".npy")]
    file_list.extend(files)
    labels.extend([idx]*len(files))

file_list = np.array(file_list)
labels = np.array(labels)

# SPLIT DATA
train_files, temp_files, train_labels, temp_labels = train_test_split(
    file_list, labels, test_size=0.3, stratify=labels, random_state=42
)
val_files, test_files, val_labels, test_labels = train_test_split(
    temp_files, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
)

train_dataset = SignDataset(train_files, train_labels)
val_dataset = SignDataset(val_files, val_labels)
test_dataset = SignDataset(test_files, test_labels)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# MODEL
model = HandGestureTransformer(input_dim=INPUT_DIM, num_classes=len(CLASS_NAMES), max_frames=MAX_FRAMES).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = torch.nn.CrossEntropyLoss()

# TRAINING LOOP 
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")

# SAVE SCRIPTED MODEL
scripted_model = torch.jit.script(model)
scripted_model.save("hand_gesture.pt")
print("Model saved as hand_gesture.pt")

