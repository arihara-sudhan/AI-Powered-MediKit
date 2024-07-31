import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTFeatureExtractor
from tqdm import tqdm

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class VisionTransformerModel(nn.Module):
    def __init__(self, num_classes):
        super(VisionTransformerModel, self).__init__()
        self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=num_classes, ignore_mismatched_sizes=True)
        self.model.classifier = nn.Linear(self.model.config.hidden_size, num_classes)

    def forward(self, xb):
        return self.model(xb).logits

feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
])

train_dataset = datasets.ImageFolder(root='/kaggle/input/medical-image-processing-ari/KIDNEY_/KIDNEY_/TRAIN', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = datasets.ImageFolder(root='/kaggle/input/medical-image-processing-ari/KIDNEY_/KIDNEY_/TEST', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

num_classes = 5
model = VisionTransformerModel(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-5)

num_epochs = 25
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}")
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        
        progress = (batch_idx + 1) * len(images)
        total = len(train_loader.dataset)
        pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Progress': f'{progress}/{total}'})
        
    epoch_loss = running_loss / len(train_dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}', flush=True)
    
    torch.save(model.state_dict(), "model_vit.pth")
    print(f'Model saved to model_vit.pth', flush=True)
    
    model.eval()
    correct = 0
    total = 0
    pbar = tqdm(test_loader, desc=f"Testing Epoch {epoch+1}/{num_epochs}")
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        progress = (batch_idx + 1) * len(images)
        total = len(test_loader.dataset)
        pbar.set_postfix({'Progress': f'{progress}/{total}'})
        
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%', flush=True)