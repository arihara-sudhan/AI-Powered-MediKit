from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from transformers import ViTForImageClassification, ViTFeatureExtractor

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

def infer_single_image(model_name, image_path, num_classes):
    try:
        model = VisionTransformerModel(num_classes=num_classes)
        model_path = f"./models/{model_name}.pth"
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
        return predicted.item()
    except RuntimeError as e:
        print(f"RuntimeError: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")