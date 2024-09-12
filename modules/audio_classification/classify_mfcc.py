import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

class MFCCNet(nn.Module):
    def __init__(self, num_classes):
        super(MFCCNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 3 * 11, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def classify(mfcc, fixed_size=(13, 44)):
    model = MFCCNet(num_classes=5)
    model.load_state_dict(torch.load('models/mfcc_model.pth', map_location=torch.device('cpu')))

    with open('data/audio/heartbeat/label_encoder_mfcc.pkl', 'rb') as f:
        label_encoder = pickle.load(f)

        if mfcc.shape[1] < fixed_size[1]:
            pad_width = ((0, 0), (0, fixed_size[1] - mfcc.shape[1]))
            mfcc = np.pad(mfcc, pad_width, mode='constant')
        elif mfcc.shape[1] > fixed_size[1]:
            mfcc = mfcc[:, :fixed_size[1]]

        if mfcc.shape[0] < fixed_size[0]:
            pad_height = fixed_size[0] - mfcc.shape[0]
            mfcc = np.pad(mfcc, ((0, pad_height), (0, 0)), mode='constant')
        elif mfcc.shape[0] > fixed_size[0]:
            mfcc = mfcc[:fixed_size[0], :]

        mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        mfcc_tensor = mfcc_tensor.to(device)

        with torch.no_grad():
            model.eval()
            outputs = model(mfcc_tensor)
            _, predicted = torch.max(outputs.data, 1)
        predicted_label = label_encoder.inverse_transform(predicted.cpu().numpy())[0]
        return predicted_label

