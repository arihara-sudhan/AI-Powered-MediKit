import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

def get_classifier():
    model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
    model.eval()
    model.head = nn.Identity()

    class TEmbeddingNet(nn.Module):
        def __init__(self, modelt):
            super(TEmbeddingNet, self).__init__()
            self.modelt = modelt
            self.modelt.head = nn.Identity()

        def forward(self, x):
            return self.modelt(x)

        def get_embedding(self, x):
            return self.forward(x)


    class TripletNet(nn.Module):
        def __init__(self, embedding_net):
            super(TripletNet, self).__init__()
            self.enet = embedding_net

        def forward(self, x1, x2=None, x3=None):
            if x2 is None and x3 is None:
                return self.enet.get_embedding(x1)
            return (self.enet.get_embedding(x1),
                    self.enet.get_embedding(x2),
                    self.enet.get_embedding(x3))

        def get_embedding(self, x):
            return self.enet.get_embedding(x)

    tmodel = TEmbeddingNet(model)
    model = TripletNet(tmodel)
    model.load_state_dict(torch.load("models/fewshot_img.pth", map_location="cpu"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        
    model = model.to(device)
    return model


