import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import os, random

class EmbeddingNet(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.mobilenet_v2(pretrained=True)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embed = nn.Linear(backbone.last_channel, 64)
    def forward(self,x):
        x = self.features(x)
        x = self.pool(x).view(x.size(0),-1)
        return F.normalize(self.embed(x),p=2,dim=1)

class TripletLoss(nn.Module):
    def __init__(self,margin=0.2): super().__init__(); self.margin=margin
    def forward(self,a,p,n):
        return F.relu(F.pairwise_distance(a,p)-F.pairwise_distance(a,n)+self.margin).mean()

class TripletLensDataset(Dataset):
    def __init__(self,root,transform=None):
        self.root, self.ids = root, os.listdir(root)
        self.t = transform or transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    def __len__(self): return 100000
    def __getitem__(self,idx):
        aid = random.choice(self.ids); nid = random.choice([i for i in self.ids if i!=aid])
        ap = os.listdir(os.path.join(self.root,aid))
        a = random.choice(ap); p = random.choice([i for i in ap if i!=a])
        n = random.choice(os.listdir(os.path.join(self.root,nid)))
        A = Image.open(os.path.join(self.root,aid,a)).convert('RGB')
        P = Image.open(os.path.join(self.root,aid,p)).convert('RGB')
        N = Image.open(os.path.join(self.root,nid,n)).convert('RGB')
        return self.t(A), self.t(P), self.t(N)

def extract_deep_embedding(image, model_path="model_triplet.pth"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EmbeddingNet().to(device)
    try:
        model.load_state_dict(torch.load(model_path,map_location=device))
    except Exception:
        # If model not found, return zeros for dev
        return np.zeros(64, dtype=np.float32)
    model.eval()
    img = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    t = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    x = t(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(x).cpu().numpy().ravel()
    return emb  # 64-D normalized 

def train_triplet(root,model_path,epochs=10,bs=32,lr=1e-4):
    ds = TripletLensDataset(root)
    dl = DataLoader(ds,batch_size=bs,shuffle=True,num_workers=4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EmbeddingNet().to(device)
    crit = TripletLoss().to(device)
    opt = torch.optim.Adam(model.parameters(),lr=lr)
    for ep in range(epochs):
        tot=0
        for A,P,N in dl:
            A,P,N=A.to(device),P.to(device),N.to(device)
            EA,EP,EN = model(A),model(P),model(N)
            loss = crit(EA,EP,EN)
            opt.zero_grad(); loss.backward(); opt.step()
            tot+=loss.item()
        print(f"Epoch{ep} loss={tot/len(dl)}")
    torch.save(model.state_dict(),model_path) 