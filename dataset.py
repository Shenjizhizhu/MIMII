import toml
import argparse
from pathlib import Path
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import librosa
import librosa.display
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from typing import Optional,List,Dict,Tuple
from Test.utils import *

if __name__ =='__main__':    
    parser = argparse.ArgumentParser(description='MIMII Fault Diagnosis')
    parser.add_argument('config',help='配置文件')
    args = parser.parse_args()
    tomlfile = Path(args.config)
    with open(tomlfile,'rb') as file:
        configs = toml.load(file)

class MIMIIDataset(Dataset):
    def __init__(self,metadata:List[Dict],data_root:str,transform:Optional[A.BasicTransform] = None,dev_type:Optional[str] = None,max_samples:Optional[int] = None):
        self.metadata = metadata
        self.data_root = data_root
        self.transform = transform
        self.dev_type = dev_type
        self.max_samples = max_samples

        if self.dev_type is not None:
            self.metadata = [m for m in self.metadata if m.get("dev_type") == self.dev_type]
        if self.max_samples is not None:
            self.metadata = self.metadata[:self.max_samples]            

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self,idx):
        line = self.metadata[idx]
        wav_path = os.path.join(self.data_root,line["file_path"])
        y,sr = librosa.load(wav_path,sr = 16000,mono = True)
        mel_spec = self.wav2mel(y, use_torchaudio=False)
        
        if self.transform is not None:
            msl_spec = self.transform(image = mel_spec)["image"]

        label = 0 if line["label"] == "normal" else 1

        mel_spec = torch.from_numpy(mel_spec).float()
        label = torch.tensor(label,dtype = torch.long)

        return mel_spec,label
    
def get_mel_transfdorms(img_size: tuple = (64, 256)):
    train_transform = A.Compose([A.Resize(height = img_size[0],weight = img_size[1]),A.HorizontalFlip(p = 0.5),A.RandomBrightnessContrast(p = 0.5,brightness_limit = 0.1,contrast_limit = 0.1)],A.Normalize(mean =[np.random.randn(*img_size)],std = [np.std(np.random.randn(*img_size))],max_pixel_value = 1.0,p = 1.0 ),ToTensorV2())
    val_test_transform = A.Compose([A.Resize(height=img_size[0], width=img_size[1]),A.Normalize(mean=[0.0],std=[1.0],max_pixel_value=1.0,p=1.0),ToTensorV2()])
    return train_transform,val_test_transform
        

