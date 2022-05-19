import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import numpy as np

class CIFAR_1O_Corrupted(Dataset):
    # Working on CIFAR 10 C

	# Download data: wget https://zenodo.org/record/2535967/files/CIFAR-10-C.tar?download=1

	# Use severity one for all corruptions

    def __init__(self,datapath,transform):
        """
        Args:
            datapath (string): Path to the CIFAR-10-C folder     
          
        """
        self.transform = transform
        CORRPUTED_FILES_DIR = datapath
        label_file = np.load(os.path.join(CORRPUTED_FILES_DIR,'labels.npy'))
        corrupted_files = [os.path.join(CORRPUTED_FILES_DIR,f.name) for f in os.scandir(CORRPUTED_FILES_DIR) if str(f.name).endswith('npy') and str(f.name)!='labels.npy']

        # 10k of 5 sevrity. we  want 200 from each 10k
        all_indices = [i for i in range(len(label_file))]
       
        required_range =  [all_indices[i:i + 200] for i in range(0, len(all_indices), 10000)]
        required_range = [a for m in required_range for a in m]
        
        corrupted_files_numpy = [np.load(f)[required_range] for f in corrupted_files ]
        label_files_numpy = [label_file[required_range] for f in corrupted_files ]

        self.data = np.vstack(corrupted_files_numpy)
        self.labels = np.hstack(label_files_numpy)

        assert (self.data.shape[0] == self.labels.shape[0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            image_pil = transforms.ToPILImage()(image) # Need to transform to PIL before using default transforms
            image = self.transform(image_pil) 

        return image, label
