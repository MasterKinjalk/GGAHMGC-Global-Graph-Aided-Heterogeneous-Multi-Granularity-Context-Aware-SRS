from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os

class CustomDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.data = pd.read_csv(file_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        if self.transform:
            item = self.transform(item)
        return item

def get_data_loader(file_path, batch_size=32, shuffle=True, num_workers=0):
    dataset = CustomDataset(file_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)