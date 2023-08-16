import torch
import zarr
from torch.utils.data import Dataset
import torchvision


class HMIZarrDataset(Dataset):
    """PDE hmi dataset that reads fits files turned into a zarr folder
        Paramaters:
            filename: str
                filename pointing to the the zarr arry
            resolution: int
                resolution after subsampling
            center_crop_size:
                size of center crop
            transform_B:
                transformations to normalize and/or append a positional encoder
            n_samples:
                number of samples to use for training

    """
    def __init__(self, filename:str, resolution:int=4096, center_crop_size:int=2048, 
                 transform_B=None,
                 n_samples:int=None):
        if center_crop_size > resolution:
            raise RuntimeError(f"Got resolution of {resolution}. Resolution be larger than center_crop_size, found {center_crop_size}.")
        
        resolution_to_step = {128:32, 256:16, 512:8, 1024:4,2048:2,4096:1}
        try:
            subsample_step = resolution_to_step[resolution]
        except KeyError:
            raise ValueError(f'Got {resolution=}, expected one of {resolution_to_step.keys()}')

        self.subsample_step = subsample_step
        self.filename = str(filename)
        self.crop_start = (resolution - center_crop_size)/2
        self.crop_end = self.crop_start+center_crop_size
        
        self._data = None
        self.transform_B = transform_B

        if n_samples is not None:
            self.n_samples = n_samples
        else:
            data = zarr.open(self.filename, mode='r')
            self.n_samples = data.shape[0]
            del data

    def attrs(self, array_name, name):
        data = zarr.open(self.filename, mode='r', synchronizer=zarr.ThreadSynchronizer())
        value = data[array_name].attrs[name]
        del data
        return value

    @property
    def data(self):
        if self._data is None:
            self._data = zarr.open(self.filename, mode='r', synchronizer=zarr.ThreadSynchronizer())
        return self._data

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if isinstance(idx, int):
            assert idx < self.n_samples, f'Trying to access sample {idx} of dataset with {self.n_samples} samples'
        else:
            for i in idx:
                assert i < self.n_samples, f'Trying to access sample {i} of dataset with {self.n_samples} samples'
    
        B = self.data['Blos'][idx, ::self.subsample_step, ::self.subsample_step]
        B = B[self.crop_start:self.crop_end,self.crop_start:self.crop_end]        
        
        B = torch.tensor(B, dtype=torch.float32)

        if self.transform_B:
            B = self.transform_B(B)

        return B
    
    def __getitems__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        B = torch.tensor([self.data['Blos'][i, ::self.subsample_step, ::self.subsample_step]
                          [self.crop_start:self.crop_end,self.crop_start:self.crop_end] for i in idx], dtype=torch.float32)
        
        if self.transform_B:
            B = self.transform_B(B)

        return B
