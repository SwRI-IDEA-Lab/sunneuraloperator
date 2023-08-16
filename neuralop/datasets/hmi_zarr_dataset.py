import torch
import zarr
from torch.utils.data import Dataset
import torchvision
from pathlib import Path
from torchvision import transforms

from ..utils import UnitGaussianNormalizer
from .hdf5_dataset import H5pyDataset
from .zarr_dataset import ZarrDataset
from .tensor_dataset import TensorDataset
from .positional_encoding import append_2d_grid_positional_encoding
from .transforms import Normalizer, PositionalEmbedding, MGPTensorDataset




class HMIZarrDataset(Dataset):
    """PDE hmi dataset that reads fits files turned into a zarr folder
        Parameters:
            filename: str
                filename pointing to the the zarr arry
            resolution: int
                resolution after subsampling
            center_crop_size: int
                size of center crop
            transform_x:
                input transformations to normalize and/or append a positional encoder
            transform_y:
                output transformations to normalize and/or append a positional encoder
            n_samples: int
                number of samples to use for training
            zarr_group: str
                group name in zarr dataset containing HMI data
            hmi_channel: int
                index of HMI in zarr group 
            tstep: int
                index difference between x and y, actual time difference depends on dataset cadence
    """
    def __init__(self, filename:str, resolution:int=4096, center_crop_size:int=2048, 
                 transform_x=None, transform_y=None,
                 n_samples:int=None, zarr_group:str='hmi', 
                 hmi_channel:int=0, tstep:int=1):
        self.zarr_group = zarr_group
        self.hmi_channel = hmi_channel
        self.tstep = tstep

        if center_crop_size > resolution:
            raise RuntimeError(f"Got resolution of {resolution}. Resolution be larger than center_crop_size, found {center_crop_size}.")
        
        resolution_to_step = {128:32, 256:16, 512:8, 1024:4,2048:2,4096:1}
        try:
            subsample_step = resolution_to_step[resolution]
        except KeyError:
            raise ValueError(f'Got {resolution=}, expected one of {resolution_to_step.keys()}')

        self.subsample_step = subsample_step
        self.filename = str(filename)
        self.crop_start = int((resolution - center_crop_size)/2)
        self.crop_end = int(self.crop_start+center_crop_size)
        
        self._data = None
        self.transform_x = transform_x
        self.transform_y = transform_y

        if n_samples is not None:
            self.n_samples = n_samples
        else:
            data = zarr.open(self.filename, mode='r')
            self.n_samples = data[self.zarr_group].shape[0] - self.tstep
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
    
        x = self.data[self.zarr_group][idx, self.hmi_channel, ::self.subsample_step, ::self.subsample_step]
        x = x[self.crop_start:self.crop_end,self.crop_start:self.crop_end]        
        
        x = torch.tensor(x, dtype=torch.float32)

        y = self.data[self.zarr_group][idx + self.tstep, self.hmi_channel, ::self.subsample_step, ::self.subsample_step]
        y = y[self.crop_start:self.crop_end,self.crop_start:self.crop_end]        
        
        y = torch.tensor(y, dtype=torch.float32)

        if self.transform_x:
            x = self.transform_x(x)

        if self.transform_y:
            y = self.transform_y(y)

        return {'x': x, 'y': y}
    
    def __getitems__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = torch.tensor([self.data[self.zarr_group][i, self.hmi_channel, ::self.subsample_step, ::self.subsample_step]
                          [self.crop_start:self.crop_end,self.crop_start:self.crop_end] for i in idx], dtype=torch.float32)
        y = torch.tensor([self.data[self.zarr_group][i+self.tstep, self.hmi_channel, ::self.subsample_step, ::self.subsample_step]
                          [self.crop_start:self.crop_end,self.crop_start:self.crop_end] for i in idx], dtype=torch.float32)

        if self.transform_x:
            x = self.transform_x(x)

        if self.transform_y:
            y = self.transform_y(y)

        return {'x': x, 'y': y}

def load_hmi_zarr(data_path, n_train, batch_size,
                            train_resolution=128,
                            test_resolutions=[128, 256, 512, 1024],
                            n_tests=[2000, 500, 500, 500],
                            test_batch_sizes=[8, 4, 1],
                            positional_encoding=True,
                            grid_boundaries=[[0,1],[0,1]],
                            encode_input=True,
                            encode_output=True,
                            num_workers=0, pin_memory=True, persistent_workers=False):
    data_path = Path(data_path)

    training_db = ZarrDataset(data_path / 'navier_stokes_1024_train.zarr', n_samples=n_train, resolution=train_resolution)
    transform_x = []
    transform_y = None

    if encode_input:
        x_mean = training_db.attrs('x', 'mean')
        x_std = training_db.attrs('x', 'std')
        
        transform_x.append(Normalizer(x_mean, x_std))
    
    if positional_encoding:
        transform_x.append(PositionalEmbedding(grid_boundaries, 0))

    if encode_output:
        y_mean = training_db.attrs('y', 'mean')
        y_std = training_db.attrs('y', 'std')
        
        transform_y = Normalizer(y_mean, y_std)

    training_db.transform_x = transforms.Compose(transform_x)
    training_db.transform_y = transform_y
    
    train_loader = torch.utils.data.DataLoader(training_db,
                                               batch_size=batch_size, drop_last=True,
                                               shuffle=True,
                                               num_workers=num_workers,
                                               pin_memory=pin_memory,
                                               persistent_workers=persistent_workers)

    test_loaders = dict()
    for (res, n_test, test_batch_size) in zip(test_resolutions, n_tests, test_batch_sizes):
        print(f'Loading test db at resolution {res} with {n_test} samples and batch-size={test_batch_size}')
        transform_x = []
        transform_y = None
        if encode_input:
            transform_x.append(Normalizer(x_mean, x_std))
        if positional_encoding:
            transform_x.append(PositionalEmbedding(grid_boundaries, 0))

        if encode_output:
            transform_y = Normalizer(y_mean, y_std)

        test_db = ZarrDataset(data_path / 'navier_stokes_1024_test.zarr', n_samples=n_test, resolution=res, 
                              transform_x=transforms.Compose(transform_x), transform_y=transform_y)
    
        test_loaders[res] = torch.utils.data.DataLoader(test_db, 
                                                        batch_size=test_batch_size,
                                                        shuffle=False,
                                                        num_workers=num_workers, 
                                                        pin_memory=pin_memory, 
                                                        persistent_workers=persistent_workers)

    return train_loader, test_loaders, transform_y