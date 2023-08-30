from typing import Optional, List
import torch
from datetime import timedelta
from torch.utils.data import Dataset
from pathlib import Path
from torchvision import transforms
import numpy as np
import pandas as pd
import xarray as xr

from .transforms import Normalizer, PositionalEmbedding
from ..utils import HMINormalizer

class HMIZarrDataset(Dataset):
    """PDE hmi dataset that reads fits files turned into a zarr folder
        Parameters:
            filename: str
                filename pointing to the the zarr arry
            months: Optional[List[int]]
                which months to use in this dataset
            resolution: int
                resolution after subsampling
            center_crop_size: int
                size of center crop
            transform_x:
                input transformations to normalize and/or append a positional encoder
            transform_y:
                output transformations to normalize and/or append a positional encoder
            zarr_group: str
                group name in zarr dataset containing HMI data
            hmi_channel: int
                index of HMI in zarr group 
            cadence: timedelta
                required cadence of the data
            cadence_epsilon: timedelta
                tolerance for cadence, 
                    i.e. only observations with cadence +/- cadence_epsilon are used
    """
    def __init__(self, filename:str, 
                 months: Optional[List[int]] = None, 
                 resolution: int = 4096, 
                 center_crop_size: int = 2048, 
                 transform_x = None, 
                 transform_y = None, 
                 zarr_group:str='hmi',
                 hmi_channel:int=0, 
                 cadence: timedelta = timedelta(minutes=96), 
                 cadence_epsilon: timedelta = timedelta(minutes=5)):
        self.zarr_group = zarr_group
        self.hmi_channel = hmi_channel
        self.cadence_window = (cadence - cadence_epsilon, cadence + cadence_epsilon)

        if center_crop_size > resolution:
            raise RuntimeError(f"Got resolution of {resolution}."
                               "Resolution be larger than center_crop_size,"
                               f"found {center_crop_size}.")
        
        resolution_to_step = {128:32, 256:16, 512:8, 1024:4, 2048:2, 4096:1}
        try:
            subsample_step = resolution_to_step[resolution]
        except KeyError:
            raise ValueError(f'Got {resolution=},'
                             f'expected one of {resolution_to_step.keys()}')

        self.subsample_step = subsample_step
        self.filename = str(filename)
        self.crop_start = int((resolution - center_crop_size)/2)
        self.crop_end = int(self.crop_start+center_crop_size)
        
        self.transform_x = transform_x
        self.transform_y = transform_y

        self.data = xr.open_zarr(self.filename)
        
        # select only observations in the requested months
        if months is None:
            months = list(range(1, 13))
        self.t_obs =  pd.to_datetime(self.data['t_obs'].data)
        selected_times = np.array([(i, t) for i, t in 
                                   enumerate(self.t_obs) if t.month in months])
        
        # filter to make sure that each x observation has a corresponding valid y
        t_diff = np.median(np.diff(self.t_obs))
        step = int(np.round(np.timedelta64(cadence)/t_diff))
        self.x_index = [i for (i, t) in selected_times 
         if i+step < len(self.t_obs) 
            and (self.t_obs[i+step] - t) > self.cadence_window[0] 
            and (self.t_obs[i+step] - t) < self.cadence_window[1]]
        self.y_index = [i+step for i in self.x_index]
        #TODO: drop indices where y is not in months

        self.n_samples = len(self.x_index)

        # set up the commonly used slices
        self.crop_slice = np.index_exp[self.crop_start:self.crop_end,
                                        self.crop_start:self.crop_end]
        self.sample_slicer = lambda idx: np.index_exp[idx, 
                                                        self.hmi_channel, 
                                                        ::self.subsample_step, 
                                                        ::self.subsample_step]

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if isinstance(idx, int):
            assert idx < self.n_samples, f'Trying to access sample {idx}'
            f'of dataset with {self.n_samples} samples'
        else:
            for i in idx:
                assert i < self.n_samples, f'Trying to access sample {i}'
                f'of dataset with {self.n_samples} samples'
    
        x = self.data[self.zarr_group][self.sample_slicer(self.x_index[idx])][self.crop_slice].load().data
        if x.ndim == 2:
            x = np.expand_dims(x,axis=0)
        x = torch.tensor(x, dtype=torch.float32)

        y = self.data[self.zarr_group][self.sample_slicer(self.y_index[idx])][self.crop_slice].load().data
        if y.ndim == 2:
            y = np.expand_dims(y,axis=0)
        y = torch.tensor(y, dtype=torch.float32)

        if self.transform_x:
            x = self.transform_x(x)

        if self.transform_y:
            y = self.transform_y(y)

        return {'x': x, 'y': y}
    
    # def __getitems__(self, idx):
    #     if torch.is_tensor(idx):
    #         idx = idx.tolist()

    #     x = torch.tensor([self.data[self.zarr_group][self.sample_slicer(self.x_index[i])][self.crop_slice].load().data for i in idx],
    #                      dtype=torch.float32)
    #     y = torch.tensor([self.data[self.zarr_group][self.sample_slicer(self.y_index[i])][self.crop_slice].load().data for i in idx],
    #                      dtype=torch.float32)

    #     if self.transform_x:
    #         x = self.transform_x(x)

    #     if self.transform_y:
    #         y = self.transform_y(y)

    #     return {'x': x, 'y': y}

def load_hmi_zarr(data_path: str, batch_size: int,
                            train_resolution=128,
                            test_resolutions=[128, 256, 512, 1024],
                            test_batch_sizes=[8, 4, 1],
                            positional_encoding=False,
                            grid_boundaries=[[0,1],[0,1]],
                            encode_input=True,
                            encode_output=True,
                            num_workers=0, pin_memory=True, persistent_workers=False,
                            center_crop_size=2048,
                            hmi_std=300,
                            train_months=(1,2,3,4,5,6,7,8,9),
                            test_months=(11,12)):
    """Load train, validation and test dataloaders from HMI Zarr dataset

    Args:
        data_path (str): path to HMI zarr
        batch_size (int): batch size for training
        train_resolution (int, optional): Resolution for training data. Defaults to 128.
        test_resolutions (list, optional): List of resolutions for testing. 
            Defaults to [128, 256, 512, 1024].
        test_batch_sizes (list, optional): Batch sizes for testing.
            Defaults to [8, 4, 1].
        positional_encoding (bool, optional): Whether or not to 
            include positional encoding. Defaults to False.
        grid_boundaries (list, optional): Defaults to [[0,1],[0,1]].
        encode_input (bool, optional): _description_. Defaults to True.
        encode_output (bool, optional): _description_. Defaults to True.
        num_workers (int, optional): _description_. Defaults to 0.
        pin_memory (bool, optional): _description_. Defaults to True.
        persistent_workers (bool, optional): _description_. Defaults to False.
        center_crop_size (int, optional): Size in pixels to 
            crop from center of full image. Defaults to 2048.
        hmi_std (int, optional): Standard deviation parameter 
            for normalizing HMI. Defaults to 300G.
        train_months (tuple) : Months to use for training. Defaults to Jan-Sept.
        test_months (tuple) : Months to use for testing. Defaults to Nov-Dec.

    Returns:
        train_loader
        test_loaders
        y_transform
    """    
    data_path = Path(data_path)

    training_db = HMIZarrDataset(data_path,
                                 resolution=train_resolution, 
                                 center_crop_size=center_crop_size,
                                 months=train_months)
    transform_x = []
    transform_y = None

    if encode_input:
        x_mean = 0
        x_std = hmi_std
        
        transform_x.append(HMINormalizer(x_mean, x_std))
    
    if positional_encoding:
        transform_x.append(PositionalEmbedding(grid_boundaries, 0))

    if encode_output:
        y_mean = 0
        y_std = hmi_std
        
        transform_y = HMINormalizer(y_mean, y_std)

    training_db.transform_x = transforms.Compose(transform_x)
    training_db.transform_y = transform_y
    
    train_loader = torch.utils.data.DataLoader(training_db,
                                               batch_size=batch_size, drop_last=True,
                                               shuffle=True,
                                               num_workers=num_workers,
                                               pin_memory=pin_memory,
                                               persistent_workers=persistent_workers)

    test_loaders = dict()
    for (res, test_batch_size) in zip(test_resolutions, test_batch_sizes):
        print(f'Loading test db at resolution {res} with batch-size={test_batch_size}')
        transform_x = []
        transform_y = None
        if encode_input:
            transform_x.append(HMINormalizer(x_mean, x_std))
        if positional_encoding:
            transform_x.append(PositionalEmbedding(grid_boundaries, 0))

        if encode_output:
            transform_y = HMINormalizer(y_mean, y_std)

        test_db = HMIZarrDataset(data_path, resolution=res, 
                              transform_x=transforms.Compose(transform_x), 
                              transform_y=transform_y,
                              months=test_months,
                              center_crop_size=center_crop_size)
    
        test_loaders[res] = torch.utils.data.DataLoader(test_db, 
                                                        batch_size=test_batch_size,
                                                        shuffle=False,
                                                        num_workers=num_workers, 
                                                        pin_memory=pin_memory, 
                                                        persistent_workers=persistent_workers)

    return train_loader, test_loaders, transform_y