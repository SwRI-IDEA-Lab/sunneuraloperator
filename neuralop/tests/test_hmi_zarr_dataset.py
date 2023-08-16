import pytest
from neuralop.datasets.hmi_zarr_dataset import HMIZarrDataset

def test_get_item():
    print("testing item shape")
    filename = '/d0/magnetograms/hmi/preprocessed/hmi_stacks_2021_2022_96m_full.zarr'
    dataset = HMIZarrDataset(filename)
    data = dataset[0]
    assert data['x'].shape == (2048, 2048)
    assert data['y'].shape == (2048, 2048)


