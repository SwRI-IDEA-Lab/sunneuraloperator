from datetime import timedelta

from neuralop.datasets.hmi_zarr_dataset import HMIZarrDataset

def test_select_months():
    filename = '/d0/magnetograms/hmi/preprocessed/hmi_stacks_2021_2022_96m_full.zarr'
    dataset = HMIZarrDataset(filename, months=[1])
    assert all([dataset.t_obs[i].month==1 for i in dataset.x_index])
    assert all([dataset.t_obs[i].month==1 for i in dataset.y_index])
    
def test_cadence_filtering_works():
    filename = '/d0/magnetograms/hmi/preprocessed/hmi_stacks_2021_2022_96m_full.zarr'
    dataset = HMIZarrDataset(filename, months=[1])
    for i in range(len(dataset)):
        x_time = dataset.t_obs[dataset.x_index[i]]
        y_time = dataset.t_obs[dataset.y_index[i]]
        assert (y_time - x_time) > timedelta(minutes=91)
        assert (y_time - x_time) < timedelta(minutes=101)

def test_get_item():
    filename = '/d0/magnetograms/hmi/preprocessed/hmi_stacks_2021_2022_96m_full.zarr'
    dataset = HMIZarrDataset(filename)
    data = dataset[0]
    assert data['x'].shape == (2048, 2048)
    assert data['y'].shape == (2048, 2048)
    
    data = dataset[100]
    assert data['x'].shape == (2048, 2048)
    assert data['y'].shape == (2048, 2048)

