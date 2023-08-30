from datetime import timedelta

from neuralop.datasets.hmi_zarr_dataset import HMIZarrDataset

def test_select_months():
    filename = '/d0/magnetograms/hmi/preprocessed/hmi_stacks_2021_2022_96m_full.zarr'
    test_months = (11,12)
    dataset = HMIZarrDataset(filename, months=test_months)
    assert len(dataset) > 0
    assert all([dataset.t_obs[i].month in test_months for i in dataset.x_index])
    #TODO: drop indices where y is not in month, see duplicate hmi_zarr_dataset.py todo
    # assert all([dataset.t_obs[i].month in test_months for i in dataset.y_index])
    
def test_cadence_filtering_default():
    filename = '/d0/magnetograms/hmi/preprocessed/hmi_stacks_2021_2022_96m_full.zarr'
    dataset = HMIZarrDataset(filename, months=[1])
    for i in range(len(dataset)):
        x_time = dataset.t_obs[dataset.x_index[i]]
        y_time = dataset.t_obs[dataset.y_index[i]]
        assert (y_time - x_time) > timedelta(minutes=91)
        assert (y_time - x_time) < timedelta(minutes=101)
        
        
def test_cadence_filtering_192_minutes():
    filename = '/d0/magnetograms/hmi/preprocessed/hmi_stacks_2021_2022_96m_full.zarr'
    dataset = HMIZarrDataset(filename, months=[1], cadence=timedelta(minutes=192))
    for i in range(len(dataset)):
        x_time = dataset.t_obs[dataset.x_index[i]]
        y_time = dataset.t_obs[dataset.y_index[i]]
        assert (y_time - x_time) > timedelta(minutes=187)
        assert (y_time - x_time) < timedelta(minutes=197)

def test_get_item():
    filename = '/d0/magnetograms/hmi/preprocessed/hmi_stacks_2021_2022_96m_full.zarr'
    dataset = HMIZarrDataset(filename)
    data = dataset[0]
    assert data['x'].shape == (1, 2048, 2048)
    assert data['y'].shape == (1, 2048, 2048)
    
    data = dataset[100]
    assert data['x'].shape == (1, 2048, 2048)
    assert data['y'].shape == (1, 2048, 2048)

    print(dataset.t_obs[dataset.x_index[-1]])

