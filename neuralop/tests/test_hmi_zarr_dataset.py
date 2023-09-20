from datetime import timedelta

from neuralop.datasets.hmi_zarr_dataset import HMIZarrDataset

def test_select_months():
    filename = '/d0/magnetograms/hmi/preprocessed/hmi_stacks_2021_2022_96m_full.zarr'
    test_months = (11,12)
    dataset = HMIZarrDataset(filename, months=test_months)
    assert len(dataset) > 0
    assert all([dataset.t_obs[i].month in test_months for i in dataset.index])
    
def test_cadence_filtering_default():
    filename = '/d0/magnetograms/hmi/preprocessed/hmi_stacks_2021_2022_96m_full.zarr'
    dataset = HMIZarrDataset(filename, months=[1])
    data = dataset.getitem_times(0)
    assert (data['yt'][0] - data['xt'][0]) > timedelta(minutes=91)
    assert (data['yt'][0] - data['xt'][0]) < timedelta(minutes=101)
        
def test_cadence_filtering_192_minutes():
    filename = '/d0/magnetograms/hmi/preprocessed/hmi_stacks_2021_2022_96m_full.zarr'
    dataset = HMIZarrDataset(filename, months=[1], cadence=timedelta(minutes=192))
    data = dataset.getitem_times(0)
    assert (data['yt'][0] - data['xt'][0]) > timedelta(minutes=187)
    assert (data['yt'][0] - data['xt'][0]) < timedelta(minutes=197)

def test_seq_length_5():
    filename = '/d0/magnetograms/hmi/preprocessed/hmi_stacks_2021_2022_96m_full.zarr'
    dataset = HMIZarrDataset(filename, months=[1], x_seq_length=5, y_seq_length=5)
    data = dataset[0]
    data_times = dataset.getitem_times(0)
    assert data['x'].shape == (1, 5, 2048, 2048)
    assert data['y'].shape == (1, 5, 2048, 2048)
    assert (data_times['yt'][0] - data_times['xt'][0]) > (5*timedelta(minutes=96)-timedelta(minutes=5))
    assert (data_times['yt'][0] - data_times['xt'][0]) < (5*timedelta(minutes=96)+timedelta(minutes=5))
    assert (data_times['yt'][-1] - data_times['xt'][-1]) > (5*timedelta(minutes=96)-timedelta(minutes=5))
    assert (data_times['yt'][-1] - data_times['xt'][-1]) < (5*timedelta(minutes=96)+timedelta(minutes=5))
    assert data_times['xt'][-1] != data_times['yt'][0]

def test_get_item():
    filename = '/d0/magnetograms/hmi/preprocessed/hmi_stacks_2021_2022_96m_full.zarr'
    dataset = HMIZarrDataset(filename)
    data = dataset[0]
    assert data['x'].shape == (1, 1, 2048, 2048)
    assert data['y'].shape == (1, 1, 2048, 2048)
    
    data = dataset[100]
    assert data['x'].shape == (1, 1, 2048, 2048)
    assert data['y'].shape == (1, 1, 2048, 2048)

    print(dataset.t_obs[dataset.index[-1]])

