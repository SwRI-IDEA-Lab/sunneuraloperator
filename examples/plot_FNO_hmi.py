"""
Training a TFNO on HMI surface flux transport
=============================


"""

# %%
# 
import sys
sys.path.append('../')

import torch
import matplotlib.pyplot as plt
import sys
from neuralop.models import TFNO
from neuralop import Trainer
from neuralop.datasets.hmi_zarr_dataset import load_hmi_zarr
from neuralop.utils import count_params
from neuralop import LpLoss, H1Loss

device = 'cuda'


# %%
# Loading the HMI dataset in 128x128 resolution
train_loader, test_loaders, output_encoder = load_hmi_zarr(
        data_path='/d0/magnetograms/hmi/preprocessed/hmi_stacks_2021_2022_96m_full.zarr',
        batch_size=32, 
        train_resolution=128,
        test_resolutions=[128], 
        test_batch_sizes=[32],
        center_crop_size=64,
        train_months = (11,),
        test_months = (12,)
)


# %%
# We create a tensorized FNO model

model = TFNO(in_channels=1,n_modes=(16, 16), hidden_channels=32, projection_channels=64, factorization='tucker', rank=0.42)
model = model.to(device)

n_params = count_params(model)
print(f'\nOur model has {n_params} parameters.')
sys.stdout.flush()


# %%
#Create the optimizer
optimizer = torch.optim.Adam(model.parameters(), 
                                lr=8e-3, 
                                weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)


# %%
# Creating the losses
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)

train_loss = h1loss
eval_losses={'h1': h1loss, 'l2': l2loss}


# %%


print('\n### MODEL ###\n', model)
print('\n### OPTIMIZER ###\n', optimizer)
print('\n### SCHEDULER ###\n', scheduler)
print('\n### LOSSES ###')
print(f'\n * Train: {train_loss}')
print(f'\n * Test: {eval_losses}')
sys.stdout.flush()


# %% 
# Create the trainer
trainer = Trainer(model, n_epochs=3,
                  device=device,
                  mg_patching_levels=0,
                  wandb_log=False,
                  log_test_interval=1,
                  use_distributed=False,
                  verbose=True)


# %%
# Actually train the model on our HMI dataset

trainer.train(train_loader, test_loaders,
              output_encoder,
              model, 
              optimizer,
              scheduler, 
              regularizer=False, 
              training_loss=train_loss,
              eval_losses=eval_losses)


# %%
# Plot the prediction, and compare with the ground-truth 
# Note that we trained on a very small resolution for
# a very small number of epochs
# In practice, we would train at larger resolution, on many more samples.
# 
# However, for practicity, we created a minimal example that
# i) fits in just a few Mb of memory
# ii) can be trained quickly on CPU
#
# In practice we would train a Neural Operator on one or multiple GPUs

test_samples = test_loaders[128].dataset


cmap = 'gray'
vmin = torch.min(test_samples[0]['y'][:])
vmax = torch.max(test_samples[0]['y'][:])
fig = plt.figure(figsize=(7, 7))
sample_indices = [0, 100, 200] # list(range(0,len(test_samples),int(len(test_samples)/3)))

for index, sample_index in enumerate(sample_indices):
    data = test_samples[sample_index]
    # Input x
    x = data['x']
    # Ground-truth
    y = data['y']
    # Model prediction
    out = model(x.unsqueeze(0).to(device))

    ax = fig.add_subplot(3, 3, index*3 + 1)
    ax.imshow(x[0], cmap=cmap, vmin=vmin, vmax=vmax)
    if index == 0: 
        ax.set_title('Input x')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 3, index*3 + 2)
    ax.imshow(y.squeeze(), cmap=cmap, vmin=vmin, vmax=vmax)
    if index == 0: 
        ax.set_title('Ground-truth y')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 3, index*3 + 3)
    ax.imshow(out.squeeze().detach().cpu().numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
    if index == 0: 
        ax.set_title('Model prediction')
    plt.xticks([], [])
    plt.yticks([], [])

fig.suptitle('Inputs, ground-truth output and prediction.', y=0.98)
plt.tight_layout()
fig.show()

# %%
