# Notes

## Environment Installation

`conda env create -f environment.yml`

## Make sure that Wandb config are in a writable folder
`export WANDB_CONFIG_DIR=/home/amunozj/tmp`

## TODOs

1. Make `__getitems` in the loader work...
2. Convert to lightning
3. Revise colormap, if grayscale is not giving enough info
4. Expand dataloader to return sequences.
5. Expand datalodader to return random crops.

## Last thing done
Fixing and expanding the tests to reflect the new indexing paradigm.  Check that x and y don't overlap
