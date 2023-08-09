from astropy.io import fits
import os
import zarr
from numcodecs import Blosc, Delta

PATCH_SIZE = 2048

hmi_dir = '/d0/magnetograms/hmi/raw/2010_2023_1d'
filenames = sorted(os.listdir(hmi_dir))
output_path = "/d0/magnetograms/hmi/preprocesed/fno/hmi_center_crop.zarr"
store = zarr.DirectoryStore(output_path)
compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)
root = zarr.group(store=store, overwrite=True)
Blos = root.create_dataset('Blos', 
                           shape=(len(filenames), PATCH_SIZE, PATCH_SIZE), 
                           chunks=(15, None, None), 
                           dtype='f4',
                           compressor=compressor)

for i, file in enumerate(filenames):
    try:
        # open file
        with fits.open(os.path.join(hmi_dir, file), cache=False) as data_fits:
            data_fits.verify('fix')

            img = data_fits[1].data
            header = data_fits[1].header
            Blos[i, :, :] = img  # TODO: crop out the patch
    except Exception as e:
        print(e)

zarr.save('data/example.zarr', store)  # TODO: is this right?