from astropy.io import fits
import os
import glob
import zarr
from numcodecs import Blosc, Delta
from tqdm import tqdm
from sunpy.map import Map

PATCH_SIZE = 4096
HEADER_FIELD_IGNORE = ['keycomments', 'simple', 'history', 'comment', 'longstrn', 'headsum', 'license', 'primaryk', 'drms_id', 'recnum']


hmi_dir = '/d0/magnetograms/hmi/raw/2010_2023_1d'
filenames = sorted(glob.glob(hmi_dir+'/**.fits'))[0:2]
output_path = "/d0/magnetograms/hmi/preprocessed/fno/hmi_miniset.zarr"
store = zarr.DirectoryStore(output_path)
compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)
root = zarr.group(store=store, overwrite=True)
Blos = root.create_dataset('Blos', 
                           shape=(len(filenames), PATCH_SIZE, PATCH_SIZE), 
                           chunks=(15, None, None), 
                           dtype='f4',
                           compressor=compressor)

# open first file to obtain header keys
data = Map(filenames[0])
for key in data.meta:
    if key not in HEADER_FIELD_IGNORE:
        vars()[key] = []

for i, file in tqdm(enumerate(filenames),total=len(filenames)):
    try:
        # open file
        data = Map(file)
        for key in data.meta:
            if key not in HEADER_FIELD_IGNORE:
                vars()[key].append(data.meta[key])

        Blos[i, :, :] = data.data  
    except Exception as e:
        print(e)

for key in data.meta:
    if key not in HEADER_FIELD_IGNORE:
        Blos.attrs[key.lower()]=vars()[key]

