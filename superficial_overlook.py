import numpy as np	
import os
import matplotlib.pyplot as plt
from scipy import ndimage as nd
import code

# Path to directory where images are stored
DIR = '/Volumes/MoritzBertholdHD/CellData/Experiments/Ex3/TIF_images/Ex3_ch-PGP_rb-CGRP_mo-RIIb/'

images_files = os.listdir(DIR) # use this for full dataset
print "# of image files, including DIBs and all channels:", len(images_files)

tifs = [DIR+i for i in images_files if '.TIF' in i]
print "# of TIFs:", len(tifs)

dibs = [DIR+i for i in images_files if '.DIB' in i]
masks = [DIR+i for i in images_files if 'o1.TIF' in i]
print "# of DIBs:", len(dibs)
print "# of Masks:", len(masks)
print "# of images:", (len(tifs)-len(masks))/4
print "# of images without masks:", (len(tifs)-len(masks))/4 - len(masks)
# If there is too much noise in the TIF, no mask can be produced.



# np.save("/Volumes/MoritzBertholdHD/CellData/Experiments/Ex3/PreparedData/all_channels_80_80_full_no_zeros_in_cells", cells, allow_pickle=True, fix_imports=True)
# np.save("/Volumes/MoritzBertholdHD/CellData/Experiments/Ex3/PreparedData/labels_80_80_full_no_zeros_in_cells", labels.astype(int), allow_pickle=True, fix_imports=True)
