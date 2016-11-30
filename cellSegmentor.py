import numpy as np	
import os
import matplotlib.pyplot as plt
from scipy import ndimage as nd

import code
def binaryMask(mask):
	print np.unique(mask, return_counts=True)
	print np.unique(mask)
	mask[mask >= 1] = 1
	mask[mask < 1] = 0
	mask = np.transpose(mask)
	plt.figure()
	plt.imshow(mask)
	plt.show()
	# print np.unique(mask)
	# mask[mask >= 1] = 1
	# mask[mask < 1] = 0
	# mask = np.transpose(mask)
	return mask

def detectBlobs(mask):
	base_array, num_features = nd.label(mask)
	indices = np.unique(base_array, return_counts=True)
	threshold_size = 256
	vals = []
	cell_coords = []
	for i in xrange(1, len(indices[1])):
		if indices[1][i] > threshold_size:
			vals.append(i)
	for entry in vals:
		labeled_array = np.zeros_like(base_array)
		labeled_array[base_array != entry] = 0
		labeled_array[base_array == entry] = 1
		padding = 20
		cell_x_l = np.min(np.where(labeled_array == 1)[1])
		cell_x_r = np.max(np.where(labeled_array == 1)[1])
		cell_y_b = np.min(np.where(labeled_array == 1)[0])
		cell_y_t = np.max(np.where(labeled_array == 1)[0])
		coordinates = {'x_min': cell_x_l, 'x_max': cell_x_r, 'y_min': cell_y_b, 'y_max': cell_y_t}
		cell_coords.append(coordinates)
	return cell_coords	

def printWholeImages(mask, ch1, ch2, ch3, ch4, ch1_m, ch2_m, ch3_m, ch4_m):
	print np.unique(mask)
	fig = plt.figure()
	a=fig.add_subplot(2,3,1)
	imgplot = plt.imshow(ch1)
	a.set_title('Ch1')
	a=fig.add_subplot(2,3,2)
	imgplot = plt.imshow(ch2)
	a.set_title('Ch2')
	a=fig.add_subplot(2,3,3)
	imgplot = plt.imshow(ch3)
	a.set_title('Ch3')
	a=fig.add_subplot(2,3,4)
	imgplot = plt.imshow(ch4)
	a.set_title('Ch4')
	a=fig.add_subplot(2,3,5)
	imgplot = plt.imshow(ch1_m)
	a.set_title('Ch1_Masked')
	a=fig.add_subplot(2,3,6)
	imgplot = plt.imshow(mask)
	a.set_title('Mask')
	plt.show()

def printSingleCroppedCells(ch1_m, ch2_m, ch3_m, ch4_m, cell_coords):
	# print "# of cells in image", len(cell_coords)
	for i in xrange(len(cell_coords)):
		cell = cell_coords[i]
		fig = plt.figure()
		a=fig.add_subplot(2,3,1)
		imgplot = plt.imshow(ch1[cell["y_min"]:cell["y_max"],cell["x_min"]:cell["x_max"]])
		a.set_title('Ch1')
		a=fig.add_subplot(2,3,2)
		imgplot = plt.imshow(ch2[cell["y_min"]:cell["y_max"],cell["x_min"]:cell["x_max"]])
		a.set_title('Ch2')
		a=fig.add_subplot(2,3,3)
		imgplot = plt.imshow(ch3[cell["y_min"]:cell["y_max"],cell["x_min"]:cell["x_max"]])
		a.set_title('Ch3')
		a=fig.add_subplot(2,3,4)
		imgplot = plt.imshow(ch4[cell["y_min"]:cell["y_max"],cell["x_min"]:cell["x_max"]])
		a.set_title('Ch4')
		a=fig.add_subplot(2,3,5)
		imgplot = plt.imshow(ch1_m[cell["y_min"]:cell["y_max"],cell["x_min"]:cell["x_max"]])
		a.set_title('Ch1_Masked')
		a=fig.add_subplot(2,3,6)
		imgplot = plt.imshow(mask[cell["y_min"]:cell["y_max"],cell["x_min"]:cell["x_max"]])
		a.set_title('Mask')
		plt.show()	


DIR = '/Volumes/MoritzBertholdHD/CellData/Experiments/Ex1/TIF_images/Ex1_ch-PGP_rb-CGRP_mo-RIIb/'

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


# Looping over DIB masks and all channels to select image:
for i in masks[0:30]:
	mask = binaryMask(plt.imread(i[:-6]+"o1.TIF"))
	# Mask have to be point-reflected
	ch1 = plt.imread(i[:-6]+"d0.TIF")
	ch2 = plt.imread(i[:-6]+"d1.TIF")
	ch3 = plt.imread(i[:-6]+"d2.TIF")
	ch4 = plt.imread(i[:-6]+"d3.TIF")
	ch1_m = ch1 * mask
	ch2_m = ch2 * mask
	ch3_m = ch3 * mask
	ch4_m = ch4 * mask
	cell_coords = detectBlobs(mask)

	# printWholeImages(mask, ch1, ch2, ch3, ch4, ch1_m, ch2_m, ch3_m, ch4_m)

	# printSingleCroppedCells(ch1_m, ch2_m, ch3_m, ch4_m, cell_coords)

plt.show()

# plt.hist(mask.ravel(), bins=50, range=(0.0, 41), fc='k', ec='k')



