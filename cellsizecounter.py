import numpy as np	
import os
import matplotlib.pyplot as plt
from scipy import ndimage as nd
import cv2
import code


def binaryMask(mask, padding=0, kernelsize = 10):
	# print np.unique(mask, return_counts=True)
	# print np.unique(mask)
	mask[mask >= 1] = 1
	mask[mask < 1] = 0
	mask = np.transpose(mask)
	mask = 	np.lib.pad(mask, padding, zeroPad)
	mask = morphOp(mask, kernelsize)
	# plt.figure()
	# plt.imshow(mask)
	# plt.show()

	# print np.unique(mask)
	# mask[mask >= 1] = 1
	# mask[mask < 1] = 0
	# mask = np.transpose(mask)
	return mask

def morphOp(mask, kernelsize):
	kernel = np.ones((kernelsize, kernelsize),np.uint8)
	opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations = 1)
	# fig = plt.figure()
	# a=fig.add_subplot(1,2,1)
	# imgplot = plt.imshow(mask)
	# a.set_title('Normal')
	# a=fig.add_subplot(1,2,2)
	# imgplot = plt.imshow(opening)
	# a.set_title('Opening')
	# plt.show()
	return opening

def morphDil(mask, kernelsize):
	kernel = np.ones((kernelsize, kernelsize),np.uint8)
	dilation = cv2.dilate(mask, kernel, iterations = 1)
	return dilation

	dilation = cv2.dilate(img,kernel,iterations = 1)

def detectBlobs(mask, minimum_cellsize):
	# Morph. operations remove tiny blobs
	base_array, num_features = nd.label(mask)
	indices = np.unique(base_array, return_counts=True)
	vals = []
	cell_coords = []
	for i in xrange(1, len(indices[1])):
		if (indices[1][i] >= minimum_cellsize):
			# code.interact(local=dict(globals(), **locals()))
			vals.append(i)

	for entry in vals:
		labeled_array = np.zeros_like(base_array)
		labeled_array[base_array != entry] = 0
		labeled_array[base_array == entry] = 1
		cell_x_l = np.min(np.where(labeled_array == 1)[1])
		cell_x_r = np.max(np.where(labeled_array == 1)[1])		
		cell_y_b = np.min(np.where(labeled_array == 1)[0])
		cell_y_t = np.max(np.where(labeled_array == 1)[0])
		coordinates = {'x_min': cell_x_l, 'x_max': cell_x_r, 'y_min': cell_y_b, 'y_max': cell_y_t}
		cell_coords.append(coordinates)
	return cell_coords

def zeroPad(vector, pad_width, iaxis, kwargs):
    vector[:pad_width[0]] = 0
    vector[-pad_width[1]:] = 0
    return vector

def cellSize(cell_coords):
	sizes = []
	for i in xrange(len(cell_coords)):
		x_range = cell_coords[i]['x_max'] - cell_coords[i]['x_min']
		y_range = cell_coords[i]['y_max'] - cell_coords[i]['y_min']
		sizes.append(x_range * y_range)
		# sizes.append(y_range)
	return sizes


def plotHistogram(x, min_x, max_x):
	# the histogram of the data
	n, bins, patches = plt.hist(x, 50)

	plt.xlabel('x and y widths')
	plt.ylabel('Frequency')
	plt.title(r'Width and Size Frequency Plot')
	plt.axis([min_x, max_x, None, None])
	plt.grid(False)

	plt.show()



# Path to directory where images are stored
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


##### ------------- Settings here: ------------ #####
##### ----------------------------------------- #####

# Zero Padding around the image
padding = 10
# Opening Kernel size: Recommend 10
kernelsize = 10
# # Cell cropping extension: Recommend 10
# cell_pad = 10
# Cellsizes (32*32) and (128 * 128) 
minimum_cellwidth = 18
maximum_cellwidth = 128
minimum_cellsize = 124
imagewidth = 80

# mask dilation for cropping of final images: Recommend 5
dilation_coef = 5
# Process images until ith mask
max_mask = 200






#### ------ ITERATION -------------------------- #####
#### ------------------------------------------- #####


cellWidths = []
cellHeigths = []
cellSizes = []
# Looping over DIB masks and all channels to select image:
for i in masks[0:max_mask]:
	# Masks have to be point-reflected
	mask = binaryMask(plt.imread(i[:-6]+"o1.TIF"), padding, kernelsize)
	cell_coords = detectBlobs(mask, minimum_cellsize)
	cellSizes.append(cellSize(cell_coords))

# Plott Cell Width frequency distribution
plotHistogram(cellSizes, 124, 8000)
# code.interact(local=dict(globals(), **locals()))
