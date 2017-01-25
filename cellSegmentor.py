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
def detectBlobs(mask, cell_pad, minimum_cellsize, maximum_cellsize):
	# Morph. operations remove tiny blobs
	base_array, num_features = nd.label(mask)
	indices = np.unique(base_array, return_counts=True)
	vals = []
	cell_coords = []
	for i in xrange(1, len(indices[1])):
		if (indices[1][i] >= minimum_cellsize) & (indices[1][i] <= maximum_cellsize):
			vals.append(i)
	for entry in vals:
		labeled_array = np.zeros_like(base_array)
		labeled_array[base_array != entry] = 0
		labeled_array[base_array == entry] = 1
		cell_x_l = np.min(np.where(labeled_array == 1)[1]) - cell_pad
		cell_x_r = np.max(np.where(labeled_array == 1)[1]) + cell_pad
		cell_y_b = np.min(np.where(labeled_array == 1)[0]) - cell_pad
		cell_y_t = np.max(np.where(labeled_array == 1)[0]) + cell_pad
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

def printWholeImages(mask, ch1, ch2, ch3, ch4, ch1_m):
	print np.unique(mask)
	# fig = plt.figure()
	# a=fig.add_subplot(2,3,1)
	# imgplot = plt.imshow(ch1)
	# a.set_title('Ch1')
	# a=fig.add_subplot(2,3,2)
	# imgplot = plt.imshow(ch2)
	# a.set_title('Ch2')
	# a=fig.add_subplot(2,3,3)
	# imgplot = plt.imshow(ch3)
	# a.set_title('Ch3')
	# a=fig.add_subplot(2,3,4)
	# imgplot = plt.imshow(ch4)
	# a.set_title('Ch4')
	# a=fig.add_subplot(2,3,5)
	# imgplot = plt.imshow(ch1_m)
	# a.set_title('Ch1_Masked')
	# a=fig.add_subplot(2,3,6)
	# imgplot = plt.imshow(mask)
	# a.set_title('Mask')
	# plt.show()

def plotHistogram(x, min_x, max_x):
	# the histogram of the data
	n, bins, patches = plt.hist(x, 150)

	plt.xlabel('x and y widths')
	plt.ylabel('Frequency')
	plt.title(r'Width and Size Frequency Plot')
	plt.axis([min_x, max_x, None, None])
	plt.grid(False)

	plt.show()


def printSingleCroppedCells(ch1, ch2, ch3, ch4, ch1_m, mask, cell_coords):
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
		# plt.show()	

def detectLabel(cell, cell_mask):
	mean1 = np.mean(cell[0,0,:,:][cell_mask==1])
	mean2 = np.mean(cell[0,1,:,:][cell_mask==1])
	mean3 = np.mean(cell[0,2,:,:][cell_mask==1])
	mean4 = np.mean(cell[0,3,:,:][cell_mask==1])
	# mean4 = np.mean(cell[0,3,:,:][np.where(cell[0,3,:,:]>0)])


	# Check Zeros within cells

	# Neurons (+/+), CGRP positive, RIIb positive -> Red
	if (mean2 > 1800) & (mean3 > 700) & (mean4 < 1500):
		label = 0
	# +/-, CGRP positive, RIIb negative -> Green
	elif (mean2 > 1800) & (mean3 <= 700) & (mean4 < 1500):
		label = 1
	# -/+, CGRP negative, RIIb positive -> Blue
	elif (mean2 <= 1800) & (mean3 > 700) & (mean4 < 1500):
		label = 2
	# -/-, CGRP negative, RIIb negative -> Black
	elif (mean2 <= 1800) & (mean3 <= 700) & (mean4 < 1500):
		label = 3
	# outliers -> White
	elif mean4 >= 1500:
		label = 4
	else:
		print "Segmentation error"
		print mean1, mean2, mean3, mean4
	return np.array([label, mean1, mean2, mean3, mean4])


def storeData(ch1, ch2, ch3, ch4, mask, cell_coords, imagewidth):
	# somehow cells are turned 90 degrees
	ch1 = ch1 * mask
	ch2 = ch2 * mask
	ch3 = ch3 * mask
	ch4 = ch4 * mask
	cells_per_image = np.empty([0, 4, imagewidth,imagewidth])
	labels_per_image = np.empty([0, 5])
	for item in cell_coords:
		height = item["x_max"] - item["x_min"]
		width = item["y_max"] - item["y_min"]
		# listing, label, channels, means, pixels, pixels
		cell_container = np.zeros([1, 4, imagewidth, imagewidth])
		cell_mask = np.zeros([imagewidth, imagewidth])
		if (width > imagewidth) & (height <= imagewidth):
			cell_container[0, 0, :, 0:height] = cv2.resize(ch1[item["y_min"]:item["y_max"],item["x_min"]:item["x_max"]], (height, imagewidth), interpolation = cv2.INTER_LINEAR)
			cell_container[0, 1, :, 0:height] = cv2.resize(ch2[item["y_min"]:item["y_max"],item["x_min"]:item["x_max"]], (height, imagewidth), interpolation = cv2.INTER_LINEAR)
			cell_container[0, 2, :, 0:height] = cv2.resize(ch3[item["y_min"]:item["y_max"],item["x_min"]:item["x_max"]], (height, imagewidth), interpolation = cv2.INTER_LINEAR)
			cell_container[0, 3, :, 0:height] = cv2.resize(ch4[item["y_min"]:item["y_max"],item["x_min"]:item["x_max"]], (height, imagewidth), interpolation = cv2.INTER_LINEAR)
			cell_mask[:, 0:height] = cv2.resize(mask[item["y_min"]:item["y_max"],item["x_min"]:item["x_max"]], (height, imagewidth), interpolation = cv2.INTER_LINEAR)
		elif (width <= imagewidth) & (height > imagewidth):
			cell_container[0, 0, 0:width,:] = cv2.resize(ch1[item["y_min"]:item["y_max"],item["x_min"]:item["x_max"]], (imagewidth, width), interpolation = cv2.INTER_LINEAR)
			cell_container[0, 1, 0:width,:] = cv2.resize(ch2[item["y_min"]:item["y_max"],item["x_min"]:item["x_max"]], (imagewidth, width), interpolation = cv2.INTER_LINEAR)
			cell_container[0, 2, 0:width,:] = cv2.resize(ch3[item["y_min"]:item["y_max"],item["x_min"]:item["x_max"]], (imagewidth, width), interpolation = cv2.INTER_LINEAR)
			cell_container[0, 3, 0:width,:] = cv2.resize(ch4[item["y_min"]:item["y_max"],item["x_min"]:item["x_max"]], (imagewidth, width), interpolation = cv2.INTER_LINEAR)
			cell_mask[0:width,:] = cv2.resize(mask[item["y_min"]:item["y_max"],item["x_min"]:item["x_max"]], (imagewidth, width), interpolation = cv2.INTER_LINEAR)
		elif (width > imagewidth) & (height > imagewidth):
			cell_container[0, 0, :,:] = cv2.resize(ch1[item["y_min"]:item["y_max"],item["x_min"]:item["x_max"]], (imagewidth, imagewidth), interpolation = cv2.INTER_LINEAR)
			cell_container[0, 1, :,:] = cv2.resize(ch2[item["y_min"]:item["y_max"],item["x_min"]:item["x_max"]], (imagewidth, imagewidth), interpolation = cv2.INTER_LINEAR)
			cell_container[0, 2, :,:] = cv2.resize(ch3[item["y_min"]:item["y_max"],item["x_min"]:item["x_max"]], (imagewidth, imagewidth), interpolation = cv2.INTER_LINEAR)
			cell_container[0, 3, :,:] = cv2.resize(ch4[item["y_min"]:item["y_max"],item["x_min"]:item["x_max"]], (imagewidth, imagewidth), interpolation = cv2.INTER_LINEAR)
			cell_mask[:,:] = cv2.resize(mask[item["y_min"]:item["y_max"],item["x_min"]:item["x_max"]], (imagewidth, imagewidth), interpolation = cv2.INTER_LINEAR)
		else:
			cell_container[0, 0, 0:width, 0:height] = ch1[item["y_min"]:item["y_max"],item["x_min"]:item["x_max"]]
			cell_container[0, 1, 0:width, 0:height] = ch2[item["y_min"]:item["y_max"],item["x_min"]:item["x_max"]]
			cell_container[0, 2, 0:width, 0:height] = ch3[item["y_min"]:item["y_max"],item["x_min"]:item["x_max"]]
			cell_container[0, 3, 0:width, 0:height] = ch4[item["y_min"]:item["y_max"],item["x_min"]:item["x_max"]]
			cell_mask[0:width, 0:height] = mask[item["y_min"]:item["y_max"],item["x_min"]:item["x_max"]]
									
			# resize to fit boundary	

		y_temp = detectLabel(cell_container, cell_mask)

		# print y_temp
		# plt.imshow(cell_container[0, 0, :,:])
		# plt.show()
		cells_per_image = np.vstack((cells_per_image, cell_container))
		labels_per_image = np.vstack((labels_per_image, y_temp))
	return cells_per_image, labels_per_image

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







##### ------------- Settings ------------------ #####
##### ----------------------------------------- #####

# Zero Padding around the image
# padding = 0
padding = 10

# Opening Kernel size: Recommend 10
kernelsize = 10
# Cell cropping extension: Recommend 10
cell_pad = 10
# Cellsizes (32*32) and (128 * 128) 
minimum_cellwidth = 18
maximum_cellwidth = 128
imagewidth = 80

# mask dilation for cropping of final images: Recommend 5
dilation_coef = 5
# Process images until ith mask
max_mask = len(masks)-1








#### ------ ITERATION -------------------------- #####
#### ------------------------------------------- #####


cellSizes = []
cells = np.empty([0, 4, imagewidth, imagewidth])
labels = np.empty([0, 5])
# max_ch1 = 0
# max_ch2 = 0
# max_ch3 = 0
# max_ch4 = 0
# Looping over DIB masks and all channels to select image:
for i in masks[0:max_mask]:
	# Masks have to be point-reflected
	mask = binaryMask(plt.imread(i[:-6]+"o1.TIF"), padding, kernelsize)
	ch1 = plt.imread(i[:-6]+"d0.TIF")
	ch2 = plt.imread(i[:-6]+"d1.TIF")
	ch3 = plt.imread(i[:-6]+"d2.TIF")
	ch4 = plt.imread(i[:-6]+"d3.TIF")
	# ZeroPadding
	ch1 = np.lib.pad(ch1, padding, zeroPad)
	ch2 = np.lib.pad(ch2, padding, zeroPad)
	ch3 = np.lib.pad(ch3, padding, zeroPad)
	ch4 = np.lib.pad(ch4, padding, zeroPad)

	# Applying the mask
	ch1_m = ch1 * mask

	# # Setting new Maxima - Normalization
	# if temp_max_ch1 > max_ch1:
	# 	max_ch1 = temp_max_ch1
	# if temp_max_ch2 > max_ch2:
	# 	max_ch2 = temp_max_ch2
	# if temp_max_ch3 > max_ch3:
	# 	max_ch3 = temp_max_ch3
	# if temp_max_ch4 > max_ch4:
	# 	max_ch4 = temp_max_ch4 

	# Storing cell images in array
	cell_coords = detectBlobs(mask, cell_pad, minimum_cellwidth**2, 8000)
	# Mask dilation for final cropping
	mask = morphDil(mask, dilation_coef)
	cells_per_image, labels_per_image = storeData(ch1, ch2, ch3, ch4, mask, cell_coords, imagewidth)
	cells = np.vstack((cells, cells_per_image))	
	labels = np.vstack((labels, labels_per_image))
	if masks.index(i)%10 == 0:
		print "Image no.", masks.index(i)
		print "Number of used cells:", cells.shape[0]
	

	# printing images
	# printWholeImages(mask, ch1, ch2, ch3, ch4, ch1_m)
	# printSingleCroppedCells(ch1, ch2, ch3, ch4, ch1_m, mask, cell_coords)
	# cellSizes.append(cellSize(cell_coords))

print cells.shape
print labels.shape
uniques, frequency = np.unique(labels[:,0], return_counts=True)
print "Unique labels are:", uniques
print "Label Frequencies are:", frequency

print labels
# code.interact(local=dict(globals(), **locals()))
# Plott Cell Width frequency distribution
# frequencies = sum(cellSizes, [])
# plotHistogram(frequencies, minimum_cellwidth**2, 8000)

# np.save("/Volumes/MoritzBertholdHD/CellData/Experiments/Ex3/PreparedData/all_channels_80_80_full_no_zeros_in_cells", cells, allow_pickle=True, fix_imports=True)
# np.save("/Volumes/MoritzBertholdHD/CellData/Experiments/Ex3/PreparedData/labels_80_80_full_no_zeros_in_cells", labels.astype(int), allow_pickle=True, fix_imports=True)
