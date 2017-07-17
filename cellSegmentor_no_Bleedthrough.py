import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import ndimage as nd
import cv2
import code
import re
import os.path
# replacin DAPI with ch1 bleed through
plot = True

def binaryMask(mask, padding=0, kernelsize = 10):
	# print np.unique(mask, return_counts=True)
	# print np.unique(mask)
	mask[mask >= 1] = 1
	mask[mask < 1] = 0
	mask = np.transpose(mask)
	mask = 	np.lib.pad(mask, padding, zeroPad)
	mask = morphOp(mask, kernelsize)
	# if plot:
		# plt.figure()
		# plt.imshow(mask, cmap='gray')
		# plt.show()

	# print np.unique(mask)
	# mask[mask >= 1] = 1
	# mask[mask < 1] = 0
	# mask = np.transpose(mask)
	return mask

def morphOp(mask, kernelsize):
	kernel = np.ones((kernelsize, kernelsize),np.uint8)
	opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations = 1)
	# if plot:
		# fig = plt.figure()
		# a=fig.add_subplot(1,2,1)
		# imgplot = plt.imshow(mask, cmap='gray')
		# a.set_title('Normal')
		# a=fig.add_subplot(1,2,2)
		# imgplot = plt.imshow(opening, cmap='gray')
		# a.set_title('Opening')
		# plt.show()

	return opening

def morphDil(mask, kernelsize):
	kernel = np.ones((kernelsize, kernelsize),np.uint8)
	dilation = cv2.dilate(mask, kernel, iterations = 1)
	return dilation

def channelShifter(ch, xoffset, yoffset):
	xabsset = abs(xoffset)
	yabsset = abs(yoffset)
	padded_ch = np.pad(ch, ((yabsset, yabsset), (xabsset, xabsset)), "minimum")
	shifted_ch = padded_ch[yabsset + yoffset:padded_ch.shape[-0]-yabsset+yoffset, xabsset + xoffset:padded_ch.shape[-1]-xabsset+xoffset]
	return shifted_ch

def shiftIndicesByOffset(base_array, indices, xoffset, yoffset):
	xabsset = abs(xoffset)
	yabsset = abs(yoffset)
	padded_array = np.pad(base_array, ((yabsset, yabsset), (xabsset, xabsset)), "minimum")
	shifted_array = padded_array[yabsset + yoffset:padded_array.shape[-0]-yabsset+yoffset, xabsset + xoffset:padded_array.shape[-1]-xabsset+xoffset]
	border_array = np.append(shifted_array[:,0], np.append(shifted_array[:,-1], np.append(shifted_array[0,:], shifted_array[-1,:])))
	uniques, counts = np.unique(border_array, return_counts=True)
	discard_count = 0
	for i in indices[0]:
		if i == 0: continue
		if len(uniques) <= i: break
		if counts[i] >= 1:
			# code.interact(local=dict(globals(), **locals()))
			shifted_array[shifted_array == uniques[i]] = 0
			discard_count += 1
	return shifted_array, discard_count

def adjust(base_array, mask_test):
	binary_base_array = base_array
	binary_base_array[binary_base_array>=1]=1
	overlay = binary_base_array - mask_test
	fig = plt.figure()
	a=fig.add_subplot(1,3,1)
	imgplot = plt.imshow(binary_base_array, cmap='gray')
	a.set_title('New Mask')
	a=fig.add_subplot(1,3,2)
	imgplot = plt.imshow(mask_test, cmap='gray')
	a.set_title('Clean Mask')
	a=fig.add_subplot(1,3,3)
	imgplot = plt.imshow(overlay, cmap='gray')
	a.set_title('Overlay')
	plt.show()

def onlyadjust(base_array, mask_test):
	binary_base_array = base_array.copy()
	binary_base_array[binary_base_array>=1]=1
	overlay = binary_base_array - mask_test
	fig = plt.figure()
	plt.imshow(overlay, cmap='gray')
	plt.show()

def detectBlobs(mask, mask_test, xoffset, yoffset, cell_pad, minimum_cellsize, maximum_cellsize):
	# Morph. operations remove tiny blobs
	base_array, num_features = nd.label(mask)

	indices = np.unique(base_array, return_counts=True)
	base_array, discard_count = shiftIndicesByOffset(base_array, indices, xoffset, yoffset)
	indices = np.unique(base_array, return_counts=True)
	# printMaskandChannel(mask, base_array, mask_test)
	# shift indices by offset
	check_array, num_cells = nd.label(mask_test)

	vals = []
	cell_coords = []
	cell_vectors = []
	new_xoffset = 0
	new_yoffset = 0
	for i in xrange(1, len(indices[1]-1)):
		if (indices[1][i] >= minimum_cellsize) & (indices[1][i] <= maximum_cellsize):
			vals.append(indices[0][i])
	for entry in vals:
		labeled_array = np.zeros_like(base_array)
		calc_distance_array = np.zeros_like(base_array)
		new_check_array = np.zeros_like(check_array)
		labeled_array[base_array != entry] = 0
		labeled_array[base_array == entry] = 1
		calc_distance_array = mask_test * labeled_array
		numbers, cnts = np.unique(check_array * calc_distance_array, return_counts=True)
		if len(cnts) <=1:
			pass
		else:
			new_check_array[check_array != numbers[np.argmax(cnts[1:])+1]] = 0
			new_check_array[check_array == numbers[np.argmax(cnts[1:])+1]] = 1

			cell_x_l = np.min(np.where(labeled_array == 1)[1]) - cell_pad
			cell_x_r = np.max(np.where(labeled_array == 1)[1]) + cell_pad
			cell_y_b = np.min(np.where(labeled_array == 1)[0]) - cell_pad
			cell_y_t = np.max(np.where(labeled_array == 1)[0]) + cell_pad
			coordinates = {'x_min': cell_x_l, 'x_max': cell_x_r, 'y_min': cell_y_b, 'y_max': cell_y_t}
			cell_coords.append(coordinates)
			x_center = (cell_x_r - cell_x_l)/2 + cell_x_l
			y_center = (cell_y_t - cell_y_b)/2 + cell_y_b
			x_center_check = (np.max(np.where(new_check_array == 1)[1]) - np.min(np.where(new_check_array == 1)[1]))/2 + np.min(np.where(new_check_array == 1)[1])
			y_center_check = (np.max(np.where(new_check_array == 1)[0]) - np.min(np.where(new_check_array == 1)[0]))/2 + np.min(np.where(new_check_array == 1)[0])
			cell_vectors.append([x_center, y_center, x_center_check, y_center_check])
			# code.interact(local=dict(globals(), **locals()))
	# print cell_vectors
	if len(cell_vectors) > 2:
		cell_distances = [((i[0]-i[2])**2 + (i[1]-i[3])**2)**.5 for i in cell_vectors]
		vector_delta = cell_vectors[np.argmin(cell_distances)]
		new_xoffset = vector_delta[0] - vector_delta[2]
		new_yoffset = vector_delta[1] - vector_delta[3]
		#print "Vector_delta: ", vector_delta
		#print "Offset: ", (new_xoffset**2 + new_yoffset**2)**.5
		#print "new_xoffset ", new_xoffset
		#print "new_yoffset ", new_yoffset
	base_array[base_array > 0]=1
	onlyadjust(base_array, mask_test)
	return cell_coords, discard_count, base_array, new_xoffset, new_yoffset

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
	# print np.unique(mask)
	if plot:
		fig = plt.figure()
		a=fig.add_subplot(2,3,1)
		imgplot = plt.imshow(ch1, cmap='gray')
		a.set_title('Ch1')
		a=fig.add_subplot(2,3,2)
		imgplot = plt.imshow(ch2, cmap='gray')
		a.set_title('Ch2')
		a=fig.add_subplot(2,3,3)
		imgplot = plt.imshow(ch3, cmap='gray')
		a.set_title('Ch3')
		a=fig.add_subplot(2,3,4)
		imgplot = plt.imshow(ch4, cmap='gray')
		a.set_title('Ch4')
		a=fig.add_subplot(2,3,5)
		imgplot = plt.imshow(mask, cmap='gray')
		a.set_title('Mask')
		a=fig.add_subplot(2,3,6)
		imgplot = plt.imshow(ch1_m, cmap='gray')
		a.set_title('Ch1 + Mask')
		plt.show()

def plotHistogram(x, min_x, max_x):
	# the histogram of the data
	n, bins, patches = plt.hist(x, 150)

	plt.xlabel('x and y widths')
	plt.ylabel('Frequency')
	plt.title(r'Width and Size Frequency Plot')
	plt.axis([min_x, max_x, None, None])
	plt.grid(False)

	# plt.show()


def printSingleCroppedCells(ch1, ch2, ch3, ch4, ch1_m, mask, cell_coords):
	# print "# of cells in image", len(cell_coords)
	for i in xrange(len(cell_coords)):
		cell = cell_coords[i]
		if plot:
			fig = plt.figure()
			a=fig.add_subplot(2,3,1)
			imgplot = plt.imshow(ch1[cell["y_min"]:cell["y_max"],cell["x_min"]:cell["x_max"]], cmap='gray')
			a.set_title('Ch1')
			a=fig.add_subplot(2,3,2)
			imgplot = plt.imshow(ch2[cell["y_min"]:cell["y_max"],cell["x_min"]:cell["x_max"]], cmap='gray')
			a.set_title('Ch2')
			a=fig.add_subplot(2,3,3)
			imgplot = plt.imshow(ch3[cell["y_min"]:cell["y_max"],cell["x_min"]:cell["x_max"]], cmap='gray')
			a.set_title('Ch3')
			a=fig.add_subplot(2,3,4)
			imgplot = plt.imshow(ch4[cell["y_min"]:cell["y_max"],cell["x_min"]:cell["x_max"]], cmap='gray')
			a.set_title('Ch4')
			a=fig.add_subplot(2, 3, 5)
			imgplot = plt.imshow(mask[cell["y_min"]:cell["y_max"],cell["x_min"]:cell["x_max"]], cmap='gray')
			a.set_title('Mask')
			a=fig.add_subplot(2, 3, 6)
			imgplot = plt.imshow(ch1_m[cell["y_min"]:cell["y_max"],cell["x_min"]:cell["x_max"]], cmap='gray')
			a.set_title('Ch1 + Mask')
			plt.show()

def printMaskandChannel(mask, shifted_mask, test_mask):
	# if plot:
	fig = plt.figure()
	a=fig.add_subplot(1,3,1)
	imgplot = plt.imshow(mask, cmap='gray')
	a.set_title('Old Mask')
	a=fig.add_subplot(1,3,2)
	imgplot = plt.imshow(shifted_mask, cmap='gray')
	a.set_title('Shifted Mask')
	a=fig.add_subplot(1,3,3)
	imgplot = plt.imshow(test_mask, cmap='gray')
	a.set_title('Test Mask (PGP-clean)')
	# plt.show()
	# # if plot:
	# fig = plt.figure()
	# a=fig.add_subplot(1,2,1)
	# imgplot = plt.imshow(ch1, cmap='gray')
	# a.set_title('Ch1 Image')
	# a=fig.add_subplot(1,2,2)
	# imgplot = plt.imshow(mask, cmap='gray')
	# a.set_title('Mask')
	# plt.show()

def detectLabel(cell, cell_mask):
	mean1 = np.mean(cell[0,0,:,:][cell_mask==1])
	mean2 = np.mean(cell[0,1,:,:][cell_mask==1])
	mean3 = np.mean(cell[0,2,:,:][cell_mask==1])
	mean4 = np.mean(cell[0,3,:,:][cell_mask==1])
# 	mean5 = np.mean(cell[0,4,:,:][cell_mask==1])
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
	# ch1_b = ch1_b * mask
	cells_per_image = np.empty([0, 4, imagewidth,imagewidth])
	labels_per_image = np.empty([0, 5])
	for item in cell_coords:
		height = item["x_max"] - item["x_min"]
		width = item["y_max"] - item["y_min"]
		# listing, label, channels, means, pixels, pixels
		cell_container = np.zeros([1, 4, imagewidth, imagewidth])
		cell_mask = np.zeros([imagewidth, imagewidth])
		# print item
		if (width > imagewidth) & (height <= imagewidth):
			cell_container[0, 0, :, 0:height] = cv2.resize(ch1[item["y_min"]:item["y_max"],item["x_min"]:item["x_max"]], (height, imagewidth), interpolation = cv2.INTER_LINEAR)
			cell_container[0, 1, :, 0:height] = cv2.resize(ch2[item["y_min"]:item["y_max"],item["x_min"]:item["x_max"]], (height, imagewidth), interpolation = cv2.INTER_LINEAR)
			cell_container[0, 2, :, 0:height] = cv2.resize(ch3[item["y_min"]:item["y_max"],item["x_min"]:item["x_max"]], (height, imagewidth), interpolation = cv2.INTER_LINEAR)
			cell_container[0, 3, :, 0:height] = cv2.resize(ch4[item["y_min"]:item["y_max"],item["x_min"]:item["x_max"]], (height, imagewidth), interpolation = cv2.INTER_LINEAR)
			# cell_container[0, 4, :, 0:height] = cv2.resize(ch1_b[item["y_min"]:item["y_max"],item["x_min"]:item["x_max"]], (height, imagewidth), interpolation = cv2.INTER_LINEAR)
			cell_mask[:, 0:height] = cv2.resize(mask[item["y_min"]:item["y_max"],item["x_min"]:item["x_max"]], (height, imagewidth), interpolation = cv2.INTER_LINEAR)
		elif (width <= imagewidth) & (height > imagewidth):
			cell_container[0, 0, 0:width,:] = cv2.resize(ch1[item["y_min"]:item["y_max"],item["x_min"]:item["x_max"]], (imagewidth, width), interpolation = cv2.INTER_LINEAR)
			cell_container[0, 1, 0:width,:] = cv2.resize(ch2[item["y_min"]:item["y_max"],item["x_min"]:item["x_max"]], (imagewidth, width), interpolation = cv2.INTER_LINEAR)
			cell_container[0, 2, 0:width,:] = cv2.resize(ch3[item["y_min"]:item["y_max"],item["x_min"]:item["x_max"]], (imagewidth, width), interpolation = cv2.INTER_LINEAR)
			cell_container[0, 3, 0:width,:] = cv2.resize(ch4[item["y_min"]:item["y_max"],item["x_min"]:item["x_max"]], (imagewidth, width), interpolation = cv2.INTER_LINEAR)
			# cell_container[0, 4, 0:width,:] = cv2.resize(ch1_b[item["y_min"]:item["y_max"],item["x_min"]:item["x_max"]], (imagewidth, width), interpolation = cv2.INTER_LINEAR)
			cell_mask[0:width,:] = cv2.resize(mask[item["y_min"]:item["y_max"],item["x_min"]:item["x_max"]], (imagewidth, width), interpolation = cv2.INTER_LINEAR)
		elif (width > imagewidth) & (height > imagewidth):
			cell_container[0, 0, :,:] = cv2.resize(ch1[item["y_min"]:item["y_max"],item["x_min"]:item["x_max"]], (imagewidth, imagewidth), interpolation = cv2.INTER_LINEAR)
			cell_container[0, 1, :,:] = cv2.resize(ch2[item["y_min"]:item["y_max"],item["x_min"]:item["x_max"]], (imagewidth, imagewidth), interpolation = cv2.INTER_LINEAR)
			cell_container[0, 2, :,:] = cv2.resize(ch3[item["y_min"]:item["y_max"],item["x_min"]:item["x_max"]], (imagewidth, imagewidth), interpolation = cv2.INTER_LINEAR)
			cell_container[0, 3, :,:] = cv2.resize(ch4[item["y_min"]:item["y_max"],item["x_min"]:item["x_max"]], (imagewidth, imagewidth), interpolation = cv2.INTER_LINEAR)
			# cell_container[0, 4, :,:] = cv2.resize(ch1_b[item["y_min"]:item["y_max"],item["x_min"]:item["x_max"]], (imagewidth, imagewidth), interpolation = cv2.INTER_LINEAR)
			cell_mask[:,:] = cv2.resize(mask[item["y_min"]:item["y_max"],item["x_min"]:item["x_max"]], (imagewidth, imagewidth), interpolation = cv2.INTER_LINEAR)
		else:
			cell_container[0, 0, 0:width, 0:height] = ch1[item["y_min"]:item["y_max"],item["x_min"]:item["x_max"]]
			cell_container[0, 1, 0:width, 0:height] = ch2[item["y_min"]:item["y_max"],item["x_min"]:item["x_max"]]
			cell_container[0, 2, 0:width, 0:height] = ch3[item["y_min"]:item["y_max"],item["x_min"]:item["x_max"]]
			cell_container[0, 3, 0:width, 0:height] = ch4[item["y_min"]:item["y_max"],item["x_min"]:item["x_max"]]
			# cell_container[0, 4, 0:width, 0:height] = ch1_b[item["y_min"]:item["y_max"],item["x_min"]:item["x_max"]]
			cell_mask[0:width, 0:height] = mask[item["y_min"]:item["y_max"],item["x_min"]:item["x_max"]]

			# resize to fit boundary

		y_temp = detectLabel(cell_container, cell_mask)

		# print y_temp
		# if plot:
			# plt.imshow(cell_container[0, 0, :,:])
			# plt.show()
		cells_per_image = np.vstack((cells_per_image, cell_container))
		labels_per_image = np.vstack((labels_per_image, y_temp))
	return cells_per_image, labels_per_image


ex = 3

# Path to directory where images are stored
DIR = '/Volumes/MoritzBertholdHD/CellData/Experiments/Ex' + str(ex) + '/TIF_images/Ex' + str(ex) + '_ch-PGP_rb-CGRP_mo-RIIb/'
DIR2 = '/Volumes/MoritzBertholdHD/CellData/Experiments/Ex' + str(ex) + '/TIF_images/Ex' + str(ex) + '_ch-PGP/'
images_files2 = os.listdir(DIR2) # use this for full dataset
tifs2 = [DIR2+i for i in images_files2 if '.TIF' in i]
dibs2 = [DIR2+i for i in images_files2 if '.DIB' in i]
masks2 = [DIR2+i for i in images_files2 if 'o1.TIF' in i]

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

# Opening Kernel size: Recommend 10
kernelsize = 10
# Cell cropping extension: Recommend 10
cell_pad = 10
# Cellsizes (32*32) and (128 * 128)
minimum_cellwidth = 18
maximum_cellwidth = 128
imagewidth = 66

# mask dilation for cropping of final images: Recommend 5
dilation_coef = 5
# Process images until ith mask
max_mask = len(masks)-1

#### ------ ITERATION -------------------------- #####
#### ------------------------------------------- #####


cellSizes = []
cells = np.empty([0, 4, imagewidth, imagewidth])
labels = np.empty([0, 5])

if ex ==1:
	xoffset = -3
	yoffset = -12
if ex ==2:
	xoffset = -122
	yoffset = 2
if ex ==3:
	xoffset = 70
	yoffset = 10
	# # second_half:
	# xoffset = 40
	# yoffset = 27

faults = 0
final_discard_count = 0
for j in xrange(0,max_mask):
# a = [0,200,1000, 1001, 2000, 2001, 3000, 3001]
# for j in xrange(0, max_mask/2):
# for j in xrange(, max_mask):
	i = masks[j]
	# Masks have to be point-reflected
	# replace string for ch1_b
	i_b = re.sub('_rb-CGRP_mo-RIIb', '', i)
	if ex ==1:
		i_b = re.sub('161020140001', '161018170001', i_b)
	if ex ==2:
		i_b = re.sub('161020170001', '161018200001', i_b)
	if ex ==3:
		i_b = re.sub('161020200001', '161018220001', i_b)
	# check if file exists:
	if os.path.isfile(i_b)==False:
		faults +=1
		continue

	# Zero Padding around the image
	# padding = 0
	padding = np.max([abs(xoffset), abs(yoffset)])+cell_pad + kernelsize + 1

	mask_test = binaryMask(plt.imread(i_b[:-6]+"o1.TIF"), padding, kernelsize=0)
	mask = binaryMask(plt.imread(i[:-6]+"o1.TIF"), padding, kernelsize)
	cell_coords, discard_count, mask, new_xoffset, new_yoffset = detectBlobs(mask, mask_test, xoffset, yoffset, cell_pad, minimum_cellwidth**2, 8000)
	if new_xoffset <= 20 and masks.index(i)%100 == 0:
		xoffset += new_xoffset
		print "Setting new x ", xoffset
	if new_yoffset <= 20 and masks.index(i)%100 == 0:
		yoffset += new_yoffset
		print "Setting new y ", yoffset
	# code.interact(local=dict(globals(), **locals()))

	# ch1 = plt.imread(i[:-6]+"d0.TIF")
	ch1 = plt.imread(i_b[:-6]+"d0.TIF")
	ch2 = plt.imread(i[:-6]+"d1.TIF")
	ch3 = plt.imread(i[:-6]+"d2.TIF")
	ch4 = plt.imread(i[:-6]+"d3.TIF")
	# ZeroPadding
	ch1 = np.lib.pad(ch1, padding, zeroPad)
	ch2 = channelShifter(np.lib.pad(ch2, padding, zeroPad), xoffset, yoffset)
	ch3 = channelShifter(np.lib.pad(ch3, padding, zeroPad), xoffset, yoffset)
	ch4 = channelShifter(np.lib.pad(ch4, padding, zeroPad), xoffset, yoffset)

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



			# fig = plt.figure()
			# a=fig.add_subplot(2,3,1)
			# imgplot = plt.imshow(ch1a, cmap='gray')
			# a.set_title('Ch1 dirty')
			# a=fig.add_subplot(2,3,2)
			# imgplot = plt.imshow(ch1b, cmap='gray')
			# a.set_title('Ch1 clean')
			# a=fig.add_subplot(2,3,3)
			# imgplot = plt.imshow(ch2, cmap='gray')
			# a.set_title('Shifted ch4')
			# a=fig.add_subplot(2,3,4)
			# imgplot = plt.imshow(mask, cmap='gray')
			# a.set_title('Dirty Mask')
			# a=fig.add_subplot(2,3,5)
			# imgplot = plt.imshow(mask_test, cmap='gray')
			# a.set_title('Clean Mask')
			# a=fig.add_subplot(2,3,6)
			# imgplot = plt.imshow(mask_, cmap='gray')
			# a.set_title('new_mask')
			# plt.show()

	# Storing cell images in array

	final_discard_count += discard_count
	# Mask dilation for final cropping
	mask = np.array(mask, dtype="uint16")

	mask = morphDil(mask, dilation_coef)
	# code.interact(local=dict(globals(), **locals()))


	cells_per_image, labels_per_image = storeData(ch1, ch2, ch3, ch4, mask, cell_coords, imagewidth)
	cells = np.vstack((cells, cells_per_image))
	labels = np.vstack((labels, labels_per_image))
	if masks.index(i)%100 == 0:
		print "Mask no.", masks.index(i)
		print "Number of used cells:", cells.shape[0]
		print "Total no. of masks", len(masks)


	# printing images
	if plot:
		# printMaskandChannel(mask,mask_old)
		# printWholeImages(mask, ch1, ch2, ch3, ch4, ch1_m)
		print("X_offset, ", xoffset)
		print("Y_offset, ", yoffset)
		printSingleCroppedCells(ch1, ch2, ch3, ch4, ch1_m, mask, cell_coords)
	# cellSizes.append(cellSize(cell_coords))
print "Cells shape ", cells.shape
print labels.shape
uniques, frequency = np.unique(labels[:,0], return_counts=True)
print "Unique labels are:", uniques
print "Label Frequencies are:", frequency
print "Faults: ", faults
print "Discard_count", final_discard_count

# print labels
# code.interact(local=dict(globals(), **locals()))
# Plott Cell Width frequency distribution
# frequencies = sum(cellSizes, [])
# plotHistogram(frequencies, minimum_cellwidth**2, 8000)

code.interact(local=dict(globals(), **locals()))
# np.save("/Volumes/MoritzBertholdHD/CellData/Experiments/Ex" + str(ex) + "/PreparedData/all_channels_66_66_full_no_zeros_in_cells_no_bleed_trough_shifted", cells, allow_pickle=True, fix_imports=True)
# np.save("/Volumes/MoritzBertholdHD/CellData/Experiments/Ex" + str(ex) + "/PreparedData/labels_66_66_full_no_zeros_in_cells_no_bleed_trough_shifted", labels.astype(int), allow_pickle=True, fix_imports=True)
print "Done with Experiment no ", ex
del(cells)
del(labels)
del(masks)
del(i)
del(uniques)
del(frequency)
del(new_yoffset)
del(new_xoffset)
print "Donedone"


# import numpy as np
# import matplotlib.pyplot as plt
# a = np.load("/Volumes/MoritzBertholdHD/CellData/Experiments/Ex3/PreparedData/all_channels_66_66_full_no_zeros_in_cells_no_bleed_trough_shifted.npy")
#
# new_array = np.vstack((a,b))
# new_labels = np.vstack((labels_a, labels_b))
#
# a.shape
# image = a[-1,3,:,:]
# plt.figure()
# plt.imshow(image)
# plt.show()
