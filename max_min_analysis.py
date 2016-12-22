import numpy as np	
import os
import matplotlib.pyplot as plt
from scipy import ndimage as nd
# import cv2
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

def storeData(ch1, ch2, ch3, ch4, mask, cell_coords, imagewidth):
	# somehow cells are turned 90 degrees
	ch1 = ch1 * mask
	ch2 = ch2 * mask
	ch3 = ch3 * mask
	ch4 = ch4 * mask
	cells_per_image = np.empty([0, 4, imagewidth,imagewidth])
	for item in cell_coords:
		height = item["x_max"] - item["x_min"]
		width = item["y_max"] - item["y_min"]
		cell_container = np.zeros([1, 4, imagewidth, imagewidth])
		if (width > imagewidth) & (height <= imagewidth):
			cell_container[0, 0, :, 0:height] = cv2.resize(ch1[item["y_min"]:item["y_max"],item["x_min"]:item["x_max"]], (height, imagewidth), interpolation = cv2.INTER_LINEAR)
			cell_container[0, 1, :, 0:height] = cv2.resize(ch2[item["y_min"]:item["y_max"],item["x_min"]:item["x_max"]], (height, imagewidth), interpolation = cv2.INTER_LINEAR)
			cell_container[0, 2, :, 0:height] = cv2.resize(ch3[item["y_min"]:item["y_max"],item["x_min"]:item["x_max"]], (height, imagewidth), interpolation = cv2.INTER_LINEAR)
			cell_container[0, 3, :, 0:height] = cv2.resize(ch4[item["y_min"]:item["y_max"],item["x_min"]:item["x_max"]], (height, imagewidth), interpolation = cv2.INTER_LINEAR)
		elif (width <= imagewidth) & (height > imagewidth):
			cell_container[0, 0, 0:width,:] = cv2.resize(ch1[item["y_min"]:item["y_max"],item["x_min"]:item["x_max"]], (imagewidth, width), interpolation = cv2.INTER_LINEAR)
			cell_container[0, 1, 0:width,:] = cv2.resize(ch2[item["y_min"]:item["y_max"],item["x_min"]:item["x_max"]], (imagewidth, width), interpolation = cv2.INTER_LINEAR)
			cell_container[0, 2, 0:width,:] = cv2.resize(ch3[item["y_min"]:item["y_max"],item["x_min"]:item["x_max"]], (imagewidth, width), interpolation = cv2.INTER_LINEAR)
			cell_container[0, 3, 0:width,:] = cv2.resize(ch4[item["y_min"]:item["y_max"],item["x_min"]:item["x_max"]], (imagewidth, width), interpolation = cv2.INTER_LINEAR)
		elif (width > imagewidth) & (height > imagewidth):
			cell_container[0, 0,:,:] = cv2.resize(ch1[item["y_min"]:item["y_max"],item["x_min"]:item["x_max"]], (imagewidth, imagewidth), interpolation = cv2.INTER_LINEAR)
			cell_container[0, 1,:,:] = cv2.resize(ch2[item["y_min"]:item["y_max"],item["x_min"]:item["x_max"]], (imagewidth, imagewidth), interpolation = cv2.INTER_LINEAR)
			cell_container[0, 2,:,:] = cv2.resize(ch3[item["y_min"]:item["y_max"],item["x_min"]:item["x_max"]], (imagewidth, imagewidth), interpolation = cv2.INTER_LINEAR)
			cell_container[0, 3,:,:] = cv2.resize(ch4[item["y_min"]:item["y_max"],item["x_min"]:item["x_max"]], (imagewidth, imagewidth), interpolation = cv2.INTER_LINEAR)
		else:
			cell_container[0, 0, 0:width, 0:height] = ch1[item["y_min"]:item["y_max"],item["x_min"]:item["x_max"]]
			cell_container[0, 1, 0:width, 0:height] = ch2[item["y_min"]:item["y_max"],item["x_min"]:item["x_max"]]
			cell_container[0, 2, 0:width, 0:height] = ch3[item["y_min"]:item["y_max"],item["x_min"]:item["x_max"]]
			cell_container[0, 3, 0:width, 0:height] = ch4[item["y_min"]:item["y_max"],item["x_min"]:item["x_max"]]
						# resize to fit boundary	
		# plt.imshow(cell_container[0, 0, :,:])
		# plt.show()

		cells_per_image = np.vstack((cells_per_image, cell_container))
	
	return cells_per_image


def convertNPYtoLMDB(X, path, db_name):
	N = X.shape[0]
	# X = np.zeros((N, 3, 32, 32), dtype=np.uint8)
	# y = np.zeros(N, dtype=np.int64)
	save_as = path + db_name
	map_size = X.nbytes * 5
	env = lmdb.open(save_as, map_size=map_size)
	with env.begin(write=True) as txn:
	    # txn is a Transaction object
	    for i in range(N):
	        datum = caffe.proto.caffe_pb2.Datum()
	        datum.channels = X.shape[1]
	        datum.height = X.shape[2]
	        datum.width = X.shape[3]
	        datum.data = X[i].tobytes()  # or .tostring() if numpy < 1.9
	        # datum.label = int(y[i])
	        str_id = '{:08}'.format(i)
	        # The encode is only essential in Python 3
	        txn.put(str_id.encode('ascii'), datum.SerializeToString())

def readLMDB(db_path):
	env = lmdb.open(db_path, readonly=True)
	with env.begin() as txn:
	    raw_datum = txn.get(b'00000000')

	datum = caffe.proto.caffe_pb2.Datum()
	datum.ParseFromString(raw_datum)

	flat_x = np.fromstring(datum.data, dtype=np.uint8)
	print flat_x.shape
	X = flat_x.reshape(datum.channels, datum.height, datum.width)
	# y = datum.label


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
# Cell cropping extension: Recommend 10
cell_pad = 10
# Cellsizes (32*32) and (128 * 128) 
minimum_cellwidth = 18
maximum_cellwidth = 128
imagewidth = 80

# mask dilation for cropping of final images: Recommend 5
dilation_coef = 5
# Process images until ith mask
max_mask = 50






#### ------ ITERATION -------------------------- #####
#### ------------------------------------------- #####


cellSizes = []
cells = np.empty([0, 4, imagewidth, imagewidth])
max_ch1 = []
max_ch2 = [] 
max_ch3 = []
max_ch4 = []
min_ch1 = []
min_ch2 = [] 
min_ch3 = []
min_ch4 = []
max_total = []

total_maximum_ch1 = 0
total_maximum_ch2 = 0
total_maximum_ch3 = 0
total_maximum_ch4 = 0



# Looping over DIB masks and all channels to select image:
for i in masks[0:max_mask]:
	# Masks have to be point-reflected
	# mask = binaryMask(plt.imread(i[:-6]+"o1.TIF"), padding, kernelsize)
	ch1 = plt.imread(i[:-6]+"d0.TIF")
	ch2 = plt.imread(i[:-6]+"d1.TIF")
	ch3 = plt.imread(i[:-6]+"d2.TIF")
	ch4 = plt.imread(i[:-6]+"d3.TIF")


	if np.max(ch1) > total_maximum_ch1:
		total_maximum_ch1 = np.max(ch1)
	if np.max(ch2) > total_maximum_ch2:
		total_maximum_ch2 = np.max(ch2)
	if np.max(ch3) > total_maximum_ch3:
		total_maximum_ch3 = np.max(ch3)
	if np.max(ch4) > total_maximum_ch4:
		total_maximum_ch4 = np.max(ch4)


print "max ch1", total_maximum_ch1
print "max ch2", total_maximum_ch2
print "max ch3", total_maximum_ch3
print "max ch4", total_maximum_ch4

# code.interact(local=dict(globals(), **locals()))
# Plott Cell Width frequency distribution
frequencies = sum(cellSizes, [])
plotHistogram(frequencies, minimum_cellwidth**2, 8000)

# np.save("/Volumes/MoritzBertholdHD/CellData/Experiments/Ex1/PreData/all_channels_80_80_unsup", cells, allow_pickle=True, fix_imports=True)
