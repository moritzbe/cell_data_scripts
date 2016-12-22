import numpy as np
from libtiff import TIFFfile, TIFFimage
import code 

def loadnumpy(filename):
	array = np.load(filename)
	return array


# Loading data in Python:


# Loading the data into X, y
X = loadnumpy("/Volumes/MoritzBertholdHD/CellData/Experiments/Ex1/PreparedData/all_channels_80_80_full.npy")
# X = loadnumpy("/Volumes/MoritzBertholdHD/CellData/Experiments/Ex1/PreparedData/all_channels_80_80_full1.npy")
labels = loadnumpy("/Volumes/MoritzBertholdHD/CellData/Experiments/Ex1/PreparedData/labels_80_80_full.npy")

# +/+ path
p1 = "/Volumes/MoritzBertholdHD/CellData/Experiments/Ex1/LabeledImages/full_ex1/++/"
# +/- path
p2 = "/Volumes/MoritzBertholdHD/CellData/Experiments/Ex1/LabeledImages/full_ex1/+-/"
# -/+ path
p3 = "/Volumes/MoritzBertholdHD/CellData/Experiments/Ex1/LabeledImages/full_ex1/-+/"
# -/- path
p4 = "/Volumes/MoritzBertholdHD/CellData/Experiments/Ex1/LabeledImages/full_ex1/--/"
# outliers
# p5 = "/Volumes/MoritzBertholdHD/CellData/Experiments/Ex1/LabeledImages/full_ex1/outliers/"


y = labels
# for i in range(y.shape[0]):
for i in range(y.shape[0]):
	# to create a tiff structure from image data
	tiff_ch1 = TIFFimage(X[i,0,:,:].astype('uint16'), description='')
	tiff_ch2 = TIFFimage(X[i,1,:,:].astype('uint16'), description='')
	tiff_ch3 = TIFFimage(X[i,2,:,:].astype('uint16'), description='')
	tiff_ch4 = TIFFimage(X[i,3,:,:].astype('uint16'), description='')

	if i%100 == 0:
		print "Cell no.", i

	if y[i,0]==0:
		tiff_ch1.write_file(p1 + 'id'+ str(i) + '_class' + str(y[i,0])+ '_' + str(y[i,1]) + '_ch1.tif', compression='none')
		tiff_ch2.write_file(p1 + 'id'+ str(i) + '_class' + str(y[i,0])+ '_' + str(y[i,1]) + '_ch2.tif', compression='none')
		tiff_ch3.write_file(p1 + 'id'+ str(i) + '_class' + str(y[i,0])+ '_' + str(y[i,1]) + '_ch3.tif', compression='none')
		tiff_ch4.write_file(p1 + 'id'+ str(i) + '_class' + str(y[i,0])+ '_' + str(y[i,1]) + '_ch4.tif', compression='none')
	if y[i,0]==1:
		tiff_ch1.write_file(p2 + 'id'+ str(i) + '_class' + str(y[i,0])+ '_' + str(y[i,2]) + '_ch1.tif', compression='none')
		tiff_ch2.write_file(p2 + 'id'+ str(i) + '_class' + str(y[i,0])+ '_' + str(y[i,2]) + '_ch2.tif', compression='none')
		tiff_ch3.write_file(p2 + 'id'+ str(i) + '_class' + str(y[i,0])+ '_' + str(y[i,2]) + '_ch3.tif', compression='none')
		tiff_ch4.write_file(p2 + 'id'+ str(i) + '_class' + str(y[i,0])+ '_' + str(y[i,2]) + '_ch4.tif', compression='none')
	if y[i,0]==2:
		tiff_ch1.write_file(p3 + 'id'+ str(i) + '_class' + str(y[i,0])+ '_' + str(y[i,3]) + '_ch1.tif', compression='none')
		tiff_ch2.write_file(p3 + 'id'+ str(i) + '_class' + str(y[i,0])+ '_' + str(y[i,3]) + '_ch2.tif', compression='none')
		tiff_ch3.write_file(p3 + 'id'+ str(i) + '_class' + str(y[i,0])+ '_' + str(y[i,3]) + '_ch3.tif', compression='none')
		tiff_ch4.write_file(p3 + 'id'+ str(i) + '_class' + str(y[i,0])+ '_' + str(y[i,3]) + '_ch4.tif', compression='none')
	if y[i,0]==3:
		tiff_ch1.write_file(p4 + 'id'+ str(i) + '_class' + str(y[i,0])+ '_' + str(y[i,4]) + '_ch1.tif', compression='none')
		tiff_ch2.write_file(p4 + 'id'+ str(i) + '_class' + str(y[i,0])+ '_' + str(y[i,4]) + '_ch2.tif', compression='none')
		tiff_ch3.write_file(p4 + 'id'+ str(i) + '_class' + str(y[i,0])+ '_' + str(y[i,4]) + '_ch3.tif', compression='none')
		tiff_ch4.write_file(p4 + 'id'+ str(i) + '_class' + str(y[i,0])+ '_' + str(y[i,4]) + '_ch4.tif', compression='none')
	# if y[i,0]==4:
	# 	tiff_ch1.write_file(p5 + 'id='+ str(i) + '_class' + str(y[i,0])+ '_ch1_' + str(y[i,5]) + '.tif', compression='none')
	# 	tiff_ch2.write_file(p5 + 'id='+ str(i) + '_class' + str(y[i,0])+ '_ch2_' + str(y[i,5]) + '.tif', compression='none')
	# 	tiff_ch3.write_file(p5 + 'id='+ str(i) + '_class' + str(y[i,0])+ '_ch3_' + str(y[i,5]) + '.tif', compression='none')
	# 	tiff_ch4.write_file(p5 + 'id='+ str(i) + '_class' + str(y[i,0])+ '_ch4_' + str(y[i,5]) + '.tif', compression='none')



# to write tiff structure to file
# tiff.write_file('filename.tif', compression='none') # or 'lzw'