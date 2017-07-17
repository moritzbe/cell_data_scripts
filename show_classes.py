import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import ndimage as nd
import code

plot = True

X = np.load("/Volumes/MoritzBertholdHD/CellData/Experiments/Ex1/PreparedData/all_channels_66_66_full_no_zeros_in_cells.npy")
y = np.load("/Volumes/MoritzBertholdHD/CellData/Experiments/Ex1/PreparedData/labels_66_66_full_no_zeros_in_cells.npy")

class_nr = 4
class_names = ["CGRP+ RIIb+", "CGRP+ RIIb-", "CGRP- RIIb+", "CGRP- RIIb-", "Glial cell"]

y_ = y[y[:,0]==class_nr,:]
X_ = X[y[:,0]==class_nr,:,:,:]

ind = [10,20,30,40,50,60,70,80,90,100,110,120,130,140]

def printWholeImages(X,y,ind,class_nr, class_names):
	fig = plt.figure()
	a=fig.add_subplot(2,2,1)
	imgplot = plt.imshow(X_[ind,0,:,:], cmap='gray')
	a.set_title('Ch1-PGP, intensity=' + str(y[ind,1]))
	plt.axis('off')
	a=fig.add_subplot(2,2,2)
	imgplot = plt.imshow(X_[ind,1,:,:], cmap='gray')
	a.set_title('Ch2-CGRP, intensity=' + str(y[ind,2]))
	plt.axis('off')
	a=fig.add_subplot(2,2,3)
	imgplot = plt.imshow(X_[ind,2,:,:], cmap='gray')
	a.set_title('Ch3-RIIb, intensity=' + str(y[ind,3]))
	plt.axis('off')
	a=fig.add_subplot(2,2,4)
	imgplot = plt.imshow(X_[ind,3,:,:], cmap='gray')
	a.set_title('Ch4-DAPI, intensity=' + str(y[ind,4]))
	plt.axis('off')
	name = class_names[class_nr]
	path = "/Users/moritzberthold/Desktop/Thesis/images/data/class_examples/"
	plt.tight_layout()
	plt.savefig(path + str(name) + str(ind) + ".eps")
	plt.show()


i = 10
printWholeImages(X_,y_,i,class_nr, class_names)

code.interact(local=dict(globals(), **locals()))
