from data_tools import *
from algorithms import *
from plot_lib import *
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
import numpy as np
import code 




# Loading data in Python:


# Loading the data into X, y
X = loadnumpy("/Volumes/MoritzBertholdHD/CellData/Experiments/Ex1/PreparedData/all_channels_80_80_full.npy")
# X = loadnumpy("/Volumes/MoritzBertholdHD/CellData/Experiments/Ex1/PreparedData/all_channels_80_80_full1.npy")
labels = loadnumpy("/Volumes/MoritzBertholdHD/CellData/Experiments/Ex1/PreparedData/labels_80_80_full.npy")
y = labels

code.interact(local=dict(globals(), **locals()))



# X -= np.mean(X)
# X = X[::10,:]
# y = y[::10]
# X_test -= np.mean(X)


# Perform PCA: 
th = 1

X_ = X.reshape(X.shape[0], 4 * X.shape[-1] * X.shape[-1])
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_)
plot3d(X_pca[::th,:], y[::th], "PCA")


# T-SNE
# X_tSNE = tsne(X_, 3, 50, 20.0);
# plot3d(X_tSNE,y,"tSNE-Plot")

code.interact(local=dict(globals(), **locals()))

# th = 50
# plot3d(X[::th,:], y[::th], "PCA")
# plot2d(X[::th,:], y[::th], "PCA")

# m = X.shape[0]
# m_test = X_test.shape[0]
# n = X.shape[1]




# print "The number of training samples m is", m
# print "The number of test samples m is", m_test
# print "The number of features is", n

# #Cross Validation
# cv = 7


# # Logistic Regression (Ridge)
# # achieved Train Accuracy of 91% with reg = .3
# # accuracy largely equal for reg .1 - 20
# reg = [.3]
# accuracy = 0
# X_off = addOffset(X)
# for i in reg:
# 	log_reg = logRegress(X_off, y, i)
# 	train_accuracy = np.mean(cross_val_score(log_reg, X_off, y, cv=cv))
# 	print "The train accuracy of SVM is", train_accuracy
# 	print "c2 is", i
# 	if train_accuracy >= accuracy:
# 		accuracy = np.mean(cross_val_score(log_reg, X_off, y, cv=cv))
# 		best_reg = i
# 		best_model = log_reg
# 		print "new best!"

# y_pred = best_model.predict(addOffset(X_test))
	



# SVM
# achieved Train Accuracy of 91% with c2 = .1 - 4
# best c2 = .1
# adding bias/offset
# X_off = addOffset(X)

# c2 = [0.1]
# accuracy = 0
# for i in c2:
# 	svm = supvecm(X_off,y,i)
# 	train_accuracy = np.mean(cross_val_score(svm, X_off, y, cv=cv))
# 	print "The train accuracy of SVM is", train_accuracy
# 	print "c2 is", i
# 	if train_accuracy >= accuracy:
# 		accuracy = np.mean(cross_val_score(svm, X_off, y, cv=cv))
# 		best_c2 = i
# 		best_model = svm
# 		print "new best!"
# y_pred = best_model.predict(addOffset(X_test))

# k_nn
# train accuracy with n=11 is 0.96
# neighbors = 11
# k_nn = k_nn(X, y, neighbors)
# print "The train accuracy of K-NN is", np.mean(cross_val_score(k_nn, X, y, cv=cv))
# plotConfusionMatrix(y, k_nn.predict(X))
# y_pred = k_nn.predict(X_test)

# Random Forest:
# Number of Trees K: K = 30 -> 95%, K = 100 -> 96%, K = 
# K = 2000
# rf = randForest(X, y, K)
# print "The train accuracy of Random Forest is", np.mean(cross_val_score(rf, X, y, cv=cv))
# y_pred = rf.predict(X_test)




# Fully Connected Net:
# After 100 epoqs achieved accuracy of 99.98% on training, Test_accuracy is 97% - Overfit
# y = y.T 
# model = fullyConnectedNet(X, y, epochs = 20)
# results = model.predict(X_test)
# y_pred = np.zeros([results.shape[0]])
# for i in xrange(results.shape[0]):
# 	y_pred[i] = np.argmax(results[i,:]).astype(int)


# Plotting wrong predicted images
# y = y.T 
# model = fullyConnectedNet(X, y, epochs = 9)
# results = model.predict(X)
# y_pred = np.zeros([results.shape[0]])
# for i in xrange(results.shape[0]):
# 	y_pred[i] = np.argmax(results[i,:]).astype(int)
# 	if y_pred[i] != y[i]:
# 		plotImage(X_original, i, label = np.array_str(results[i]))




# First CovNet:
# Train Accuracy = 99.6%
# Test Accuracy = 98.8%
# X_ = X.reshape(X.shape[0], 28, 28, 1)
# X_test_ = X_test.reshape(X_test.shape[0], 28, 28, 1)
# # default epochs: 12
# model = covNet(X_, y, batch_size = 128, epochs = 12)
# results = model.predict(X_test_)
# y_pred = np.zeros([results.shape[0]])
# for i in xrange(results.shape[0]):
# 	y_pred[i] = np.argmax(results[i,:]).astype(int)



