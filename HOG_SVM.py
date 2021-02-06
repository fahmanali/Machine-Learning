# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 21:59:00 2020

@author: User
"""

from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import os,numpy as np

print(__doc__)

# Read images and extract Hog features. you need to install skimage library
from skimage.feature import hog
from skimage import data, exposure
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
from PIL import Image
import glob
image_list=[];features_list = [];label=[]
path=os.getcwd()+'/Dataset'
for subfolder in os.listdir(path):
    subpath=path+'/'+subfolder
    for filename in os.listdir(subpath): #assuming gif
        # print(subpath+'/'+filename)
        im=Image.open(subpath+'/'+filename)
        im=im.resize((512,512)) 
        fd, hog_image = hog(im, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, multichannel=True,feature_vector=True)
        features_list.append(fd)
        label.append(subfolder)
        image_list.append(im)# if you need to use image itself not its features
X=np.asarray(features_list)
y=np.asarray(label)
# images=np.asarray(image_list)
target_names =np.unique(y)
n_samples=X.shape[0]
n_features = X.shape[1]

n_classes = target_names.shape[0]
print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)
print('labels: ',target_names)
# #############################################################################
# Download the data, if not already on disk and load it as numpy arrays






# #############################################################################
# Split into a training set and a test set using a stratified k fold

# split into a training and testing set
# to use images for pca set X=images or diable it
# X=images
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# #############################################################################
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
# n_components = 10

# print("Extracting the top %d eigenfaces from %d faces"
#       % (n_components, X_train.shape[0]))
# t0 = time()
# pca = PCA(n_components=n_components, svd_solver='randomized',
#           whiten=True).fit(X_train)
# print("done in %0.3fs" % (time() - t0))

# eigenfaces = pca.components_.reshape((n_components, 32, 256))

# print("Projecting the input data on the eigenfaces orthonormal basis")
# t0 = time()
# X_train = pca.transform(X_train)
# X_test = pca.transform(X_test)
# print("done in %0.3fs" % (time() - t0))


# #############################################################################
# Train a SVM classification model

print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(
    SVC(kernel='rbf', class_weight='balanced'), param_grid
)
clf = clf.fit(X_train, y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)


# #############################################################################
# Quantitative evaluation of the model quality on the test set

print("Predicting people's names on the test set")
t0 = time()
y_pred = clf.predict(X_test)
print("done in %0.3fs" % (time() - t0))

print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred))


# #############################################################################
# Qualitative evaluation of the predictions using matplotlib

# def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
#     """Helper function to plot a gallery of portraits"""
#     plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
#     plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
#     for i in range(n_row * n_col):
#         plt.subplot(n_row, n_col, i + 1)
#         plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
#         plt.title(titles[i], size=12)
#         plt.xticks(())
#         plt.yticks(())


# # plot the result of the prediction on a portion of the test set

# def title(y_pred, y_test, target_names, i):
#     pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
#     true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
#     return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

# prediction_titles = [title(y_pred, y_test, target_names, i)
#                      for i in range(y_pred.shape[0])]

# plot_gallery(X_test, prediction_titles, h, w)

# # plot the gallery of the most significative eigenfaces

# eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
# plot_gallery(eigenfaces, eigenface_titles, h, w)

# plt.show()