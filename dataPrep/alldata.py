# -*- coding: utf-8 -*-
# @Author: yll
# @Date:   2018-05-09 16:27:09
# @Last Modified by:   yll
# @Last Modified time: 2018-05-09 20:12:00



'''
prepare data for neural network

Input: 
    sub1_01cpd.mat
Output:
    
'''

from scipy.io import loadmat, savemat
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils


for subi in range(1,11):

	# cpd = 0.1
	file = "sub" + str(subi) + "_01cpd.mat"
	data_mat = loadmat(file)
	data = data_mat['data'] # data: n_chns * n_temporal * n_samples
	data = np.transpose(data,(2,0,1))# data: n_samples * n_chns * n_temporal
	label = np.zeros((data.shape[0],1)) # label: n_samples * 1
	data_all = data # data_all: n_samples * n_chns * n_temporal
	labels = label
	del file, data_mat, data, label


	# cpd = 0.3
	file = "sub" + str(subi) + "_03cpd.mat"
	data_mat = loadmat(file)
	data = data_mat['data'] # data: n_chns * n_temporal * n_samples
	data = np.transpose(data,(2,0,1))# data: n_samples * n_chns * n_temporal
	label = np.zeros((data.shape[0],1)) + 1 # label: n_samples * 1
	data_all = np.concatenate((data_all, data), axis=0)# data_all: n_samples * n_chns * n_temporal
	labels = np.concatenate((labels,label),axis = 0)
	del file, data_mat, data, label

	# cpd = 0.05
	file = "sub" + str(subi) + "_005cpd.mat"
	data_mat = loadmat(file)
	data = data_mat['data'] # data: n_chns * n_temporal * n_samples
	data = np.transpose(data,(2,0,1))# data: n_samples * n_chns * n_temporal
	label = np.zeros((data.shape[0],1)) + 2 # label: n_samples * 1
	data_all = np.concatenate((data_all, data), axis=0)# data_all: n_samples * n_chns * n_temporal
	labels = np.concatenate((labels,label),axis = 0)
	del file, data_mat, data, label



	# change to one hot code
	encoder = LabelEncoder()
	encoder.fit(np.ravel(labels))
	encode_labels = encoder.transform(np.ravel(labels))
	labels_dummy = np_utils.to_categorical(encode_labels)

	savefile = '../../data/alldata_sub' + str(subi) + '.mat'
	savemat(savefile,mdict={'eeg':data_all,'labels':labels,"labels_dummy":labels_dummy},oned_as = 'row')

	del savefile, data_all, labels, labels_dummy

