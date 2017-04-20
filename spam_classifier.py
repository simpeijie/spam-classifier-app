import numpy as np 
import random
from random import randint
import re
import featurize as fe
from collections import defaultdict
from scipy.io import loadmat

np.set_printoptions(threshold=float('inf'))

def standardize(X_train):
	return ((X_train - np.mean(X_train, axis=0))/ np.std(X_train, axis=0))

def transform(X_train):
	return np.log(X_train + 0.1)

def binarize(X_train):
	return (np.array([[1 if X_train[i][j] > 0 else 0 for j in range(X_train.shape[1])] for i in range(X_train.shape[0])]))
			
def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def compute(x, y, n, d):
	y = y.reshape(n, 1)

	beta = np.ones(d).reshape(d, 1)
	mu = sigmoid(x.dot(beta))

	return x, y, beta, mu

def train_gd(X_train, y_train, alpha, reg, num_iter=8000):
	n, d = X_train.shape
	x, y, beta, mu = compute(X_train, y_train, n, d)
	loss = []
	for i in range(num_iter):
		dl = 2*reg*(beta) - x.T.dot(y-mu)
		beta = beta - (alpha/n)*dl
		mu = sigmoid(x.dot(beta))
		loss.append((1/n)*(reg*(beta.T.dot(beta)) - (y.T.dot(np.log(mu+1e-30)) + (1-y).T.dot(np.log(1-mu+1e-30))))[0][0])
	return beta, loss

def train_sgd(X_train, y_train, alpha, reg, num_iter=10000, decrease=False):
	n, d = X_train.shape
	x, y, beta, mu = compute(X_train, y_train, n, d)
	loss = []
	for i in range(num_iter):
		s = randint(0, n-1)
		xi, yi, mui = x[s, :].reshape((d,1)), y[s], mu[s]
		dl = 2*reg*beta - xi*(yi-mui)
		if decrease:
			beta = beta - 2*(alpha/((i+1)*n))*dl
		else:
			beta = beta - (alpha/n)*dl
		mu = sigmoid(x.dot(beta))
		loss.append((1/n)*(reg*(beta.T.dot(beta)) - (y.T.dot(np.log(mu+1e-30)) + (1-y).T.dot(np.log(1-mu+1e-30))))[0][0])
	return beta, loss

def predict(preds, boundary):
	return [1 if p > boundary else 0 for p in preds]

def load_data():
	mat_dict = loadmat("./data/spam_data.mat", appendmat=False)
	X_train = mat_dict['training_data']
	y_train = mat_dict['training_labels'][0]
	return transform(X_train), y_train

def generate_feature_vector(text):
	text = text.replace('\r\n', ' ') # Remove newline character
	words = re.findall(r'\w+', text)
	word_freq = defaultdict(int) # Frequency of all words
	for word in words:
	    word_freq[word] += 1
	feature_vector = fe.generate_feature_vector(text, word_freq)
	return np.array([feature_vector])



