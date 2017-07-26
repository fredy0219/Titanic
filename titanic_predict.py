import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from data_reconstructe import getData

class ANN(object):
	def __init__(self,M):
		self.M = M

	def fit(self, X, Y, learning_rate=10e-5, reg=10e-2, epochs=10000, show_fig=False ):
		
		K = len(set(Y))

		N, D = X.shape
		self.W1 = np.random.randn(D, self.M) 
		self.b1 = np.zeros(self.M)
		self.W2 = np.random.randn(self.M)
		self.b2 = 0

		for i in range(epochs):
			# forward propagation
			pY , Z = self.forward(X)


			# gradient descent step / back propagation
			pY_Y = pY - Y
			self.W2 -= learning_rate * (Z.T.dot(pY_Y) + reg*self.W2)
			self.b2 -= learning_rate * ((pY_Y).sum() + reg*self.b2)

			#relu
			#dZ = np.outer(pY_Y, self.W2) * (Z > 0)

			# tanh
			dZ = np.outer(pY_Y, self.W2) * (1 - Z * Z)
			
			self.W1 -= learning_rate*(X.T.dot(dZ) + reg*self.W1)
			self.b1 -= learning_rate*(np.sum(dZ, axis=0) + reg*self.b1)

			if i%100 == 0:
				print i

	def forward(self, X):
		#Z = relu(X.dot(self.W1) + self.b1)
		Z = np.tanh(X.dot(self.W1) + self.b1)
		return sigmoid(Z.dot(self.W2) + self.b2) , Z

	def predict(self, X):
		pY, _ = self.forward(X)
		return np.round(pY)


	def score(self, X, Y):
		prediction = self.predict(X)
		return 1 - error_rate(Y, prediction)

	def predict_test(self , X):

		N,D = X.shape
		pY, _ = self.forward(X)
		pY = np.round(pY)
		pY_result = np.zeros((N,2))


		pY_result[:,0] = (np.arange(N)+892)
		pY_result[:,1] = pY

		df = pd.DataFrame(pY_result ,index=None,columns=['PassengerId','Survived'])
		df['PassengerId'] = df['PassengerId'].astype(np.int32)

		df['Survived'] = df['Survived'].astype(np.int32)

		df.to_csv('gender_submission.csv' , index = False)


def relu(x):
	return x * (x>0)

def sigmoid(A):
	return 1 / (1 + np.exp(-A))

def error_rate(targets, perdictions):
	return np.mean(targets != perdictions)


def main():
	X,Y,XTest= getData()

	#print XTest.shape
	model = ANN(100)
	model.fit(X, Y, show_fig=False)
	print model.score(X,Y)

	model.predict_test(XTest)

if __name__ == '__main__':
    main()

