from activation import Activation
import numpy as np

class Tanh(Activation):
	def __init__(self):
		tanh=lambda x:np.tanh(x)
		tanh_prime=lambda x:1.0-np.tanh(x)**2
		super().__init__(tanh,tanh_prime)

class Sigmoid(Activation):
	def __init__(self):
		sigmoid=lambda x : 1.0/(1.0+np.exp(-x))
		sigmoid_prime=lambda x : sigmoid(x)*(1.0-sigmoid(x))
		super().__init__(sigmoid,sigmoid_prime) 
class Relu(Activation):
	def __init__(self):
		relu=lambda x : np.maximum(x,0)
		relu_prime=lambda x: np.where(x <= 0, 0, 1)
		super().__init__(relu,relu_prime)
