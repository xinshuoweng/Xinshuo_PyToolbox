# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import numpy as np


from layer import Convolution, Pooling, Input
from net import Net



def main():
	# create network
	network = SequentialNet()
	network.append(Input(name='data', inputshape=(100, 100, 3)))
	network.append(Convolution(name='conv1', nOutputPlane=128, kernal_size=3))
	network.append(Pooling(name='pool1', kernal_size=2, stride=1))
	print(network)
	# network.get_memory_usage()



if __name__ == '__main__':
	main()