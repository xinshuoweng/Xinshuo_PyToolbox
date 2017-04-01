# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import numpy as np


from layer import Convolution, Pooling, Input
from net import Net



def main():
	# create network
	network = Net(Input(np.array((100, 100, 3), dtype='float32')))
	network.append(Convolution(name='conv1', nInputPlane=3, nOutputPlane=128, kernal_size=3))
	network.append(Pooling(name='pool1', nInputPlane=128, nOutputPlane=16, kernal_size=2, stride=1))
	network.print()
	network.get_memory_usage()



if __name__ == '__main__':
	main()