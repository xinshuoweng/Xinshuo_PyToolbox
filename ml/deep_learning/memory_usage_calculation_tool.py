# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import numpy as np


from layer import Convolution, Pooling, Input
from net import SequentialNet



def main():
	# create network
	network = SequentialNet()
	network.append(Input(name='input', inputshape=(384, 256, 3)))
	network.append(Convolution(name='conv1', nOutputPlane=128, kernal_size=3, padding=1))
	network.append(Pooling(name='pool1', kernal_size=2, stride=2))
	network.append(Convolution(name='conv2', nOutputPlane=512, kernal_size=3, padding=1))
	network.append(Pooling(name='pool2', kernal_size=2, stride=2))
	network.append(Convolution(name='conv3', nOutputPlane=1024, kernal_size=3, padding=1))
	network.append(Pooling(name='pool3', kernal_size=2, stride=2))
	network.compile(np.ndarray((1, 384, 256, 3)))
	network.summary(savepath='/home/xinshuo/summary.png')



if __name__ == '__main__':
	main()