#!/usr/bin/env python
# -*- coding: utf-8 -*- 

# Author: Xinshuo
# Email: xinshuow@andrew.cmu.edu
import numpy as np
import yaml
import math
import sys, time
from copy import deepcopy

import __init__paths__
import caffe
from math_function import cart2pol_2d_degree
from image_processing import imagecoor2cartesian_center
from conversions import float2percent
from timer import Timer, format_time


class MaxPolarPoolingLayer(caffe.Layer):
    """Max Polar Pooling layer used for training."""
    def setup(self, bottom, top):
        assert len(bottom) == 1, 'only accept 1 bottom layer for MaxCircularPooling'
        assert len(bottom[0].shape) == 4, 'The bottom layer should have a 3-d blob but got %d-d' % len(bottom[0].shape)

        if hasattr(self, 'param_str'):
            layer_params = yaml.load(self.param_str)
        if hasattr(self, 'param_str_'):
            layer_params = yaml.load(self.param_str_)
        self._kernal_size_rho = int(layer_params.get('kernal_size_rho') or 1)
        self._stride_rho = int(layer_params.get('stride_rho') or 1)
        self._kernal_size_phi = int(layer_params.get('kernal_size_phi') or 1)
        self._stride_phi = int(layer_params.get('stride_phi') or 1)
        # self._pad_rho = layer_params['pad_rho']
        # self._pad_phi = layer_params['pad_phi']

        self._vis = bool(layer_params.get('vis') or False)
        self._debug = bool(layer_params.get('debug') or False)
        print('stride of rho is %d' % self._stride_rho)
        print('kernal size of rho is %d' % self._kernal_size_rho)
        print('stride of phi is %d' % self._stride_phi)
        print('kernal size of phi is %d' % self._kernal_size_phi)
        if self._vis:
            print('vis mode is on. one could have access to the visualization of activation')
            assert len(top) == 2, '2 top layers for visualizing activation of MaxCircularPooling'
        else:
            assert len(top) == 1, 'only accept 1 top layer for MaxCircularPooling'
        if self._debug:
            print('debug mode is on. all debug info will be printed')
        self._firsttime = True
        self._test = False

    def forward(self, bottom, top):
        # reinitialize the data
        top[0].data[...] = -1       # reset the top data the very small value
        if self._vis:
            normalized_activation = 1./(self._height * self._width)
            top[1].data[...] = 0.
        self._switches.fill(999999)       # reset the top data the very small value
        self._mask.fill(False)

        if self._debug:
            assert all(item == -1 for item in top[0].data.flatten().tolist()), 'item reinitialization is not correct'
            assert all(item > 100000 for item in self._switches.flatten().tolist()), 'item reinitialization is not correct'
            assert all(item is False for item in self._mask.flatten().tolist()), 'item reinitialization is not correct'
            if self._vis:
                assert all(item == 0 for item in top[1].data.flatten().tolist()), 'item reinitialization is not correct'

        # forward  
        for index, index_circle_list in self._dict_index_circle_list.items():
            pts_tmp = self._dict_pts[index]
            index_sector_list = self._dict_index_sector_list[index]

            if self._debug:
                print 'current point coordinate: {}, '.format(pts_tmp), 
                print 'lies in {}th circle, '.format(index_circle_list),
                print 'lies in {}th sector, '.format(index_sector_list),
                assert len(index_circle_list) > 0 and len(index_sector_list) > 0, 'no space to lie in'

            x = pts_tmp[0]
            y = pts_tmp[1]
            current_bottom_value = deepcopy(bottom[0].data[:, :, y, x])   

            for index_circle_tmp in index_circle_list:
                for index_sector_tmp in index_sector_list:
                    pre_max = deepcopy(top[0].data[:, :, index_circle_tmp, index_sector_tmp])                             
                    spatial_bool = current_bottom_value >= pre_max

                    # fire
                    new_max = pre_max
                    new_max[spatial_bool] = current_bottom_value[spatial_bool]
                    top[0].data[:, :, index_circle_tmp, index_sector_tmp] = new_max

                    self._dummy_switches = self._switches[:, :, index_circle_tmp, index_sector_tmp, 0]
                    self._dummy_switches[spatial_bool] = x
                    self._switches[:, :, index_circle_tmp, index_sector_tmp, 0] = self._dummy_switches

                    self._dummy_switches = self._switches[:, :, index_circle_tmp, index_sector_tmp, 1]
                    self._dummy_switches[spatial_bool] = y
                    self._switches[:, :, index_circle_tmp, index_sector_tmp, 1] = self._dummy_switches

                    pre_spatial_bool = self._mask[:, :, index_circle_tmp, index_sector_tmp]
                    spatial_bool[pre_spatial_bool] = True
                    self._mask[:, :, index_circle_tmp, index_sector_tmp] = spatial_bool

        # ignore the value in the unused cell based on mask
        ignore_unused = deepcopy(top[0].data)
        ignore_unused[~self._mask] = 0
        top[0].data[...] = ignore_unused

        if self._debug:
            assert self._mask.any(), 'mask should have at least one element fired'

        # output the argmax image blob        
        if self._vis:
            for batch in range(self._batchsize):
                for channel in range(self._channels):
                    max_x = self._switches[batch, channel, :, :, 0]
                    max_y = self._switches[batch, channel, :, :, 1]
                    max_x = max_x[self._mask[batch, channel, :, :]] # only take valid element out
                    max_y = max_y[self._mask[batch, channel, :, :]]

                    for index in range(len(max_x)):
                        top[1].data[batch, channel, max_y[index], max_x[index]] += normalized_activation

            # normalizing the activation map across all batch and channels
            normalizer = deepcopy(top[1].data)
            top[1].data[...] /= np.max(normalizer)

        if self._firsttime:
            num = 0.
            for item in self._mask.flatten().tolist():
                if item:
                    num += 1
            print('neuron usage persontage %s') % float2percent(num / (self._number_sector * self._number_circle * self._batchsize * self._channels))
            self._firsttime = False

    def backward(self, top, propagate_down, bottom):
        # only top 0 layer can propagate down
        for batch in range(self._batchsize):
            for channel in range(self._channels):
                all_index_x = self._switches[batch, channel, :, :, 0]                   # get (x, y) index for the max value and pass the gradient
                all_index_y = self._switches[batch, channel, :, :, 1]
                selected_index_x = all_index_x[self._mask[batch, channel, :, :]]        # 1 x number of true in mask
                selected_index_y = all_index_y[self._mask[batch, channel, :, :]]
                topdata = deepcopy(top[0].diff[batch, channel, :, :])
                wait_for_assign = topdata[self._mask[batch, channel, :, :]]
                bottom[0].diff[batch, channel, selected_index_y, selected_index_x] = wait_for_assign

    def reshape(self, bottom, top):
        self._height = bottom[0].shape[2]
        self._width = bottom[0].shape[3]
        self._channels = bottom[0].shape[1]
        self._batchsize = bottom[0].shape[0]
        bottom_shape = (self._height, self._width)
        self._forward_map, _ = imagecoor2cartesian_center(bottom_shape, self._debug)       
        self._rho_range = math.ceil(math.sqrt(((math.ceil(self._width-1)/2.))**2 + ((math.ceil(self._height-1)/2))**2))
        self._number_circle = int(math.ceil((self._rho_range - self._kernal_size_rho) / self._stride_rho + 1))   # ceil
        self._number_sector = int((360 - self._kernal_size_phi) / self._stride_phi + 1)    # floor

        assert self._kernal_size_rho <= self._rho_range and self._kernal_size_rho > 0, 'kernal size of rho is not correct'
        assert self._kernal_size_phi <= 360 and self._kernal_size_phi > 0, 'kernal size of phi is not correct'
        if self._debug:
            print 'range of rho is [0, %f]' % self._rho_range

        if self._firsttime:
            print('channels: %d, circles: %d, sector: %d, number of neuron is: %d' % (self._channels, self._number_circle, self._number_sector, self._channels*self._number_circle*self._number_sector))
        
        top[0].reshape(self._batchsize, self._channels, self._number_circle, self._number_sector)
        if self._vis:
            top[1].reshape(self._batchsize, self._channels, self._height, self._width)
        self._mask = np.zeros(tuple(top[0].shape), dtype=bool)      # mask to record which bin has been fired
        self._switches = np.empty(tuple(top[0].shape) + (2, ), dtype='int')    # save the coordinate which has the maximum value
        self._dummy_switches = np.empty((self._batchsize, self._channels), dtype='int')       # dummy matrix for helping constructing switches 


        # precompute polar coordinate for all points
        list_pts_coordinate = np.indices((self._width, self._height)).swapaxes(0,2).swapaxes(0,1)
        list_pts_coordinate = list(list_pts_coordinate.reshape(self._width * self._height, 2))
        self._dict_index_circle_list = dict()         # list of list, each list item contains all circle this point lies in
        self._dict_index_sector_list = dict()
        self._dict_pts = dict()

        index = 0
        for pts_tmp in list_pts_coordinate:
            normalized_pts = self._forward_map(pts_tmp, debug=self._debug)     # normalize point in the center
            normalized_polar = cart2pol_2d_degree(normalized_pts, debug=self._debug)
            rho = normalized_polar[0]       # get rho polar coordinate
            phi = normalized_polar[1]
            self._dict_index_circle_list[index] = self._circle_index_mapping(rho)     # return a list of circle which this point lies in
            self._dict_index_sector_list[index] = self._sector_index_mapping(phi)
            self._dict_pts[index] = pts_tmp
            index += 1


    def _circle_index_mapping(self, rho, debug=True):
        '''
        return a list of index of circle which this point lies in
        '''
        start_index = max(0, int((rho - self._kernal_size_rho) / self._stride_rho + 1))
        end_index = min(self._number_circle-1, int(rho / self._stride_rho))      # inclusive
        return range(start_index, end_index + 1)


    def _sector_index_mapping(self, phi, debug=True):
        '''
        return a list of index of sector which this point lies in
        '''
        start_index = max(0, int((phi - self._kernal_size_phi) / self._stride_phi + 1))
        end_index = min(self._number_sector-1, int(phi / self._stride_phi))      # inclusive
        return range(start_index, end_index + 1)







class MaxCircularPoolingLayer(caffe.Layer):
    def setup(self, bottom, top):
        assert len(bottom) == 1, 'only accept 1 bottom layer for MaxCircularPooling'
        assert len(bottom[0].shape) == 4, 'The bottom layer should have a 3-d blob but got %d-d' % len(bottom[0].shape)

        layer_params = yaml.load(self.param_str)
        self._kernal_size_rho = int(layer_params.get('kernal_size_rho') or 1)
        self._stride_rho = int(layer_params.get('stride_rho') or 1)
        # self._pad_rho = layer_params['pad_rho']

        self._vis = bool(layer_params.get('vis') or False)
        self._debug = bool(layer_params.get('debug') or False)

        print('stride of rho is %d' % self._stride_rho)
        print('kernal size of rho is %d' % self._kernal_size_rho)
        if self._vis:
            print('vis mode is on. one could have access to the visualization of activation')
            assert len(top) == 2, '2 top layers for visualizing activation of MaxCircularPooling'
        else:
            assert len(top) == 1, 'only accept 1 top layer for MaxCircularPooling'
        if self._debug:
            print('debug mode is on. all debug info will be printed')
        self._firsttime = True

    def forward(self, bottom, top):
        # reinitialize the data
        top[0].data[...] = -1               # reset the top data the very small value
        self._switches.fill(0)              # reset the top data the very small value
        if self._vis:
            top[1].data[...] = 0.
            normalized_activation = 1./(self._height * self._width)

        if self._debug:
            assert all(item < -100000 for item in top[0].data.flatten().tolist()), 'item reinitialization is not correct'
            assert all(item > 100000 for item in self._switches.flatten().tolist()), 'item reinitialization is not correct'
            if self._vis:
                assert all(item == 0 for item in top[1].data.flatten().tolist()), 'item reinitialization is not correct'

        # forward 
        for index, index_circle_list in self._dict_index_circle_list.items():
            pts_tmp = self._dict_pts[index]
            if self._debug:
                print '\ncurrent point coordinate: {}, '.format(pts_tmp), 
                print 'lies in {}th circle, '.format(index_circle_list),
                assert len(index_circle_list) > 0, 'no circle lies in.'
                # time.sleep(1)

            x = pts_tmp[0]
            y = pts_tmp[1]

            current_bottom_value = deepcopy(bottom[0].data[:, :, y, x])
            for index_circle_tmp in index_circle_list:
                pre_max = deepcopy(top[0].data[:, :, index_circle_tmp])    
                spatial_bool = current_bottom_value >= pre_max

                # fire on spatial region
                new_max = pre_max
                new_max[spatial_bool] = current_bottom_value[spatial_bool]
                top[0].data[:, :, index_circle_tmp] = new_max
                
                self._dummy_switches = self._switches[:, :, index_circle_tmp, 0]
                self._dummy_switches[spatial_bool] = x
                self._switches[:, :, index_circle_tmp, 0] = self._dummy_switches

                self._dummy_switches = self._switches[:, :, index_circle_tmp, 1]
                self._dummy_switches[spatial_bool] = y
                self._switches[:, :, index_circle_tmp, 1] = self._dummy_switches

        if self._debug:
            assert (self._switches < max(self._height, self._width)).all(), 'switch is not well assigned'

        # output the argmax image blob        
        if self._vis:
            for circle in range(self._number_circle):
                max_x = self._switches[:, :, circle, 0]
                max_y = self._switches[:, :, circle, 1]
                # print max_x.shape
                # self._dummy_blob.fill(0)
                # self._dummy_blob[max_y, max_x] = normalized_activation          # 4x96

                for batch in range(self._batchsize):
                    for channel in range(self._channels):
                        max_y_tmp = max_y[batch, channel]
                        max_x_tmp = max_x[batch, channel]
                        top[1].data[batch, channel, max_y_tmp, max_x_tmp] += normalized_activation        # 4x96x28x28

            # normalizing the activation map across all batch and channels
            normalizer = deepcopy(top[1].data)
            top[1].data[...] /= np.max(normalizer)

    def backward(self, top, propagate_down, bottom):
        # only top 0 layer can propagate down
        for batch in range(self._batchsize):
            for channel in range(self._channels):
                all_index_x = self._switches[batch, channel, :, 0]   # get (x, y) index for the max value and pass the gradient
                all_index_y = self._switches[batch, channel, :, 1]
                topdata = deepcopy(top[0].diff[batch, channel, :])
                bottom[0].diff[batch, channel, all_index_y, all_index_x] = topdata

    def reshape(self, bottom, top):
        self._height = bottom[0].shape[2]
        self._width = bottom[0].shape[3]
        self._channels = bottom[0].shape[1]
        self._batchsize = bottom[0].shape[0]
        bottom_shape = (self._height, self._width)
        self._forward_map, _ = imagecoor2cartesian_center(bottom_shape, self._debug)       
        self._rho_range = math.ceil(math.sqrt(((math.ceil(self._width-1)/2.))**2 + ((math.ceil(self._height-1)/2))**2))
        self._number_circle = int(math.ceil((self._rho_range - self._kernal_size_rho) / self._stride_rho + 1))   # ceil

        assert self._kernal_size_rho <= self._rho_range and self._kernal_size_rho > 0, 'kernal size of rho is not correct'
        if self._debug:
            print 'range of rho is [0, %f]' % self._rho_range
            print 'number of circle is %d' % self._number_circle

        if self._firsttime:
            print('channels: %d, circles: %d, number of neuron is: %d' % (self._channels, self._number_circle, self._channels*self._number_circle))
            self._firsttime = False

        top[0].reshape(self._batchsize, self._channels, self._number_circle)
        if self._vis:
            top[1].reshape(self._batchsize, self._channels, self._height, self._width)
            self._dummy_blob = np.empty(tuple(top[1].shape), dtype='float32')
        self._switches = np.empty(tuple(top[0].shape) + (2, ), dtype='int')    # save the coordinate which has the maximum value
        self._dummy_switches = np.empty((self._batchsize, self._channels), dtype='int')       # dummy matrix for helping constructing switches 


        list_pts_coordinate = np.indices((self._width, self._height)).swapaxes(0,2).swapaxes(0,1)
        list_pts_coordinate = list(list_pts_coordinate.reshape(self._width * self._height, 2))
        self._dict_index_circle_list = dict()         # list of list, each list item contains all circle this point lies in
        self._dict_pts = dict()

        index = 0
        for pts_tmp in list_pts_coordinate:
            normalized_pts = self._forward_map(pts_tmp, debug=self._debug)     # normalize point in the center
            normalized_polar = cart2pol_2d_degree(normalized_pts, debug=self._debug)
            rho = normalized_polar[0]       # get rho polar coordinate
            self._dict_index_circle_list[index] = self._circle_index_mapping(rho)     # return a list of circle which this point lies in
            self._dict_pts[index] = pts_tmp
            index += 1

    def _circle_index_mapping(self, rho, debug=True):
        '''
        return a list of index of circle which this point lies in
        '''
        start_index = max(0, int((rho - self._kernal_size_rho) / self._stride_rho + 1))
        end_index = min(self._number_circle-1, int(rho / self._stride_rho))      # inclusive
        return range(start_index, end_index + 1)




class ConvertThetaLayer(caffe.Layer):
    def setup(self, bottom, top):
        assert len(bottom) == 1, 'only accept 1 bottom layer for MaxCircularPooling'
        assert len(bottom[0].shape) == 2, 'The bottom layer should have a 2-d blob but got %d-d' % len(bottom[0].shape)
        assert bottom[0].shape[1] == 1, 'The bottom layer should have only one output but got %d' % bottom[0].shape[1]

    def forward(self, bottom, top):
        for batch in range(self._batchsize):
            rot_radian = deepcopy(bottom[0].data[batch, :])
            top[0].data[batch, :] = np.array([math.cos(rot_radian), math.sin(rot_radian), 0, -math.sin(rot_radian), math.cos(rot_radian), 0])


    def backward(self, top, propagate_down, bottom):
        for batch in range(self._batchsize):
            diff1 = deepcopy(top[0].diff[batch, 0])
            diff2 = deepcopy(top[0].diff[batch, 1])
            diff3 = deepcopy(top[0].diff[batch, 3])
            diff4 = deepcopy(top[0].diff[batch, 4])
            bottom[0].diff[batch, 0] = -math.sin(diff1) + math.cos(diff2) - math.cos(diff3) - math.sin(diff4)


    def reshape(self, bottom, top):
        self._batchsize = bottom[0].shape[0]
        top[0].reshape(self._batchsize, 6)





class SinLayer(caffe.Layer):
    def setup(self, bottom, top):
        assert len(bottom) == 1, 'only accept 1 bottom layer for MaxCircularPooling'
        assert len(bottom[0].shape) == 2, 'The bottom layer should have a 2-d blob but got %d-d' % len(bottom[0].shape)
        assert bottom[0].shape[1] == 1, 'The bottom layer should have only one output but got %d' % bottom[0].shape[1]

    def forward(self, bottom, top):
        for batch in range(self._batchsize):
            rot_radian = deepcopy(bottom[0].data[batch, 0])
            top[0].data[batch, :] = math.sin(rot_radian)

    def backward(self, top, propagate_down, bottom):
        for batch in range(self._batchsize):
            diff = deepcopy(top[0].diff[batch, 0])
            bottom[0].diff[batch, 0] = math.cos(diff)


    def reshape(self, bottom, top):
        self._batchsize = bottom[0].shape[0]
        top[0].reshape(self._batchsize, 1)




class CosLayer(caffe.Layer):
    def setup(self, bottom, top):
        assert len(bottom) == 1, 'only accept 1 bottom layer for MaxCircularPooling'
        assert len(bottom[0].shape) == 2, 'The bottom layer should have a 2-d blob but got %d-d' % len(bottom[0].shape)
        assert bottom[0].shape[1] == 1, 'The bottom layer should have only one output but got %d' % bottom[0].shape[1]

    def forward(self, bottom, top):
        for batch in range(self._batchsize):
            rot_radian = deepcopy(bottom[0].data[batch, 0])
            top[0].data[batch, :] = math.cos(rot_radian)

    def backward(self, top, propagate_down, bottom):
        for batch in range(self._batchsize):
            diff = deepcopy(top[0].diff[batch, 0])
            bottom[0].diff[batch, 0] = -math.sin(diff)


    def reshape(self, bottom, top):
        self._batchsize = bottom[0].shape[0]
        top[0].reshape(self._batchsize, 1)


