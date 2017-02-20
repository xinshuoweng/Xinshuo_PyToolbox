#!/usr/bin/env th

require 'totem'

local tester = totem.Tester()
tester:add(dofile('test_nn.lua'), 'nn')
tester:add(dofile('test_simple.lua'), 'simple')
tester:add(dofile('test_tensor.lua'), 'tensor')
return tester:run()
