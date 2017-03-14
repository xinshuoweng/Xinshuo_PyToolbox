#!/usr/bin/env th

require 'totem'

local tester = totem.Tester()
tester:add('test_nn.lua')
tester:add('test_simple.lua')
tester:add('test_tensor.lua')
return tester:run()
