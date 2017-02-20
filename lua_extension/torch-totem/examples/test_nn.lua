#!/usr/bin/env th

require 'totem'
require 'nn'

local test = totem.TestSuite()

local tester = totem.Tester()


local function net()
    local net = nn.Linear(10, 10)
    local input = torch.randn(5, 10)
    return net, input
end


function test.gradients()
    totem.nn.checkGradients(tester, net())
end


function test.minibatch()
    totem.nn.checkMinibatch(tester, net())
end


return tester:add(test):run()
