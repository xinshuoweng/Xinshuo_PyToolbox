#!/usr/bin/env th

require 'totem'

local test = totem.TestSuite()

local tester = totem.Tester()

function test.dimension()
    local a = torch.Tensor(1, 2)
    local b = torch.Tensor(2)
    tester:assertTensorEq(a, b, 1e-16, 'a == b')
    tester:assertTensorNe(a, b, 1e-16, 'a ~= b')
end

function test.size()
    local a = torch.Tensor(1, 2)
    local b = torch.Tensor(2, 2)
    tester:assertTensorEq(a, b, 1e-16, 'a == b')
    tester:assertTensorNe(a, b, 1e-16, 'a ~= b')
end

function test.tensorsOfDimZero()
    local a = torch.Tensor()
    local b = torch.Tensor()
    tester:assertTensorEq(a, b, 1e-16, 'a == b')
    tester:assertTensorNe(a, b, 1e-16, 'a ~= b')
end

function test.type()
    local a = torch.DoubleTensor(3):zero()
    local b = torch.IntTensor(3):zero()
    tester:assertTensorEq(a, b, 1e-16, 'a == b')
    tester:assertTensorNe(a, b, 1e-16, 'a ~= b')
end

function test.differentValues()
    local a = torch.zeros(1, 2)
    local b = torch.ones(1, 2)
    tester:assertTensorEq(a, b, 1e-16, 'a == b')
    tester:assertTensorNe(a, b, 1e-16, 'a ~= b')
end

function test.sameValues()
    local a = torch.zeros(1, 2)
    local b = torch.zeros(1, 2)
    tester:assertTensorEq(a, b, 1e-16, 'a == b')
    tester:assertTensorNe(a, b, 1e-16, 'a ~= b')
end

function test.byteTensor()
    local a = torch.zeros(1, 2):byte()
    local b = torch.zeros(1, 2):byte()
    tester:assertTensorEq(a, b, 1e-16, 'a == b')
    tester:assertTensorNe(a, b, 1e-16, 'a ~= b')
end

return tester:add(test):run()
