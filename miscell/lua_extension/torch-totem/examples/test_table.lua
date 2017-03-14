#!/usr/bin/env th

require 'totem'

local test = totem.TestSuite()

local tester = totem.Tester()

function test.size()
    local a = {1}
    local b = {1, 2}
    tester:assertTableEq(a, b, 1e-16, 'a == b')
    tester:assertTableNe(a, b, 1e-16, 'a ~= b')
end

function test.sameValues()
    local a = {1, 2, 3}
    local b = {1, 2, 3}
    tester:assertTableEq(a, b, 1e-16, 'a == b')
    tester:assertTableNe(a, b, 1e-16, 'a ~= b')
end

function test.differentValues()
    local a = {1, 3}
    local b = {1, 2}
    tester:assertTableEq(a, b, 1e-16, 'a == b')
    tester:assertTableNe(a, b, 1e-16, 'a ~= b')
end

function test.sameValuesNested()
    local a = {1, {1, 2}}
    local b = {1, {1, 2}}
    tester:assertTableEq(a, b, 1e-16, 'a == b')
    tester:assertTableNe(a, b, 1e-16, 'a ~= b')
end

function test.differentValuesNested()
    local a = {1, {1, 2}}
    local b = {1, {1, 3}}
    tester:assertTableEq(a, b, 1e-16, 'a == b')
    tester:assertTableNe(a, b, 1e-16, 'a ~= b')
end

function test.sameValuesNestedWithOtherTypes()
    local a = {1, {1, torch.DoubleTensor(2):zero()}}
    local b = {1, {1, torch.DoubleTensor(2):zero()}}
    tester:assertTableEq(a, b, 1e-16, 'a == b')
    tester:assertTableNe(a, b, 1e-16, 'a ~= b')
end

function test.differentValuesNestedWithOtherTypes()
    local a = {1, {1, torch.DoubleTensor(2):zero()}}
    local b = {1, {1, torch.FloatTensor(2):zero()}}
    tester:assertTableEq(a, b, 1e-16, 'a == b')
    tester:assertTableNe(a, b, 1e-16, 'a ~= b')
end

return tester:add(test):run()
