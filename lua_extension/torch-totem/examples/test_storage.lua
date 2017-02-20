#!/usr/bin/env th

require 'totem'

local test = totem.TestSuite()

local tester = totem.Tester()

function test.size()
    local a = torch.Storage(1)
    local b = torch.Storage(2)
    tester:assertStorageEq(a, b, 1e-16, 'a == b')
    tester:assertStorageNe(a, b, 1e-16, 'a ~= b')
end

function test.type()
    local a = torch.DoubleStorage({1, 2, 3})
    local b = torch.IntStorage({1, 2, 3})
    tester:assertStorageEq(a, b, 1e-16, 'a == b')
    tester:assertStorageNe(a, b, 1e-16, 'a ~= b')
end

function test.differentValues()
    local a = torch.Storage({1, 2})
    local b = torch.Storage({3, 4})
    tester:assertStorageEq(a, b, 1e-16, 'a == b')
    tester:assertStorageNe(a, b, 1e-16, 'a ~= b')
end

function test.sameValues()
    local a = torch.Storage({1, 2})
    local b = torch.Storage({1, 2})
    tester:assertStorageEq(a, b, 1e-16, 'a == b')
    tester:assertStorageNe(a, b, 1e-16, 'a ~= b')
end


return tester:add(test):run()
