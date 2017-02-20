#!/usr/bin/env th

require 'totem'

local test = totem.TestSuite()

local tester = totem.Tester()

function test.a()
    local a = 10
    local b = 10
    tester:asserteq(a, b, 'a == b')
    tester:assertne(a, b, 'a ~= b')
end

function test.b()
    local a = 10
    local b = 9
    tester:assertgt(a, b, 'a > b')
end

function test.c()
    error('Errors are treated differently than failures')
end

return tester:add(test):run()
