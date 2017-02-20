#!/usr/bin/env th

require 'totem'

test = {}

tester = totem.Tester()

function concatenate(a, b)
    return a .. ' ' .. b
end

function test.A()
    local a = 'Hello'
    local b = 'World'
    tester:asserteq(concatenate(a,b), 'Hello World', 'Error in concatenation')
end

function test.B()
    local a = 'Hello'
    local function f()
        return concatenate(a, b)
    end
    -- assertError works similarly when using the command line parameter --rethrow
    tester:assertError(f, 'Error not caught')
end

function test.C()
    local a = 'Hello'
    -- This assert will produce an error while trying to concatenate a nil value
    -- The command line parameter --rethrow makes the program crash with this
    -- error with the correct information in the stack.
    tester:asserteq(concatenate(a,b), 'Hello World', 'Error in concatenation')
end

return tester:add(test):run()
