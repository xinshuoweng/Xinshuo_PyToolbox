-- Defines a totem.Tester class. This is the main object of the totem tester.

local lapp = require 'pl.lapp'

--[[ totem Tester class.
This class defines all the basic testing utilities provided by the totem
package.
Arguments: No arguments.
]]
local Tester, parent = torch.class('totem.Tester', 'torch.Tester')

function Tester:__init()
    parent.__init(self)
    self._assertTensorEqIgnoresDims = false
end

function Tester:_assert_sub(condition, message)
    -- We override the parent _assert_sub function so that we can optionally
    -- do only a return, rather than doing a test assert. This is provided for
    -- backwards compatibility.
    self._assertSubCalled = true
    if not self._ret then
        return parent._assert_sub(self, condition, message)
    else
        return condition
    end
end

function Tester:_wrapFunctionForRet(ret, f, ...)
    self._ret = ret
    local result, message = f(...)
    assert(self._assertSubCalled) -- sanity check
    self._ret = nil
    self._assertSubCalled = nil
    return result, message
end

--[[ Similar to torch.Tester.eq, with an additional arg `ret` provided for
backwards compatibility that if true, will return (boolean) the test outcome
instead of running a test assertion.
]]
function Tester:eq(got, expected, message, tolerance, ret)
    return self:_wrapFunctionForRet(ret, parent.eq, self, got, expected,
                                    message, tolerance)
end


--[[ Similar to torch.Tester.ne, with an additional arg `ret` provided for
backwards compatibility that if true, will return (boolean) the test outcome
instead of running a test assertion.
]]
function Tester:ne(got, expected, message, tolerance, ret)
    return self:_wrapFunctionForRet(ret, parent.ne, self, got, expected,
                                    message, tolerance)
end


--[[ Asserts that two storages are equal.
The storages are considered equal if they are of the same sizes and types
and if the maximum elementwise difference <= tolerance.
Arguments:
* `sa` (storage) first storage.
* `sb` (storage) second storage.
* `tolerance` (number) the maximum acceptable difference of ta and tb.
* `message` (string) the error message to be displayed in case of failure.
Returns (boolean) whether the test succeeded.
]]
function Tester:assertStorageEq(sa, sb, tolerance, message)
    return parent.eq(self, sa, sb, tolerance, message)
end


--[[ Asserts that two storages are unequal.
The storages are considered unequal if they are not of the same sizes or types
or if the maximum elementwise difference > tolerance.
Arguments:
* `sa` (storage) first storage.
* `sb` (storage) second storage.
* `tolerance` (number) the minimum acceptable difference of ta and tb.
* `message` (string) the error message to be displayed in case of failure.
Returns (boolean) whether the test succeeded.
]]
function Tester:assertStorageNe(sa, sb, tolerance, message)
    return parent.ne(self, sa, sb, tolerance, message)
end


function Tester:_listTests(tests)
    for name, _ in pairs(tests) do
        print(name)
    end
end

-- command line options
Tester.CLoptions = [[
    --list print the names of the available tests instead of running them.
    --log-output (optional file-out) redirect compact test results to file.
        This contains one line per test in the following format:
        name #passed-assertions #failed-assertions #exceptions
    --summary print only pass/fail status rather than full error messages.
    --early-abort (optional boolean) abort execution on first error.
    --rethrow (optional boolean) errors make the program crash and propagate up
        the stack.
    ]]

function Tester:run(testNames)
    if arg then
        local args = lapp([[Run tests

Usage:

  ]] .. arg[0] .. [[ [options] [test1 [test2...] ]

Options:

]]
.. Tester.CLoptions ..
[[

If any test names are specified only the named tests are run. Otherwise
all the tests are run.

]])

        if #args > 0 then
            testNames = args
        end

        if args.list then
            self:_listTests(self:_getTests(testNames))
            return 0
        end

        if args.summary then
            self:setSummaryOnly(true)
        end
        if args.early_abort then
            self:setEarlyAbort(true)
        end
        if args.rethrow then
            self:setRethrowErrors(true)
        end
        if args.log_output then
            self.extraOutputFile = args.log_output
        end
    end

    parent.run(self, testNames)
end

function Tester:_run(tests)
    parent._run(self, tests)
    if self.extraOutputFile then
        self:_logExtraOutput(tests)
    end
end

function Tester:_logExtraOutput(tests)
    local function unwords(...)
        return table.concat({...}, ' ')
    end
    local f = self.extraOutputFile

    local npasses, nfails, nerrors = 0, 0, 0
    for name, _ in pairs(tests) do
        npasses = npasses + self.assertionPass[name]
        nfails = nfails + self.assertionFail[name]
        nerrors = nerrors + self.testError[name]
        f:write(unwords(name, self.assertionPass[name],
                        self.assertionFail[name], self.testError[name]))
        f:write('\n')
    end
    f:write(unwords('[total]', npasses, nfails, nerrors))
    f:write('\n')
    f:close()
end
