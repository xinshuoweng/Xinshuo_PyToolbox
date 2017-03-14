require 'nn'
require 'totem'

local tablex = require 'pl.tablex'

local tester = totem.Tester()

local MESSAGE = "a really useful informative error message"

local subtester = totem.Tester()
-- The message only interests us in case of failure
subtester._success = function(self) return true, MESSAGE end
subtester._failure = function(self, message) return false, message end

local tests = totem.TestSuite()

local test_name_passed_to_setUp
local calls_to_setUp = 0
local calls_to_tearDown = 0

local function meta_assert_success(success, message)
  tester:assert(success == true, "assert wasn't successful")
  tester:assert(string.find(message, MESSAGE) ~= nil, "message doesn't match")
end
local function meta_assert_failure(success, message)
  tester:assert(success == false, "assert didn't fail")
  tester:assert(string.find(message, MESSAGE) ~= nil, "message doesn't match")
end

function tests.really_test_assert()
  assert((subtester:assert(true, MESSAGE)),
         "subtester:assert doesn't actually work!")
  assert(not (subtester:assert(false, MESSAGE)),
         "subtester:assert doesn't actually work!")
end

function tests.test_assert()
  meta_assert_success(subtester:assert(true, MESSAGE))
  meta_assert_failure(subtester:assert(false, MESSAGE))
end

function tests.test_assertTensorEq_alltypes()
  local allTypes = {
      torch.ByteTensor,
      torch.CharTensor,
      torch.ShortTensor,
      torch.IntTensor,
      torch.LongTensor,
      torch.FloatTensor,
      torch.DoubleTensor,
  }
  for _, tensor1 in ipairs(allTypes) do
    for _, tensor2 in ipairs(allTypes) do
      local t1 = tensor1():ones(10)
      local t2 = tensor2():ones(10)
      if tensor1 == tensor2 then
        meta_assert_success(subtester:assertTensorEq(t1, t2, 1e-6, MESSAGE))
      else
        meta_assert_failure(subtester:assertTensorEq(t1, t2, 1e-6, MESSAGE))
      end
    end
  end
end

function tests.test_assertTensorSizes()
  local t1 = torch.ones(2)
  local t2 = torch.ones(3)
  local t3 = torch.ones(1, 2)
  meta_assert_failure(subtester:assertTensorEq(t1, t2, 1e-6, MESSAGE))
  meta_assert_success(subtester:assertTensorNe(t1, t2, 1e-6, MESSAGE))
  meta_assert_failure(subtester:assertTensorEq(t1, t3, 1e-6, MESSAGE))
  meta_assert_success(subtester:assertTensorNe(t1, t3, 1e-6, MESSAGE))
end

function tests.test_assertTensorEq()
  local t1 = torch.randn(100, 100)
  local t2 = t1:clone()
  local t3 = torch.randn(100, 100)
  meta_assert_success(subtester:assertTensorEq(t1, t2, 1e-6, MESSAGE))
  meta_assert_failure(subtester:assertTensorEq(t1, t3, 1e-6, MESSAGE))
end

function tests.test_assertTensorNe()
  local t1 = torch.randn(100, 100)
  local t2 = t1:clone()
  local t3 = torch.randn(100, 100)
  meta_assert_success(subtester:assertTensorNe(t1, t3, 1e-6, MESSAGE))
  meta_assert_failure(subtester:assertTensorNe(t1, t2, 1e-6, MESSAGE))
  end

function tests.test_assertTensor_epsilon()
  local t1 = torch.rand(100, 100)
  local t2 = torch.rand(100, 100) * 1e-5
  local t3 = t1 + t2
  meta_assert_success(subtester:assertTensorEq(t1, t3, 1e-4, MESSAGE))
  meta_assert_failure(subtester:assertTensorEq(t1, t3, 1e-6, MESSAGE))
  meta_assert_success(subtester:assertTensorNe(t1, t3, 1e-6, MESSAGE))
  meta_assert_failure(subtester:assertTensorNe(t1, t3, 1e-4, MESSAGE))
end

function tests.test_assertStorageEq_alltypes()
  local allTypes = {
      torch.ByteStorage,
      torch.CharStorage,
      torch.ShortStorage,
      torch.IntStorage,
      torch.LongStorage,
      torch.FloatStorage,
      torch.DoubleStorage,
  }
  for _, storage1 in ipairs(allTypes) do
    for _, storage2 in ipairs(allTypes) do
      local s1 = storage1({1, 2, 3, 4})
      local s2 = storage2({1, 2, 3, 4})
      if storage1 == storage2 then
        meta_assert_success(subtester:assertStorageEq(s1, s2, 1e-6, MESSAGE))
      else
        meta_assert_failure(subtester:assertStorageEq(s1, s2, 1e-6, MESSAGE))
      end
    end
  end
end

function tests.test_assertStorageSizes()
  local t1 = torch.ones(2)
  local t2 = torch.ones(3)
  meta_assert_failure(subtester:assertTensorEq(t1, t2, 1e-6, MESSAGE))
end

function tests.test_assertStorageEq()
  local t1 = torch.randn(100, 100)
  local t2 = t1:clone()
  local t3 = torch.randn(100, 100)
  local s1 = t1:storage()
  local s2 = t2:storage()
  local s3 = t3:storage()
  meta_assert_success(subtester:assertStorageEq(s1, s2, 1e-6, MESSAGE))
  meta_assert_failure(subtester:assertStorageEq(s1, s3, 1e-6, MESSAGE))
end

function tests.test_assertStorageNe()
  local t1 = torch.randn(100, 100)
  local t2 = t1:clone()
  local t3 = torch.randn(100, 100)
  local s1 = t1:storage()
  local s2 = t2:storage()
  local s3 = t3:storage()
  meta_assert_success(subtester:assertStorageNe(s1, s3, 1e-6, MESSAGE))
  meta_assert_failure(subtester:assertStorageNe(s1, s2, 1e-6, MESSAGE))
end

function tests.test_assertTable()
  local tensor = torch.rand(100, 100)
  local t1 = {1, "a", key = "value", tensor = tensor, subtable = {"nested"}}
  local t2 = {1, "a", key = "value", tensor = tensor, subtable = {"nested"}}
  meta_assert_success(subtester:assertTableEq(t1, t2, MESSAGE))
  meta_assert_failure(subtester:assertTableNe(t1, t2, MESSAGE))
  for k, v in pairs(t1) do
    local x = "something else"
    t2[k] = nil
    t2[x] = v
    meta_assert_success(subtester:assertTableNe(t1, t2, MESSAGE))
    meta_assert_failure(subtester:assertTableEq(t1, t2, MESSAGE))
    t2[x] = nil
    t2[k] = x
    meta_assert_success(subtester:assertTableNe(t1, t2, MESSAGE))
    meta_assert_failure(subtester:assertTableEq(t1, t2, MESSAGE))
    t2[k] = v
    meta_assert_success(subtester:assertTableEq(t1, t2, MESSAGE))
    meta_assert_failure(subtester:assertTableNe(t1, t2, MESSAGE))
  end
end


function tests.test_genericEq()
  local tensor = torch.rand(100, 100)
  local sameTensor = tensor:clone()
  local t1 = {1, "a", key = "value", tensor = tensor, subtable = {"nested"}}
  local t2 = {1, "a", key = "value", tensor = sameTensor, subtable = {"nested"}}
  meta_assert_success(subtester:eq(t1, t2, MESSAGE, 1e-6))
  meta_assert_failure(subtester:ne(t1, t2, MESSAGE, 1e-6))
  for k, v in pairs(t1) do
    local x = "something else"
    t2[k] = nil
    t2[x] = v
    meta_assert_success(subtester:ne(t1, t2, MESSAGE, 1e-6))
    meta_assert_failure(subtester:eq(t1, t2, MESSAGE, 1e-6))
    t2[x] = nil
    t2[k] = x
    meta_assert_success(subtester:ne(t1, t2, MESSAGE, 1e-6))
    meta_assert_failure(subtester:eq(t1, t2, MESSAGE, 1e-6))
    t2[k] = v
    meta_assert_success(subtester:eq(t1, t2, MESSAGE, 1e-6))
    meta_assert_failure(subtester:ne(t1, t2, MESSAGE, 1e-6))
  end
  meta_assert_success(subtester:eq(3, 3, MESSAGE, 1e-6))
  meta_assert_failure(subtester:ne(3, 3, MESSAGE, 1e-6))
  meta_assert_success(subtester:ne(3, "3", MESSAGE, 1e-6))
  meta_assert_failure(subtester:eq(3, "3", MESSAGE, 1e-6))
  meta_assert_success(subtester:eq("3", "3", MESSAGE, 1e-6))
  meta_assert_failure(subtester:ne("3", "3", MESSAGE, 1e-6))
end

--[[ Returns a Tester with `numSuccess` success cases, `numFailure` failure
  cases, and with an error if `hasError` is true.
  Success and fail tests are evaluated with tester:eq
]]
local function genDummyTest(numSuccess, numFailure, hasError, ret)
  hasError = hasError or false
  ret = ret or false

  local dummyTester = totem.Tester()
  local dummyTests = totem.TestSuite()

  if numSuccess > 0 then
    function dummyTests.testDummySuccess()
      for i = 1, numSuccess do
        dummyTester:eq({1}, {1}, '', 0, ret)
      end
    end
  end

  if numFailure > 0 then
    function dummyTests.testDummyFailure()
      for i = 1, numFailure do
        dummyTester:eq({1}, {2}, '', 0, ret)
      end
    end
  end

  if hasError then
    function dummyTests.testDummyError()
      error('dummy error')
    end
  end

  return dummyTester:add(dummyTests)
end

function tests.test_assertEqRet()
  -- Create subtester - only a 'ret' value of false should trigger assertions

  local retTesterNoAssert        = genDummyTest(2, 0, false, true)
  local retTesterAssertSuccess   = genDummyTest(2, 0, false)
  local retTesterAssertFail      = genDummyTest(1, 1, false)
  local retTesterAssertError     = genDummyTest(1, 0, true)
  local retTesterAssertErrorFail = genDummyTest(0, 2, true)


  -- Change the write function so that the sub testers do not output anything
  local oldWrite = io.write
  io.write = function() end

  local success, msg = pcall(retTesterNoAssert.run, retTesterNoAssert)
  tester:asserteq(success, true,
                  "retTesterNoAssert should always return true (no asserts)")

  success, msg = pcall(retTesterAssertFail.run, retTesterAssertFail)
  tester:asserteq(success, false,
                  "retTesterAssertFail should return false (tests failed)")

  success, msg = pcall(retTesterAssertSuccess.run, retTesterAssertSuccess)
  tester:asserteq(success, true,
                  "retTesterAssertSuccess should return true (tests succeeded)")

  success, msg = pcall(retTesterAssertError.run, retTesterAssertError)
  tester:asserteq(success, false,
                  "retTesterAssertError should return false (tests with error)")

  success, msg = pcall(retTesterAssertErrorFail.run, retTesterAssertErrorFail)
  tester:asserteq(success, false,
      "retTesterAssertErrorFail should return false (tests with error + fail)")

  -- Restore write function
  io.write = oldWrite

  tester:asserteq(retTesterNoAssert.countasserts, 0,
                  "retTesterNoAssert should not have asserted")

  tester:asserteq(retTesterAssertFail.countasserts, 2,
                  "retTesterAssertFail should have asserted twice")

  tester:asserteq(retTesterAssertSuccess.countasserts, 2,
                  "retTesterAssertSuccess should have asserted twice")

  tester:asserteq(retTesterAssertError.countasserts, 1,
                  "retTesterAssertError should have asserted once")

  tester:asserteq(retTesterAssertErrorFail.countasserts, 2,
                  "retTesterAssertErrorFail should have asserted twice")
end

local function good_fn() end
local function bad_fn() error("muahaha!") end

function tests.test_assertError()
  meta_assert_success(subtester:assertError(bad_fn, MESSAGE))
  meta_assert_failure(subtester:assertError(good_fn, MESSAGE))
end

function tests.test_assertNoError()
  meta_assert_success(subtester:assertNoError(good_fn, MESSAGE))
  meta_assert_failure(subtester:assertNoError(bad_fn, MESSAGE))
end

function tests.test_assertErrorPattern()
  meta_assert_success(subtester:assertErrorPattern(bad_fn, "haha", MESSAGE))
  meta_assert_failure(subtester:assertErrorPattern(bad_fn, "hehe", MESSAGE))
end

function tests.test_TensorEqChecksEmpty()
  local t1 = torch.DoubleTensor()
  local t2 = t1:clone()
  local t3 = torch.randn(100,100)

  local success, msg = totem.areTensorsEq(t1, t2, 1e-5)
  tester:assert(success, "areTensorsEq should return true")

  local success, msg = totem.areTensorsEq(t1, t3, 1e-5)
  tester:assert(not success, "areTensorsEq should return false")
  tester:assertErrorPattern(function() totem.assertTensorEq(t1, t3, 1e-5) end,
                     "different dimensions",
                     "wrong error message for tensors of different dimensions")

  tester:assertNoError(function() totem.assertTensorEq(t1, t2, 1e-5) end,
                       "assertTensorEq not raising an error")

end

function tests.test_TensorEqChecks()
  local t1 = torch.randn(100,100)
  local t2 = t1:clone()
  local t3 = torch.randn(100,100)

  local success, msg = totem.areTensorsEq(t1, t2, 1e-5)
  tester:assert(success, "areTensorsEq should return true")

  local success, msg = totem.areTensorsEq(t1, t3, 1e-5)
  tester:assert(not success, "areTensorsEq should return false")
  tester:asserteq(type(msg), 'string', "areTensorsEq should return a message")

  tester:assertNoError(function() totem.assertTensorEq(t1, t2, 1e-5) end,
                       "assertTensorEq not raising an error")
  tester:assertError(function() totem.assertTensorEq(t1, t3, 1e-5) end,
                     "assertTensorEq not raising an error")
end

function tests.test_TensorNeChecks()
  local t1 = torch.randn(100, 100)
  local t2 = t1:clone()
  local t3 = torch.randn(100, 100)

  local success, msg = totem.areTensorsNe(t1, t3, 1e-5)
  tester:assert(success, "areTensorsNe should return true")

  local success, msg = totem.areTensorsNe(t1, t2, 1e-5)
  tester:assert(not success, "areTensorsNe should return false")
  tester:asserteq(type(msg), 'string', "areTensorsNe should return a message")

  tester:assertNoError(function() totem.assertTensorNe(t1, t3, 1e-5) end,
                       "assertTensorNe not raising an error")
  tester:assertError(function() totem.assertTensorNe(t1, t2, 1e-5) end,
                     "assertTensorNe not raising an error")
end

function tests.test_TensorArgumentErrorMessages()
  local t = torch.ones(1)
  local funcs = {
      totem.areTensorsEq,
      totem.areTensorsNe,
      totem.assertTensorEq,
      totem.assertTensorNe,
  }

  for _, fn in ipairs(funcs) do
    tester:assertErrorPattern(function() fn(nil, t, 0) end, "First argument")
    tester:assertErrorPattern(function() fn(t, nil, 0) end, "Second argument")
    tester:assertErrorPattern(function() fn(t, t, "nan") end, "Third argument")
  end
end

function tests.testSuite_duplicateTests()
    function createDuplicateTests()
        local tests = totem.TestSuite()
        function tests.testThis()
        end
        function tests.testThis()
        end
    end
    tester:assertErrorPattern(createDuplicateTests,
                              "Test testThis is already defined.")
end

function tests.test_checkGradientsAcceptsGenericOutput()
    local Mod = torch.class('totem.dummyClass', 'nn.Module')
    function Mod:updateOutput(input)
        self.output = {
            [1] = {
                [1] = torch.randn(3, 5),
                [2] = 1,
                strKey = 3,
            },
            [2] = 1,
            [3] = torch.randn(3, 5),
            strKey = 4
        }
        return self.output
    end
    function Mod:updateGradInput(input, gradOutput)
        self.gradInput = input:clone():fill(0)
        return self.gradInput
    end
    local mod = totem.dummyClass()
    totem.nn.checkGradients(tester, mod, torch.randn(5, 5), 1e-6)
end

function tests.test_checkTypePreservesSharing()
  local mod = nn.Linear(10, 10)
  local clonedMod = mod:clone('weights', 'bias', 'gradWeight', 'gradBias')
  local seq = nn.Sequential():add(mod):add(clonedMod)
  totem.nn.checkTypePreservesSharing(tester, seq)

  local Mod = torch.class('totem.classWhichBreaksSharing', 'nn.Linear')
  function Mod:type(type)
    self.weight = self.weight:type(type)
    self.bias = self.bias:type(type)
  end
  local mod = totem.classWhichBreaksSharing(10, 10)
  local clonedMod = mod:clone('weight', 'bias', 'gradWeight', 'gradBias')
  local seq = nn.Sequential():add(mod):add(clonedMod)
  local dummyTest = totem.Tester()
  local test = {}
  function test.checkShouldFail()
      totem.nn.checkTypePreservesSharing(dummyTest, seq, 'torch.FloatTensor')
  end
  dummyTest:add(test)
  -- change io.write behavior to not output sub-tests
  local oldWrite = io.write
  io.write = function() end
  local testSuccess = pcall(function() return dummyTest:run() end)
  io.write = oldWrite
  tester:asserteq(testSuccess, false, 'expect test failure, mod breaks sharing')
end

function tests.test_returnWithErrorFailureSuccess()
  local emptyTest    = genDummyTest(0, 0, false)
  local sucTest      = genDummyTest(1, 0, false)
  local multSucTest  = genDummyTest(4, 0, false)
  local failTest     = genDummyTest(0, 1, false)
  local errTest      = genDummyTest(0, 0, true)

  local errFailTest  = genDummyTest(0, 1, true)
  local errSucTest   = genDummyTest(1, 0, true)
  local failSucTest  = genDummyTest(1, 1, false)

  local failSucErrTest  = genDummyTest(1, 1, true)

  -- change io.write behavior to not output sub-tests
  local oldWrite = io.write
  io.write = function() end

  local success, msg = pcall(emptyTest.run, emptyTest)
  tester:asserteq(success, true, "pcall should succeed for empty tests")

  local success, msg = pcall(sucTest.run, sucTest)
  tester:asserteq(success, true, "pcall should succeed for 1 successful test")

  local success, msg = pcall(multSucTest.run, multSucTest)
  tester:asserteq(success, true, "pcall should succeed for 2+ successful tests")

  local success, msg = pcall(failTest.run, failTest)
  tester:asserteq(success, false, "pcall should fail for tests with failure")

  local success, msg = pcall(errTest.run, errTest)
  tester:asserteq(success, false, "pcall should fail for tests with error")

  local success, msg = pcall(errFailTest.run, errFailTest)
  tester:asserteq(success, false, "pcall should fail for error+fail tests")

  local success, msg = pcall(errSucTest.run, errSucTest)
  tester:asserteq(success, false, "pcall should fail for error+success tests")

  local success, msg = pcall(failSucTest.run, failSucTest)
  tester:asserteq(success, false, "pcall should fail for fail+success tests")

  local success, msg = pcall(failSucErrTest.run, failSucErrTest)
  tester:asserteq(success, false, "pcall should fail for fail+success+err test")

  -- restoring io.write original behavior
  io.write = oldWrite
end

function tests.test_setUp()
    tester:asserteq(test_name_passed_to_setUp, 'test_setUp')
    for key, value in pairs(tester.tests) do
        tester:assertne(key, '_setUp')
    end
end


function tests.test_tearDown()
    for key, value in pairs(tester.tests) do
        tester:assertne(key, '_tearDown')
    end
end


function tests._setUp(name)
    test_name_passed_to_setUp = name
    calls_to_setUp = calls_to_setUp + 1
end


function tests._tearDown(name)
    calls_to_tearDown = calls_to_tearDown + 1
end


tester:add(tests):run()


-- Additional tests to check that _setUp and _tearDown were called.
local test_count = tablex.size(tester.tests)
local postTests = totem.TestSuite()
local postTester = totem.Tester()

function postTests.test_setUp(tester)
    postTester:asserteq(calls_to_setUp, test_count,
                        "Expected " .. test_count .. " calls to _setUp")
end

function postTests.test_tearDown()
    postTester:asserteq(calls_to_tearDown, test_count,
                       "Expected " .. test_count .. " calls to _tearDown")
end


postTester:add(postTests):run()
