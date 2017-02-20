[![Build Status](https://travis-ci.org/deepmind/torch-totem.svg?branch=master)](https://travis-ci.org/deepmind/torch-totem)

# Totem - Torch test module

NOTE: Totem has been merged into torch.Tester, and it is now recommended you
use torch.Tester directly. See
[here](https://github.com/torch/torch7/blob/master/doc/tester.md) for
documentation.

## Basic test description

A test script can be written as follows:

    require 'totem'

    local tests = totem.TestSuite()

    local tester = totem.Tester()

    function tests.TestA()
      local a = 10
      local b = 10
      tester:asserteq(a, b, 'a == b')
      tester:assertne(a,b,'a ~= b')
    end

    function tests.TestB()
      local a = 10
      local b = 9
      tester:assertlt(a, b, 'a < b')
      tester:assertgt(a, b, 'a > b')
    end

    return tester:add(tests):run()

The command `totem-init` can be used to generate an empty test.

## Set-up and tear-down functions
It is sometimes useful to introduce initialization or destruction code that
is shared among all tests in a test suite. Totem allows you to define special
`_setUp` and `_tearDown` test functions. The `_setUp` function is interpreted
as a set-up function that is called *before* every test. Similarly, the
`_tearDown` is interpreted as a tear-down function that is called *after*
every test. The following example illustrates this.

    require 'totem'

    local tests = totem.TestSuite()

    local tester = totem.Tester()

    function tests.TestA()
      ....
    end

    function tests.TestB()
      ....
    end

    function tests._setUp(testName)
      .... set-up / initialization code
    end

    function tests._tearDown(testName)
      .... tear-down / clean-up up code
    end

    return tester:add(tests):run()


## Command-line usage

When running the script from the command-line you get a number of options:

```sh
Run tests

Usage:

  ./simple.lua [options] [test1 [test2...] ]

Options:

  --list print the names of the available tests instead of running them.
  --log-output (optional file-out) redirect compact test results to file.
        This contains one line per test in the following format:
        name #passed-assertions #failed-assertions #exceptions
  --no-colour suppress colour output
  --summary print only pass/fail status rather than full error messages.
  --full-tensors when printing tensors, always print in full even if large.
        Otherwise just print a summary for large tensors.
  --early-abort (optional boolean) abort execution on first error.
  --rethrow (optional boolean) errors make the program crash and propagate up
        the stack

If any test names are specified only the named tests are run. Otherwise
all the tests are run.
```

Additionally the script `totem-run` can be used to run all test files (i.e.
files with names of the form `test*.lua` in a directory (by default the current
directory).

## Nesting tests

It's possible to nest test cases. Individual test files are still assumed to be
runnable as stand-alone scripts, but a test case can include the outputs of
such files. For example

    require 'totem'

    local tester = totem.Tester()
    tester:add('test_nn.lua')
    tester:add('test_simple.lua')
    tester:add('test_tensor.lua')
    return tester:run()

will first run all the tests in each of the listed test files and then report
the overall test results. Each test is considered to pass only if all of its
subtests pass.


## Running several tests

The script `scripts/totem-run` will run all the files with a filename
`test*.lua` that are inside the folder specified by the argument `--folder`
(current folder by default). The rest of the arguments are passed to all the
individual tests.

Example:

```sh
totem-run --folder tests --summary
```
