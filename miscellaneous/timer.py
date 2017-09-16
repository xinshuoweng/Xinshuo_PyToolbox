# Author: Xinshuo Weng
# Email: xinshuo.weng@gmail.com

# visualize the blob of caffe net

import time

import __init__paths__
from check import isfloat, isinteger

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff



def format_time(seconds):
    '''
    format second to human readable way
    '''
    assert isfloat(seconds) or isinteger(seconds), 'input should be an integer or floating number to represent number of seconds'
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return '%d:%02d:%02d' % (h, m, s)


def get_timestring():
    return time.strftime('%Y%m%d_%Hh%Mm%Ss')
