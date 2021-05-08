from memory_profiler import profile
import numpy as np
import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

from src.FUNCTION import Function
from src.VARIABLE import Variable
from src.Config import Config
from src.util import *

@profile
def my_func():
    a = [1] * (10 ** 6)
    b = [2] * (2 * 10 ** 7)
    del b
    return a

@profile
def test_memory():
    for i in range(100):
        x = Variable(np.random.rand(10000))
        for j in range(100):
            x = square(x)       

@profile
# P103
def test_config_v1():
    Config.enable_backprob = True
    x = Variable(np.ones((100,100,100)))
    y = square(square(square(x)))
    y.backward()


@profile
# P103
def test_config_v2():
    Config.enable_backprob = False
    x = Variable(np.ones((100,100,100)))
    y = square(square(square(x)))



if __name__ == "__main__":
    #test_memory()
    test_config_v2()