import pytest
# https://stackoverflow.com/questions/10253826/path-issue-with-pytest-importerror-no-module-named-yadayadayada
import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')
from src.VARIABLE import Variable
from src.FUNCTION import Function
import numpy as np


def test_check_function_work():
    # P8
    # F(x) == x**2
    x = Variable(np.array(20))
    f = Function()
    y = f(x)
    assert y == Variable(np.array(20)**2)

