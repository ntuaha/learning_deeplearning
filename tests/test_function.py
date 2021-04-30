import pytest
# https://stackoverflow.com/questions/10253826/path-issue-with-pytest-importerror-no-module-named-yadayadayada
import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')
from src.VARIABLE import Variable
from src.FUNCTION import *
import numpy as np


def test_check_square():
    # P8
    # F(x) == x**2
    x = Variable(np.array(20))
    #f = Function()
    f = Square()
    y = f(x)
    assert y == Variable(np.array(20)**2)

def test_check_exp():
    x = Variable(np.array(3))
    f = Exp()
    y = f(x)
    return y == Variable(np.exp(x.data))

def test_check_square_exp_square():
    # test for composite function
    x = Variable(np.array(3))
    A = Square()
    B = Exp()
    C = Square()
    y = C(B(A(x)))
    return y == Variable(np.array(np.exp(np.array(3))))
