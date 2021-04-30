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
    assert y == Variable(np.exp(x.data))

def test_check_square_exp_square():
    # test for composite function
    x = Variable(np.array(3))
    A = Square()
    B = Exp()
    C = Square()
    y = C(B(A(x)))
    assert y == Variable(np.exp(3*3)**2)

def test_square_diff():
    x = Variable(np.array(2))
    f = Square()
    y = f(x)
    gy = f.backward(np.array(3))
    assert gy == np.array(2*2*3)

def test_exp_diff():
    x = Variable(np.array(2))
    f = Exp()
    y = f(x)
    gy = f.backward(np.array(3))
    assert gy == np.array(np.exp(2)*3)    
