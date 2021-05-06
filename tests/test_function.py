import pytest
# https://stackoverflow.com/questions/10253826/path-issue-with-pytest-importerror-no-module-named-yadayadayada
import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')
from src.VARIABLE import Variable
from src.FUNCTION import *
from src.util import *
import numpy as np


def test_as_array():
    x = np.array(1.0)
    assert type(x) == np.ndarray
    y = 1.0
    assert type(y) != np.ndarray

def test_check_square():
    # P8
    # F(x) == x**2
    x = Variable(np.array(20.0))
    #f = Function()
    f = Square()
    y = f(x)
    assert y == Variable(np.array(20*20))

def test_check_exp():
    x = Variable(np.array(3.0))
    f = Exp()
    y = f(x)
    assert y == Variable(np.array(np.exp(x.data)))

def test_check_square_exp_square():
    # test for composite function
    x = Variable(np.array(3.0))
    A = Square()
    B = Exp()
    C = Square()
    y = C(B(A(x)))
    assert y == Variable(np.array(np.exp(3*3)**2))

def test_square_diff():
    x = Variable(np.array(2.0))
    f = Square()
    y = f(x)
    gy = f.backward(np.array(3))
    assert gy == np.array(2*2*3)

def test_exp_diff():
    x = Variable(np.array(2.0))
    f = Exp()
    y = f(x)
    gy = f.backward(np.array(3.0))
    assert gy == np.array(np.exp(2.0)*3)


def test_auto_link():
    # P32
    x = Variable(np.array(1.5))
    A = Exp()
    B = Square()
    C = Square()
    a = A(x)
    b = B(a)
    y = C(b)    
    assert y.creator == C
    assert y.creator.inputs[0] == b
    assert y.creator.inputs[0].creator == B
    assert y.creator.inputs[0].creator.inputs[0] == a
    assert y.creator.inputs[0].creator.inputs[0].creator == A
    assert y.creator.inputs[0].creator.inputs[0].creator.inputs[0] == x

def test_auto_grad_and_numerical_grad():
    # P50
    x = Variable(np.random.rand(1))
    y = square(x)
    y.backward()
    num_grad = numerical_diff(square,x)
    assert np.allclose(x.grad,num_grad)

