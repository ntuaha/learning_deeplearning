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

def test_v2_function():
    x1 = Variable(np.array(1))
    x2 = Variable(np.array(2))
    y = square(x1)
    y.backward()
    num_grad = numerical_diff(square,x1)
    assert np.allclose(x1.grad,num_grad)

def test_add():
    x1 = Variable(np.array(1))
    x2 = Variable(np.array(2))
    y = add(x1,x2)
    y.backward()
    assert (x1.grad == 1) and (x2.grad == 1)

def test_add_two_same_variable():
    x1 = Variable(np.array(1))
    x2 = Variable(np.array(2))
    y = add(x1,x1)
    y.backward()
    assert (x1.grad == 2)


def test_mutiple():
    x1 = Variable(np.array(1))
    x2 = Variable(np.array(2))
    x3 = Variable(np.array(3))
    y = add(mutiple(x3,x2),x1)
    y.backward()
    assert (y.data == np.array(7)) and (x2.grad == 3) and (x3.grad==2)


def test_double_usage_same_variable():
    x = Variable(np.array(1))
    y = add(add(x,x),x)    
    y.backward()
    assert (x.grad == 3)
    x.cleargrad()
    y = add(x,x)
    y.backward()
    assert (x.grad == 2)

def test_complex_path():
    x = Variable(np.array(2))
    a = square(x)
    y = add(square(a),square(a))
    y.backward()
    assert y.data == np.array(32) and x.grad == np.array(64)

def test_mutiple2():
    # P117
    x1 = Variable(np.array(1))
    x2 = Variable(np.array(2))
    x3 = Variable(np.array(3))
    y = x3 * x2 + x1
    y.backward()
    assert (y.data == np.array(7)) and (x2.grad == 3) and (x3.grad==2)

def test_inputs():
    # P120
    x = Variable(np.array(2)) + np.array(3)
    assert np.array(5) == x.data
    # P121    
    x = Variable(np.array(2)) + 3.0
    assert np.array(5) == x.data
    # P123
    x = 3.0 * Variable(np.array(2))
    assert np.array(6) == x.data    
    # P123
    x = np.array([3.0]) * Variable(np.array([2]))
    assert np.array(6) == x.data      