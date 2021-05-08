import pytest
# https://stackoverflow.com/questions/10253826/path-issue-with-pytest-importerror-no-module-named-yadayadayada
import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')
from src.VARIABLE import Variable
from src.FUNCTION import *
from src.util import *
import numpy as np


def test_check_data_unchanged():
    # P4
    data = np.array(1.0)
    x = Variable(data)
    assert x.data == np.array(1.0)

def test_check_data_is_different():
    # P4
    data = np.array(1.0)
    x = Variable(data)
    assert x.data != np.array(2.0)

def test_check_eq_function():
    x = Variable(np.array(2.0))
    y = Variable(np.array(2.0))
    assert x == y

def test_check_not_eq_function():
    x = Variable(np.array(2.0))
    y = Variable(np.array(3.0))
    assert x != y

def test_auto_grad():
    # P34
    v = 0.5
    grad = 1
    x = Variable(np.array(v))
    A = Square()
    B = Exp()
    C = Square()
    a = A(x)
    b = B(a)
    y = C(b)
    y.grad = np.array(grad)
    C.inputs[0].grad = C.backward(y.grad)
    B.inputs[0].grad = B.backward(C.inputs[0].grad)
    A.inputs[0].grad = A.backward(B.inputs[0].grad)

    x2 = Variable(np.array(v))
    A2 = Square()
    B2 = Exp()
    C2 = Square()
    y2 = C2(B2(A2(x2)))
    y2.grad = np.array(grad)
    y2.backward()
    assert x2.grad == x.grad

def test_default_grad():
    x = Variable(np.array(2.0))
    y = square(x)
    y.backward()
    assert y.grad != np.array(1)
    # P101
    assert y.grad is None

def test_retain_grad():
    x0 = Variable(np.array(1))
    x1 = Variable(np.array(1))
    t = add(x0,x1)
    y = add(x0,t)
    y.backward()
    assert (y.grad,t.grad) == (np.array(1),None)
    assert (x0.grad,x1.grad) == (2,np.array(1)) # y = 2x0 + x1


def test_variable_name():
    x = Variable(np.array(1))
    assert x.name == None
    v_name = '1244'
    x = Variable(np.array(1),v_name)
    assert x.name == v_name
