import pytest
# https://stackoverflow.com/questions/10253826/path-issue-with-pytest-importerror-no-module-named-yadayadayada
import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')
from src.VARIABLE import Variable
from src.FUNCTION import *
from src.util import *
import numpy as np

def test_numerical_diff():
    # P16
    f = Square()
    x = Variable(np.array(2))
    y = numerical_diff(f,x)
    check_1 = Variable((f(Variable(np.array(2+1e-4))).data - f(Variable(np.array(2-1e-4))).data) / (2 * 1e-4))
    assert check_1.data == y

def test_square_and_exp():
    x = Variable(np.array(2))
    y = square(exp(square(x)))
    y2 = Square()(Exp()(Square()(x)))
    assert y2 == y 