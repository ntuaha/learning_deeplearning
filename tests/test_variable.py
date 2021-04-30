import pytest
# https://stackoverflow.com/questions/10253826/path-issue-with-pytest-importerror-no-module-named-yadayadayada
import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')
from src.VARIABLE import Variable
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


