import pytest
# https://stackoverflow.com/questions/10253826/path-issue-with-pytest-importerror-no-module-named-yadayadayada
import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')
from src.VARIABLE import Variable
from src.FUNCTION import *
from src.Config import *
from src.util import *
import numpy as np


def test_config_with():
    with using_config('enable_backprob',False):
        assert Config.enable_backprob == False
    assert Config.enable_backprob == True

def test_no_grad():
    with no_grad():
        assert Config.enable_backprob == False
    assert Config.enable_backprob == True