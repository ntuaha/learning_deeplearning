from .VARIABLE import Variable
from .FUNCTION import *
import numpy as np


def numerical_diff(f,x,eps=1e-4):
    x0 = Variable(np.array(x.data - eps))
    x1 = Variable(np.array(x.data + eps))
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data)/ (2 * eps)

