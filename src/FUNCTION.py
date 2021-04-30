from .VARIABLE import Variable
import numpy as np

class Function:
    def __call__(self,input):
        x = input.data
        # first version y = x ** 2
        # for general purpose
        y = self.forward(x)
        output = Variable(y)
        return output

    def forward(self,x):
        raise NotImplementedError()

class Square(Function):
    # P8
    def forward(self,x):
        return x ** 2

class Exp(Function):
    # P11
    def forward(self,x):
        return np.exp(x)