from .VARIABLE import Variable
import numpy as np

class Function:
    def __call__(self,input):
        self.input = input
        x = input.data
        # first version y = x ** 2
        # for general purpose
        y = self.forward(x)
        output = Variable(y)
        # P31
        output.set_creator(self)
        self.output = output
        return output

    def forward(self,x):
        raise NotImplementedError()

    def backward(self,gy):
        raise NotImplementedError()
class Square(Function):
    # P8
    def forward(self,x):
        return x ** 2
    # P26
    def backward(self,gy):
        # 2 * x * x'
        return 2 * self.input.data * gy

class Exp(Function):
    # P11
    def forward(self,x):
        return np.exp(x)
    # P26
    def backward(self,gy):
        # exp(x) * x'
        return np.exp(self.input.data)*gy