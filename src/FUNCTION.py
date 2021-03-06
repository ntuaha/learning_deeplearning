from .VARIABLE import Variable
from .Config import Config
import numpy as np
import weakref


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

def as_variable(obj):
    if isinstance(obj,Variable):
        return obj
    return Variable(obj)

class Function:
    def __call__(self,*inputs):
        # P120
        inputs = [as_variable(x) for x in inputs]
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        # P46
        outputs = [Variable(as_array(y)) for y in ys]
        # P102
        if Config.enable_backprob:
            # P84
            self.generation = max([x.generation for x in inputs])
            # P31
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self,xs):
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
        return 2 * self.inputs[0].data * gy

class Exp(Function):
    # P11
    def forward(self,x):
        return np.exp(x)
    # P26
    def backward(self,gy):
        # exp(x) * x'
        return np.exp(self.inputs[0].data)*gy

class Add(Function):
    def forward(self,x1,x2):
        return x1+x2
    
    def backward(self,gy):
        return gy,gy

class Mutiple(Function):
    def forward(self,x1,x2):
        return x1*x2
    
    def backward(self,gy):
        return self.inputs[1].data*gy,self.inputs[0].data*gy


def square(x):
    # P42
    return Square()(x)

def exp(x):
    # P42
    return Exp()(x)

def add(x1,x2):
    # P42
    # P121
    return Add()(x1,as_array(x2))

def mutiple(x1,x2):
    # P42
    # P121
    return Mutiple()(x1,as_array(x2))

Variable.__add__ = add
Variable.__mul__ = mutiple
# P122
Variable.__radd__ = add
Variable.__rmul__ = mutiple