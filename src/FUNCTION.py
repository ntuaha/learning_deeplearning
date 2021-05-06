from .VARIABLE import Variable
import numpy as np


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class Function:
    def __call__(self,*inputs):
        
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        # P46
        outputs = [Variable(as_array(y)) for y in ys]
        # P84
        self.generation = max([x.generation for x in inputs])
        # P31
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
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
        return self.inputs[1].data,self.inputs[0].data