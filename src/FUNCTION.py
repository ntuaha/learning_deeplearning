from .VARIABLE import Variable

class Function:
    def __call__(self,input):
        x = input.data
        y = x ** 2
        output = Variable(y)
        return output