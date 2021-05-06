import numpy as np
class Variable:
    def __init__(self,data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f'{type(data)} is not supported')
        self.data = data
        # P27
        self.grad = None
        # P31
        self.creator = None
        # P83
        self.generation = 0


    def __eq__(self,that):
        # how to define a good eq function
        # ref: https://openhome.cc/Gossip/Python/ObjectEquality.html
        if not isinstance(that, Variable):
            return False
        return self.data == that.data
    
    def set_creator(self,func):
        # P31
        self.creator = func
        # P83
        self.generation = func.generation + 1
    
    def cleargrad(self):
        self.grad = None
        
    def backward(self):
        '''
        # P34
        f = self.creator
        if f is not None:
            x = f.input
            x.grad = f.backward(self.grad)
            x.backward()
        # P38
        '''
        # P43
        # P67
        # 預設用的 grad
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        # 丟入創造這個 Variable 的上游函數
        # P86
        funcs = []
        seen_set = set()
        def add_func(f):
            if f not in seen_set: # 剔除重複的f
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x:x.generation)
        
        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            # 取得下游的 grad            
            gys = [output.grad for output in f.outputs]
            # 計算上游函數 grad
            gxs = f.backward(*gys)
            if not isinstance(gxs,tuple):
                gxs = (gxs,)
            # 修改上游函數的 grad
            for x,gx in zip(f.inputs,gxs):
                if x.grad is None:                    
                    x.grad = gx            
                else:
                    x.grad = x.grad + gx
                if x.creator is not None:
                    add_func(x.creator)
    