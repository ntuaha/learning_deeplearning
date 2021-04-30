class Variable:
    def __init__(self,data):
        self.data = data
        # P27
        self.grad = None
        # P31
        self.creator = None

    def __eq__(self,that):
        # how to define a good eq function
        # ref: https://openhome.cc/Gossip/Python/ObjectEquality.html
        if not isinstance(that, Variable):
            return False
        return self.data == that.data
    
    def set_creator(self,func):
        # P31
        self.creator = func
    