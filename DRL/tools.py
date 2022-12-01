import copy

class mem_state():
    def __init__(self, **args):
        self.__dict__ = args
    
    def vars(self):
        return self.__dict__
    
    def vals(self):
        return self.__dict__.copy()

    def merge(self, **args):
        c = self.__dict__.copy()
        c.update(args)
        return mem_state(**c)

    def update(self, **args):
        self.__dict__.update(args)

    def copy(self):
        return copy.copy(self)
    
    def __iter__(self):
        return self.__dict__.__iter__()

    def keys(self):
        return self.__dict__.__iter__()

    def __getitem__(self, __name: str):
        return self.__dict__[__name]

    def __setitem__(self, __name: str, __value) -> None:
        self.__dict__[__name] = __value


class mem_stack(mem_state):
    def __init__(self, stack):
        keys = stack[0].keys()
        self.__dict__ = {k:[m[k] for m in stack] for k in keys}


