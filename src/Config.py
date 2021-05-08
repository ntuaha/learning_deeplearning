import contextlib

class Config:
    enable_backprob = True

@contextlib.contextmanager
def using_config(name,value):
    # 保留變數舊值
    old_value = getattr(Config,name)
    # 置換新值
    setattr(Config,name,value)
    try:
        yield
    finally:
        setattr(Config,name,old_value)

def no_grad():
    return using_config('enable_backprob',False)
