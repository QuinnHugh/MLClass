"""append"""
import torch as tc

def append(tg_tensor, tail_tensor):
    """append in tensor"""
    if tg_tensor is None:
        return tail_tensor
    dim_tg = len(tg_tensor.size())
    dim_tail = len(tail_tensor.size())
    if dim_tg == 1 and dim_tail == 1:
        result = tc.stack((tg_tensor, tail_tensor))
    elif dim_tail == 1:
        result = tc.cat((tg_tensor, tail_tensor.unsqueeze(0)))
    else:
        result = tc.cat((tg_tensor, tail_tensor))
    return result
