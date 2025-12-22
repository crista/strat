#
# VATS
# Crista V. Lopes
#
import torch

def get_tensor_embedding(token):
    # [Value, IsNum, IsAdd, IsSub, IsOp, Reserved]
    x = torch.zeros(6)
    if isinstance(token, (int, float)):
        x[0] = float(token); x[1] = 1.0
    elif token == '+':
        x[2] = 1.0; x[4] = 1.0
    elif token == '-':
        x[3] = 1.0; x[4] = 1.0
    return x

