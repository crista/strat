#
# VATS
# Crista V. Lopes
#
import torch
import torch.nn as nn
from test import run_simple_tests, run_robust_evaluation

# --- 1. ReLU GLU (Standard) ---
class ReLUGLU(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_model)
        self.w_val = nn.Linear(d_model, d_model)
        self.act = nn.ReLU() 
        
    def forward(self, x):
        gate = self.act(self.w_gate(x))
        val = self.w_val(x)
        return gate * val

class GeometricTransformerReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = 6

        # We work with raw values to avoid (x/100)*100 precision errors.
        self.SCALE = 1.0 
        
        self.W_q = nn.Linear(6, 6, bias=False)
        self.W_k = nn.Linear(6, 6, bias=False)
        self.W_v = nn.Linear(6, 6, bias=False)
        self.ffn = ReLUGLU(6) 
        self.W_out = nn.Linear(6, 1, bias=False)

        self.hand_code_weights()

    def hand_code_weights(self):
        with torch.no_grad():
            # --- 1. ATTENTION ---
            # Q/K only look at Flags (D1, D4), which are always 1.0.
            # So the magnitude of D0 (Value) doesn't matter.
            # Q and K "meet" in D4
            w_q = torch.zeros(6, 6); w_q[4, 1] = 10.0
            w_k = torch.zeros(6, 6); w_k[4, 4] = 10.0
            self.W_q.weight.copy_(w_q)
            self.W_k.weight.copy_(w_k)
            
            # V only broadcasts for operators (D2, D3),
            # because no one is looking for numbers
            w_v = torch.zeros(6, 6)
            w_v[2, 2] = 1.0; w_v[3, 3] = 1.0
            self.W_v.weight.copy_(w_v)

            # --- 2. FFN (Dual-Gate) ---
            w_gate = torch.zeros(6, 6); b_gate = torch.zeros(6)
            
            # Gate A (D5): Active on IsSub (D3)
            w_gate[5, 3] = 10.0
            b_gate[5] = -8.0
            
            # Gate B (D4): Active on Mixed (D2 and D3) or Pure Sub (D3)
            w_gate[4, 2] = -5.0
            w_gate[4, 3] = 5.0
            b_gate[4] = 0.0
            
            self.ffn.w_gate.weight.copy_(w_gate)
            self.ffn.w_gate.bias.copy_(b_gate)
            
            # --- VAL ---
            w_val = torch.zeros(6, 6); b_val = torch.zeros(6)
            w_val[5, 0] = 1.0
            w_val[4, 0] = 1.0
            self.ffn.w_val.weight.copy_(w_val)
            self.ffn.w_val.bias.copy_(b_val)
            
            # --- 4. READOUT (The Fix) ---
            w_out = torch.zeros(1, 6)
            
            # The Logic
            w_out[0, 0] = 1.0        # Value
            w_out[0, 5] = 2.0 / 3.0  # Gate A
            w_out[0, 4] = -2.0 / 3.0 # Gate B
            
            # The Noise Cancellation
            # Dim 4 has 'IsOp' flag (1.0). This adds a bias of -2/3 per op.
            # We cancel it by adding +2/3 from IsAdd(2) and IsSub(3).
            w_out[0, 2] = 2.0 / 3.0
            w_out[0, 3] = 2.0 / 3.0
            
            self.W_out.weight.copy_(w_out)

    def forward(self, x):
        seq_len = x.shape[1]
        
        Q = self.W_q(x); K = self.W_k(x)
        scores = torch.matmul(Q, K.transpose(-2, -1))
        
        indices = torch.arange(seq_len, device=x.device)
        dist = indices.view(-1, 1) - indices.view(1, -1)
        scores -= dist.float().abs()
        
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask == 1, float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)
        V = self.W_v(x)
        Z = torch.matmul(attn_weights, V)
        
        combined = x + Z
        ffn_out = self.ffn(combined)
        x_final = x + ffn_out
        
        res = self.W_out(x_final.sum(dim=1))
        return res.squeeze(-1)
    
if __name__ == "__main__":
    model = GeometricTransformerReLU()
    run_simple_tests(model)
    run_robust_evaluation(model)