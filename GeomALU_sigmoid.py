import torch
import torch.nn as nn

from test import run_simple_tests, run_robust_evaluation

# --- 1. Sigmoid GLU (Standard) ---
class SigmoidGLU(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_model)
        self.w_val = nn.Linear(d_model, d_model)
        self.act = nn.Sigmoid() 
        
    def forward(self, x):
        gate = self.act(self.w_gate(x))
        val = self.w_val(x)
        return gate * val

class GeometricTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = 6
        
        # We work with raw values to avoid (x/100)*100 precision errors.
        self.SCALE = 1.0 
        
        self.W_q = nn.Linear(6, 6, bias=False)
        self.W_k = nn.Linear(6, 6, bias=False)
        self.W_v = nn.Linear(6, 6, bias=False)
        self.ffn = SigmoidGLU(6) 
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

            # --- 2. FFN (single gate) ---
            w_gate = torch.zeros(6, 6); b_gate = torch.zeros(6)
            
            # -- Gate --
            GAIN = 100.0
            # Target (IsNum+IsSub): 1.0 + 0.88 = 1.88 -> 188.0
            # Noise (IsNum+IsAdd):  1.0 + 0.12 = 1.12 -> 112.0
            # Bias: -150.0
            # It's triggered when IsNum (D1) AND IsSub (D3) are both true
            w_gate[5, 1] = GAIN
            w_gate[5, 3] = GAIN
            b_gate[5] = -1.5 * GAIN
            
            self.ffn.w_gate.weight.copy_(w_gate)
            self.ffn.w_gate.bias.copy_(b_gate)
            
            # -- VAL --
            # Pass through raw values from D0 to D5
            # This is COPY-AND-PASTE from D0 to D5
            w_val = torch.zeros(6, 6); b_val = torch.zeros(6)
            w_val[5, 0] = 1.0
            
            self.ffn.w_val.weight.copy_(w_val)
            self.ffn.w_val.bias.copy_(b_val)
            
            # --- 3. READOUT ---
            # y = 1xD0 - 2D5
            # We left all the numbers in D0, including the negative ones
            # That's why we need to subtract twice D5
            w_out = torch.zeros(1, 6)
            w_out[0, 0] = 1.0
            w_out[0, 5] = -2.0
            self.W_out.weight.copy_(w_out)

    def forward(self, x):
        # NO NORMALIZATION
        # x is used directly. This prevents float32 rounding errors.
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
        x_final = combined + ffn_out
        
        res = self.W_out(x_final.sum(dim=1))
        return res.squeeze(-1) # No Denorm needed

# Helpers

if __name__ == "__main__":
    model = GeometricTransformer()
    
    run_simple_tests(model)
    run_robust_evaluation(model)