import torch
import torch.nn as nn
import torch.optim as optim
import random

from embeddings import get_tensor_embedding
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

# --- 2. Compensated Hybrid Model ---
class ConfigD(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = 6
        
        # --- LEARNED LAYERS ---
        self.W_q = nn.Linear(6, 6, bias=False)
        self.W_k = nn.Linear(6, 6, bias=False)
        self.W_v = nn.Linear(6, 6, bias=False)
        
        # --- THE COMPENSATION: Learnable Pre-Amp ---
        # We initialize this to 5.0. 
        # Why? If attention is diffuse (1/5 = 0.2), we need 0.2 * Gain >= 1.0 roughly
        # to ensure the signal magnitude matches the hard logic threshold.
        self.attn_gain = nn.Parameter(torch.tensor(5.0))
        
        # --- FROZEN LAYERS ---
        self.ffn = SigmoidGLU(6) 
        self.W_out = nn.Linear(6, 1, bias=False)

        self.hand_code_frozen_logic()

    def hand_code_frozen_logic(self):
        with torch.no_grad():
            # --- FFN LOGIC (High Thresholds) ---
            w_gate = torch.zeros(6, 6); b_gate = torch.zeros(6)
            GAIN = 100.0
            
            # Target: IsNum(D1) + IsSub(D3)
            w_gate[5, 1] = GAIN
            w_gate[5, 3] = GAIN
            # Stiff Bias: -150.0
            b_gate[5] = -1.5 * GAIN
            
            self.ffn.w_gate.weight.copy_(w_gate)
            self.ffn.w_gate.bias.copy_(b_gate)
            
            # Val: Pass D0 -> D5
            w_val = torch.zeros(6, 6); b_val = torch.zeros(6)
            w_val[5, 0] = 1.0
            self.ffn.w_val.weight.copy_(w_val)
            self.ffn.w_val.bias.copy_(b_val)
            
            # --- READOUT ---
            w_out = torch.zeros(1, 6)
            w_out[0, 0] = 1.0
            w_out[0, 5] = -2.0
            self.W_out.weight.copy_(w_out)
            
        # Freeze Logic
        for p in self.ffn.parameters(): p.requires_grad = False
        self.W_out.weight.requires_grad = False

    def forward(self, x):
        seq_len = x.shape[1]
        
        # 1. Attention (Learned)
        Q = self.W_q(x)
        K = self.W_k(x)
        scores = torch.matmul(Q, K.transpose(-2, -1))
        
        # Geometric Prior
        indices = torch.arange(seq_len, device=x.device)
        dist = indices.view(-1, 1) - indices.view(1, -1)
        scores -= dist.float().abs()
        
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask == 1, float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)
        V = self.W_v(x)
        Z = torch.matmul(attn_weights, V)
        
        # --- APPLY COMPENSATION ---
        # Boost the signal before it hits the stiff logic
        Z = Z * self.attn_gain
        
        # 2. Residual + FFN (Frozen)
        combined = x + Z
        ffn_out = self.ffn(combined)
        x_final = combined + ffn_out
        
        # 3. Readout (Frozen)
        res = self.W_out(x_final.sum(dim=1))
        return res.squeeze(-1)

def generate_batch(batch_size=32):
    inputs = []
    targets = []
    for _ in range(batch_size):
        length = random.randint(2, 5)
        ops = ['+', '-']
        tokens = []
        curr = random.randint(1, 100)
        tokens.append(curr)
        running_sum = curr
        for _ in range(length):
            op = random.choice(ops)
            num = random.randint(1, 100)
            tokens.append(op)
            tokens.append(num)
            if op == '+': running_sum += num
            else: running_sum -= num
        x = torch.stack([get_tensor_embedding(t) for t in tokens])
        inputs.append(x)
        targets.append(float(running_sum))
        
    max_len = max([x.shape[0] for x in inputs])
    padded_inputs = torch.zeros(batch_size, max_len, 6)
    for i, x in enumerate(inputs):
        padded_inputs[i, :x.shape[0], :] = x
    targets = torch.tensor(targets).unsqueeze(1)
    return padded_inputs, targets

# --- 4. Training Loop ---
if __name__ == "__main__":
    torch.manual_seed(42)
    
    model = ConfigD()
    
    # Optimizer includes attn_gain
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
    criterion = nn.MSELoss()
    
    print("--- Starting Training (Config D: Compensated Hybrid) ---")
    print(f"Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"Initial Gain:     {model.attn_gain.item():.2f}")
    
    for epoch in range(2001):
        x_batch, y_batch = generate_batch(batch_size=64)
        
        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output.unsqueeze(1), y_batch)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f} | Gain: {model.attn_gain.item():.4f}")

    # --- Verification ---
    print("\n--- Final Verification ---")
    run_simple_tests(model)
    run_robust_evaluation(model)