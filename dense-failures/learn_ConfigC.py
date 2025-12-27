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

# --- 2. Hybrid Model (Frozen Attention, Learned Logic) ---
class ConfigC(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = 6
        
        # --- FROZEN LAYERS (Hand-coded later) ---
        self.W_q = nn.Linear(6, 6, bias=False)
        self.W_k = nn.Linear(6, 6, bias=False)
        self.W_v = nn.Linear(6, 6, bias=False)
        
        # --- LEARNED LAYERS (Random Init) ---
        self.ffn = SigmoidGLU(6) 
        self.W_out = nn.Linear(6, 1, bias=False)

        # Apply the frozen attention weights immediately
        self.hand_code_frozen_attention()

    def hand_code_frozen_attention(self):
        with torch.no_grad():
            # --- ATTENTION (The "Routing" Mechanism) ---
            # We enforce the logic: Numbers look for Operators.
            
            # Q: Active if IsNum (D1) -> writes to channel 4
            w_q = torch.zeros(6, 6)
            w_q[4, 1] = 10.0
            self.W_q.weight.copy_(w_q)
            
            # K: Active if IsOp (D4) -> writes to channel 4
            w_k = torch.zeros(6, 6)
            w_k[4, 4] = 10.0
            self.W_k.weight.copy_(w_k)
            
            # V: Pass the "Instruction" (IsAdd/IsSub)
            w_v = torch.zeros(6, 6)
            w_v[2, 2] = 1.0  # Pass IsAdd
            w_v[3, 3] = 1.0  # Pass IsSub
            self.W_v.weight.copy_(w_v)

        # --- FREEZE THEM ---
        self.W_q.weight.requires_grad = False
        self.W_k.weight.requires_grad = False
        self.W_v.weight.requires_grad = False

    def forward(self, x):
        seq_len = x.shape[1]
        
        # 1. Attention (FROZEN)
        Q = self.W_q(x)
        K = self.W_k(x)
        scores = torch.matmul(Q, K.transpose(-2, -1))
        
        # Fixed Geometric Prior
        indices = torch.arange(seq_len, device=x.device)
        dist = indices.view(-1, 1) - indices.view(1, -1)
        scores -= dist.float().abs()
        
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask == 1, float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)
        V = self.W_v(x)
        Z = torch.matmul(attn_weights, V)
        
        # 2. Residual + FFN (LEARNED)
        combined = x + Z
        ffn_out = self.ffn(combined)
        x_final = combined + ffn_out
        
        # 3. Readout (LEARNED)
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
    
    model = ConfigC()
    
    # Filter optimizer to only update LEARNED parameters
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
    criterion = nn.MSELoss()
    
    print("--- Starting Training (Config C: Frozen Attention, Learned Logic) ---")
    print(f"Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"Frozen Params:    {sum(p.numel() for p in model.parameters() if not p.requires_grad)}")
    
    for epoch in range(2001):
        x_batch, y_batch = generate_batch(batch_size=64)
        
        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output.unsqueeze(1), y_batch)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

    # --- Verification ---
    print("\n--- Final Verification ---")
    run_simple_tests(model)
    run_robust_evaluation(model)