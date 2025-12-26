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

# --- 2. Standard Learned Model ---
class ConfigI(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = 6
        self.W_q = nn.Linear(6, 6, bias=False)
        self.W_k = nn.Linear(6, 6, bias=False)
        self.W_v = nn.Linear(6, 6, bias=False)
        self.ffn = SigmoidGLU(6) 
        self.W_out = nn.Linear(6, 1, bias=False)

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
        x_final = combined + ffn_out
        
        res = self.W_out(x_final.sum(dim=1))
        return res.squeeze(-1)

# --- 3. The Fixed Wrapper ---
class RescaledModel(nn.Module):
    def __init__(self, model, scale_factor):
        super().__init__()
        self.model = model
        self.scale_factor = scale_factor
        
    def forward(self, x):
        # 1. Normalize Input DOWN (Raw -> 0..1)
        # We must verify we don't modify x in place to avoid side effects
        x_norm = x.clone()
        x_norm[:, :, 0] = x_norm[:, :, 0] / self.scale_factor
        
        # 2. Forward Pass
        out = self.model(x_norm)
        
        # 3. Scale Output UP (0..1 -> Raw)
        return out * self.scale_factor
    
# --- 4. Data Generation (With Correct Input Scaling) ---
SCALE_FACTOR = 100.0

def generate_batch(batch_size=32):
    inputs = []
    targets = []
    for _ in range(batch_size):
        length = random.randint(2, 5)
        ops = ['+', '-']; tokens = []
        curr = random.randint(1, 100)
        tokens.append(curr); running_sum = curr
        
        for _ in range(length):
            op = random.choice(ops); num = random.randint(1, 100)
            tokens.append(op); tokens.append(num)
            if op == '+': running_sum += num
            else: running_sum -= num
        
        # Get embeddings
        x = torch.stack([get_tensor_embedding(t) for t in tokens])
        
        # FIX: Normalize Input D0 (Value)
        x[:, 0] = x[:, 0] / SCALE_FACTOR
        
        inputs.append(x)
        
        # Normalize Target
        targets.append(float(running_sum) / SCALE_FACTOR)
        
    max_len = max([x.shape[0] for x in inputs])
    padded_inputs = torch.zeros(batch_size, max_len, 6)
    for i, x in enumerate(inputs):
        padded_inputs[i, :x.shape[0], :] = x
    targets = torch.tensor(targets).unsqueeze(1)
    return padded_inputs, targets

# --- 5. Training Loop ---
if __name__ == "__main__":
    torch.manual_seed(42)
    model = ConfigI()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    print("--- Starting Training (Config I: Full Normalization) ---")
    
    for epoch in range(2001):
        x_batch, y_batch = generate_batch(batch_size=64)
        
        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output.unsqueeze(1), y_batch)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.6f}")

    # Wrap for Verification
    eval_model = RescaledModel(model, SCALE_FACTOR)
    eval_model.eval() 

    print("\n--- Final Verification ---")
    run_simple_tests(eval_model)
    run_robust_evaluation(eval_model)