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

# --- 2. Config L: Curriculum Learning (Standard Architecture) ---
class ConfigL(nn.Module):
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
        
        Q = self.W_q(x)
        K = self.W_k(x)
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

# Modified Generate Batch to accept max_value
def generate_batch(batch_size=32, max_val=100):
    inputs = []
    targets = []
    
    for _ in range(batch_size):
        length = random.randint(2, 5)
        ops = ['+', '-']
        tokens = []
        
        curr = random.randint(1, max_val)
        tokens.append(curr)
        running_sum = curr
        
        for _ in range(length):
            op = random.choice(ops)
            num = random.randint(1, max_val)
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

if __name__ == "__main__":
    torch.manual_seed(42)
    
    model = ConfigL()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    print("--- Starting Training (Config L: Curriculum) ---")
    
    for epoch in range(2001):
        # --- CURRICULUM SCHEDULE ---
        if epoch < 500:
            current_max = 100    # Easy (Small correlations)
        elif epoch < 1000:
            current_max = 1000   # Medium
        else:
            current_max = 5000   # Hard (Full distribution)
            
        x_batch, y_batch = generate_batch(batch_size=64, max_val=current_max)
        
        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output.unsqueeze(1), y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch} | MaxVal: {current_max} | Loss: {loss.item():.4f}")

    print("\n--- Final Verification ---")
    run_simple_tests(model)
    run_robust_evaluation(model)