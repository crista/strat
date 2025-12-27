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

# --- 2. The Learned Model ---
class ConfigA(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = 6
        
        # Standard Initialization will happen automatically here
        self.W_q = nn.Linear(6, 6, bias=False)
        self.W_k = nn.Linear(6, 6, bias=False)
        self.W_v = nn.Linear(6, 6, bias=False)
        self.ffn = SigmoidGLU(6) 
        self.W_out = nn.Linear(6, 1, bias=False)

    def forward(self, x):
        seq_len = x.shape[1]
        
        # 1. Attention
        Q = self.W_q(x)
        K = self.W_k(x)
        scores = torch.matmul(Q, K.transpose(-2, -1))
        
        # Fixed Geometric Prior (ALiBi-style linear decay)
        # We keep this to give the learned model a fair shot at understanding order
        indices = torch.arange(seq_len, device=x.device)
        dist = indices.view(-1, 1) - indices.view(1, -1)
        scores -= dist.float().abs()
        
        # Causal Mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask == 1, float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)
        V = self.W_v(x)
        Z = torch.matmul(attn_weights, V)
        
        # 2. Residual + FFN
        combined = x + Z
        ffn_out = self.ffn(combined)
        x_final = combined + ffn_out
        
        # 3. Sum Reduce & Readout
        # Summing over the sequence length, just like the hand-coded version
        res = self.W_out(x_final.sum(dim=1))
        return res.squeeze(-1)

def generate_batch(batch_size=32):
    inputs = []
    targets = []
    
    for _ in range(batch_size):
        # Generate sequence: N + N - N ...
        length = random.randint(2, 5) # Random number of operations
        ops = ['+', '-']
        tokens = []
        
        # Start with a number
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
            
        # Convert to tensor
        x = torch.stack([get_tensor_embedding(t) for t in tokens])
        inputs.append(x)
        targets.append(float(running_sum))
        
    # Pad sequences to max length in batch
    max_len = max([x.shape[0] for x in inputs])
    padded_inputs = torch.zeros(batch_size, max_len, 6)
    for i, x in enumerate(inputs):
        padded_inputs[i, :x.shape[0], :] = x
        
    targets = torch.tensor(targets).unsqueeze(1)
    return padded_inputs, targets

    
# --- 4. Training and testing  ---
if __name__ == "__main__":
    torch.manual_seed(42)
    
    model = ConfigA()
    optimizer = optim.Adam(model.parameters(), lr=0.01) # Standard Adam
    criterion = nn.MSELoss()
    
    print("--- Starting Training (Learned Weights) ---")
    
    for epoch in range(2001):
        x_batch, y_batch = generate_batch(batch_size=64)
        
        optimizer.zero_grad()
        output = model(x_batch)
        
        # y_batch is shape [64, 1], output is [64]
        loss = criterion(output.unsqueeze(1), y_batch)
        loss.backward()
        
        # Gradient Clipping (Optional but helpful for non-normalized transformers)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

    # --- Verification ---
    print("\n--- Final Verification ---")
    run_simple_tests(model)
    run_robust_evaluation(model)    


