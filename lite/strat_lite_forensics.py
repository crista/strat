import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import torch.optim as optim
import pandas as pd

from embeddings import get_tensor_embedding
from test import run_simple_tests, run_robust_evaluation

# --- COPYING DEPENDENCIES TO MAKE IT STANDALONE ---
def get_base_embedding(token):
    x = torch.zeros(6)
    if isinstance(token, (int, float)):
        x[0] = float(token); x[1] = 1.0 
    elif token == '+':
        x[2] = 1.0; x[4] = 1.0 
    elif token == '-':
        x[3] = 1.0; x[4] = 1.0 
    return x

def generate_batch(batch_size=32):
    inputs, targets = [], []
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
            tokens.append(op); tokens.append(num)
            if op == '+': running_sum += num
            else: running_sum -= num
        x = torch.stack([get_base_embedding(t) for t in tokens])
        inputs.append(x)
        targets.append(float(running_sum))
        
    max_len = max([x.shape[0] for x in inputs])
    padded = torch.zeros(batch_size, max_len, 6)
    for i, x in enumerate(inputs): padded[i, :x.shape[0], :] = x
    targets = torch.tensor(targets).unsqueeze(1)
    return padded, targets

# --- 1. STRAT GLU (The Router) ---
class STRATGLU(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.idx_ctrl = [1, 2, 3, 4] # Flags
        self.idx_data = [0, 5]       # Data (D0=In, D5=Scratch)
        
        self.w_gate = nn.Linear(len(self.idx_ctrl), len(self.idx_data))
        self.w_val = nn.Linear(len(self.idx_data), len(self.idx_data), bias=False)
        self.act = nn.Sigmoid() 
        
        nn.init.constant_(self.w_gate.bias, 1.0)
        nn.init.xavier_uniform_(self.w_gate.weight, gain=0.5)
        nn.init.normal_(self.w_val.weight, std=0.01)

    def forward(self, x):
        x_ctrl = x[:, :, self.idx_ctrl]
        x_data = x[:, :, self.idx_data]
        gate = self.act(self.w_gate(x_ctrl)) 
        val = self.w_val(x_data)            
        return gate * val

# --- 2. The STRAT Model ---
class STRATLite(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = 6
        self.SCALE = 1.0 
        self.idx_ctrl = [1, 2, 3, 4]
        self.idx_data = [0, 5]
        
        n_ctrl = len(self.idx_ctrl)
        self.W_q = nn.Linear(n_ctrl, n_ctrl, bias=False)
        self.W_k = nn.Linear(n_ctrl, n_ctrl, bias=False)
        self.W_v = nn.Linear(n_ctrl, n_ctrl, bias=False)
        
        self.ffn = STRATGLU(6) 
        self.W_out = nn.Linear(6, 1, bias=False)
        nn.init.xavier_uniform_(self.W_out.weight)

    def forward(self, x):
        seq_len = x.shape[1]
        x_ctrl = x[:, :, self.idx_ctrl]
        
        Q = self.W_q(x_ctrl); K = self.W_k(x_ctrl)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(len(self.idx_ctrl))
        
        indices = torch.arange(seq_len, device=x.device)
        dist = indices.view(-1, 1) - indices.view(1, -1)
        scores -= dist.float().abs()
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask == 1, float('-inf'))
        attn_weights = torch.softmax(scores, dim=-1)
        
        V = self.W_v(x_ctrl)
        Z_ctrl = torch.matmul(attn_weights, V)
        x_mixed = x.clone()
        x_mixed[:, :, self.idx_ctrl] = x[:, :, self.idx_ctrl] + Z_ctrl
        
        data_update = self.ffn(x_mixed)
        x_final = x_mixed.clone()
        x_final[:, :, self.idx_data] = x_mixed[:, :, self.idx_data] + data_update
        
        res = self.W_out(x_final.sum(dim=1))
        return res.squeeze(-1)

def print_matrix(name, matrix, row_labels, col_labels):
    print(f"\n[{name}]")
    # Header
    print(f"{'':>12} " + " ".join([f"{l:>10}" for l in col_labels]))
    print("-" * (13 + 11 * len(col_labels)))
    # Rows
    for i, row_label in enumerate(row_labels):
        row_str = " ".join([f"{x:>10.4f}" for x in matrix[i]])
        print(f"{row_label:>12} {row_str}")

# --- EXECUTION ---
if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    
    model = STRATLite()
    optimizer = optim.Adam(model.parameters(), lr=0.01) # Faster LR for demo
    criterion = nn.MSELoss()
    
    print("--- Training STRAT12 for Inspection ---")
    for epoch in range(2001):
        x_batch, y_batch = generate_batch(batch_size=64)
        optimizer.zero_grad()
        output = model(x_batch) 
        loss = criterion(output.unsqueeze(1), y_batch) 
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if epoch % 500 == 0: print(f"Step {epoch}: Loss {loss.item():.4f}")

    print("\n" + "="*50)
    print("   AUTOPSY: THE LEARNED LOGIC OF STRAT12")
    print("="*50)
    
    with torch.no_grad():
# --- A. READOUT ---
        w_out = model.W_out.weight
        print_matrix("READOUT LAYER", w_out, ["Output"], ["D0(Val)", "D1(Num)", "D2(+)", "D3(-)", "D4(Op)", "D5(Scratch)"])
        
        # --- B. ATTENTION (THE BRAIN) ---
        # Note: Attention sees Control Dims [1, 2, 3, 4]
        labels = ["D1(Num)", "D2(+)", "D3(-)", "D4(Op)"]
        
        print_matrix("ATTENTION: QUERY (W_q)", model.W_q.weight, labels, labels)
        print_matrix("ATTENTION: KEY (W_k)", model.W_k.weight, labels, labels)
        print_matrix("ATTENTION: VALUE (W_v)", model.W_v.weight, labels, labels)
        
        # --- C. GLU ROUTER (THE SWITCH) ---
        # Gates see Control Dims [1, 2, 3, 4]
        # Values see Data Dims [0, 5]
        
        print_matrix("GLU: LOGIC GATES (W_gate)", model.ffn.w_gate.weight, 
                     ["Gate_D0", "Gate_D5"], labels)
        
        print(f"\n[GLU: GATE BIAS]\n     Gate_D0: {model.ffn.w_gate.bias[0]:.4f}\n     Gate_D5: {model.ffn.w_gate.bias[1]:.4f}")
        
        print_matrix("GLU: VALUE ROUTING (W_val)", model.ffn.w_val.weight, 
                     ["To_D0", "To_D5"], ["From_D0", "From_D5"])

    print("\n" + "="*60)
