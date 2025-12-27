import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import torch.optim as optim

from embeddings import get_tensor_embedding
from test import run_simple_tests, run_robust_evaluation

# --- 1. STRAT GLU (The Router) ---
class STRATGLU(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        
        # SPLIT BRAIN INDICES
        self.idx_ctrl = [1, 2, 3, 4] # Flags
        self.idx_data = [0, 5]       # Data (D0=In, D5=Scratch)
        
        # 1. Gate: Looks at Control -> Controls Data flow
        # We map 4 flags -> 2 gates (one for D0, one for D5)
        self.w_gate = nn.Linear(len(self.idx_ctrl), len(self.idx_data))
        
        # 2. Value: Looks at Data -> Transforms Data
        # We map 2 data -> 2 data
        self.w_val = nn.Linear(len(self.idx_data), len(self.idx_data), bias=False)
        
        self.act = nn.Sigmoid() 
        
        # --- INITIALIZATION (Mimicking Hand-Coded Router) ---
        
        # Gate: OPEN. 
        # Bias=1.0 ensures Sigmoid is active (~0.73).
        # This creates a "Gradient Highway" so the Logic can learn to shut the door.
        nn.init.constant_(self.w_gate.bias, 1.0)
        nn.init.xavier_uniform_(self.w_gate.weight, gain=0.5)
        
        # Value: IDENTITY / COPY.
        # We want the model to act like a copier initially.
        # "Copy D0 to D5".
        # We use small random weights to break symmetry, but centered on 0.
        # The optimizer will quickly find the routing path.
        nn.init.normal_(self.w_val.weight, std=0.01)

    def forward(self, x):
        # Slice inputs
        x_ctrl = x[:, :, self.idx_ctrl]
        x_data = x[:, :, self.idx_data]
        
        # Compute components
        gate = self.act(self.w_gate(x_ctrl)) 
        val = self.w_val(x_data)             
        
        # Apply Logic (Data Update)
        return gate * val

# --- 2. The STRAT Model ---
class STRATLite(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = 6
        
        # NO NORMALIZATION (Scale = 1.0)
        # We trust the Split Brain topology to handle raw magnitudes.
        self.SCALE = 1.0 
        
        self.idx_ctrl = [1, 2, 3, 4]
        self.idx_data = [0, 5]
        
        # ATTENTION: Control Only
        # Q/K/V are blind to D0/D5. 
        # This prevents the Attention layer from "smearing" the numbers.
        n_ctrl = len(self.idx_ctrl)
        self.W_q = nn.Linear(n_ctrl, n_ctrl, bias=False)
        self.W_k = nn.Linear(n_ctrl, n_ctrl, bias=False)
        self.W_v = nn.Linear(n_ctrl, n_ctrl, bias=False)
        
        self.ffn = STRATGLU(6) 
        
        # Readout: The Calculator
        self.W_out = nn.Linear(6, 1, bias=False)
        
        # Init Readout to be active
        nn.init.xavier_uniform_(self.W_out.weight)

    def forward(self, x):
        seq_len = x.shape[1]
        
        # 1. Attention (Control Stream)
        x_ctrl = x[:, :, self.idx_ctrl]
        
        Q = self.W_q(x_ctrl) 
        K = self.W_k(x_ctrl)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(len(self.idx_ctrl))
        
        # ALiBi Position Bias
        indices = torch.arange(seq_len, device=x.device)
        dist = indices.view(-1, 1) - indices.view(1, -1)
        scores -= dist.float().abs()
        
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask == 1, float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)
        
        V = self.W_v(x_ctrl)
        Z_ctrl = torch.matmul(attn_weights, V)
        
        # Residual Update (Control Only)
        x_mixed = x.clone()
        x_mixed[:, :, self.idx_ctrl] = x[:, :, self.idx_ctrl] + Z_ctrl
        
        # 2. FFN (The Router)
        # Uses Control to route Data
        data_update = self.ffn(x_mixed)
        
        # Update Data Stream
        x_final = x_mixed.clone()
        x_final[:, :, self.idx_data] = x_mixed[:, :, self.idx_data] + data_update
        
        # 3. Sum Reduce & Readout
        # Aggregates the routed data (e.g. D0 + routed D5)
        res = self.W_out(x_final.sum(dim=1))
        
        return res.squeeze(-1)

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
        x = torch.stack([get_tensor_embedding(t) for t in tokens])
        inputs.append(x)
        targets.append(float(running_sum))
        
    max_len = max([x.shape[0] for x in inputs])
    padded = torch.zeros(batch_size, max_len, 6)
    for i, x in enumerate(inputs): padded[i, :x.shape[0], :] = x
    targets = torch.tensor(targets).unsqueeze(1)
    return padded, targets

# --- EXECUTION ---
if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    
    model = STRATLite()
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    print("--- Starting Training (STRAT12) ---")
    
    for epoch in range(2001):
        x_batch, y_batch = generate_batch(batch_size=64)
        
        optimizer.zero_grad()
        output = model(x_batch) 
        
        # Standard loop with unsqueeze (Now valid because model returns [64])
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