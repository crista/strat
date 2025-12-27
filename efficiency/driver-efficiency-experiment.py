import torch
import torch.nn as nn
import math
import random
import torch.optim as optim
import numpy as np
import os
import csv

# IMPORT THE HEADLESS EVALUATOR
from test_efficiency import run_robust_evaluation_headless

SEED_OFFSET = 1000 

# ==========================================
# 1. Flexible Model Definition
# ==========================================
class LearnableTypeEmbeddings(nn.Module):
    def __init__(self, type_dim):
        super().__init__()
        # Note: We use 3 embedding slots (Literal, Plus, Minus)
        self.type_embeddings = nn.Embedding(3, type_dim)
        nn.init.normal_(self.type_embeddings.weight, mean=0.0, std=0.02)

    def forward(self, token_ids, raw_values):
        types = self.type_embeddings(token_ids)
        d0 = raw_values.unsqueeze(-1)
        d_acc = torch.zeros_like(d0)
        # [Val(1) | Types(type_dim) | Acc(1)]
        x = torch.cat([d0, types, d_acc], dim=-1)
        return x

class VATSGLU(nn.Module):
    def __init__(self, d_ctrl, d_data):
        super().__init__()
        self.w_gate = nn.Linear(d_ctrl, d_data)
        self.w_val = nn.Linear(d_data, d_data, bias=False)
        self.act = nn.Sigmoid() 
        nn.init.constant_(self.w_gate.bias, 1.0)
        nn.init.xavier_uniform_(self.w_gate.weight, gain=0.5)
        nn.init.normal_(self.w_val.weight, std=0.01)

    def forward(self, x_ctrl, x_data):
        gate = self.act(self.w_gate(x_ctrl)) 
        val = self.w_val(x_data)            
        return gate * val

class VATS_Configurable(nn.Module):
    def __init__(self, type_dim, num_heads):
        super().__init__()
        self.type_dim = type_dim
        self.num_heads = num_heads
        
        # Dimensions: 1 Val + Type_Dim + 1 Acc
        self.d_model = 1 + type_dim + 1 
        
        # Indices
        self.idx_ctrl = list(range(1, 1 + type_dim)) # Middle dims are types
        self.idx_data = [0, self.d_model - 1]        # First and Last are data
        
        self.embedding = LearnableTypeEmbeddings(type_dim)
        
        # Multi-Head Config
        self.d_attn = max(type_dim, num_heads) 
        if self.d_attn % num_heads != 0:
            self.d_attn += (num_heads - (self.d_attn % num_heads))
            
        self.head_dim = self.d_attn // num_heads
        
        self.W_q = nn.Linear(type_dim, self.d_attn, bias=False)
        self.W_k = nn.Linear(type_dim, self.d_attn, bias=False)
        self.W_v = nn.Linear(type_dim, self.d_attn, bias=False)
        
        # Projection back to type_dim if needed
        self.W_mix = nn.Linear(self.d_attn, type_dim, bias=False) if self.d_attn != type_dim else nn.Identity()

        self.ffn = VATSGLU(len(self.idx_ctrl), len(self.idx_data)) 
        self.W_out = nn.Linear(self.d_model, 1, bias=False)
        nn.init.xavier_uniform_(self.W_out.weight)

    def forward(self, token_ids, raw_values):
        x = self.embedding(token_ids, raw_values)
        batch_size, seq_len, _ = x.shape
        x_ctrl = x[:, :, self.idx_ctrl]
        
        # --- Multi-Head Attention ---
        # Shape: [Batch, Seq, Heads, HeadDim]
        Q = self.W_q(x_ctrl).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x_ctrl).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x_ctrl).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Causal Masking
        indices = torch.arange(seq_len, device=x.device)
        dist = indices.view(-1, 1) - indices.view(1, -1)
        scores -= dist.float().abs()
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask == 1, float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)
        Z = torch.matmul(attn_weights, V) 
        
        # Merge Heads
        Z = Z.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_attn)
        Z_ctrl = self.W_mix(Z) 
        
        # Residual
        x_mixed = x.clone()
        x_mixed[:, :, self.idx_ctrl] = x[:, :, self.idx_ctrl] + Z_ctrl
        
        # Router
        x_data_in = x_mixed[:, :, self.idx_data]
        x_ctrl_in = x_mixed[:, :, self.idx_ctrl]
        data_update = self.ffn(x_ctrl_in, x_data_in)
        
        x_final = x_mixed.clone()
        x_final[:, :, self.idx_data] = x_data_in + data_update
        
        res = self.W_out(x_final.sum(dim=1))
        return res.squeeze(-1)

# ==========================================
# 2. Helpers
# ==========================================
def generate_random_batch(batch_size):
    # Generates Training Data
    token_ids_list = []; raw_values_list = []; targets = []

    for _ in range(batch_size):
        length = random.randint(3, 5); 
        ops = ['+', '-']; t_ids = []; vals = []
        curr = random.randint(1, 100); t_ids.append(0); vals.append(float(curr)); running_sum = curr

        for _ in range(length):
            op = random.choice(ops); num = random.randint(1, 100)
            if op == '+': t_ids.append(1); vals.append(0.0); running_sum += num
            else: t_ids.append(2); vals.append(0.0); running_sum -= num
            t_ids.append(0); vals.append(float(num))

        token_ids_list.append(torch.tensor(t_ids)); raw_values_list.append(torch.tensor(vals)); targets.append(float(running_sum))

    max_len = max([len(t) for t in token_ids_list])
    padded_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    padded_vals = torch.zeros(batch_size, max_len, dtype=torch.float)

    for i in range(batch_size):
        l = len(token_ids_list[i])
        padded_ids[i, :l] = token_ids_list[i]; padded_vals[i, :l] = raw_values_list[i]

    targets = torch.tensor(targets).unsqueeze(1)
    return padded_ids, padded_vals, targets

def train_and_eval(config, seed, epochs=2000):
    torch.manual_seed(seed)
    random.seed(seed)
    
    # 1. Create Model
    model = VATS_Configurable(config['type_dim'], config['heads'])
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # 2. Generate ONE Static Batch for this run
    ids, vals, y = generate_random_batch(config['batch_size'])
    
    # 3. Train
    for _ in range(epochs):
        optimizer.zero_grad()
        output = model(ids, vals)
        loss = criterion(output.unsqueeze(1), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
    # 4. Evaluate (CALLING THE IMPORTED FUNCTION)
    # We test on OOD range 2000-5000
    stats = run_robust_evaluation_headless(model, n_samples=1000, ood_range=(2000, 5000))
    
    return stats['ood_mae']

# ==========================================
# 3. Main Driver
# ==========================================
if __name__ == "__main__":
    N_SEEDS = 400
    n_training_data = 10
    experiments = [
        {'name': "STRAT (3D/1H)", 'type_dim': 3, 'heads': 1, 'batch_size': n_training_data},
        {'name': "STRAT (2D/2H)", 'type_dim': 2, 'heads': 2, 'batch_size': n_training_data},
        {'name': "STRAT (4D/1H)", 'type_dim': 4, 'heads': 1, 'batch_size': n_training_data},
        {'name': "STRAT (2D/1H)", 'type_dim': 2, 'heads': 1, 'batch_size': n_training_data},
    ]
    
    csv_filename = f"experiment_results_{n_training_data}.csv"
    fieldnames = ['experiment_name', 'seed', 'mae', 'success_strict', 'success_logic']

    # Initialize file with header if it doesn't exist
    if not os.path.exists(csv_filename):
        with open(csv_filename, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    # Thresholds
    strict_threshold = 10.0  # Precision Success (< 0.25% Error)
    lax_threshold = 80.0     # Logic Success (< 2.0% Error - Allows Sigmoid Tax)
    seeds = range(N_SEEDS) # 
    
    print("-" * 85)
    
    for exp in experiments:
        maes = []
        for s in seeds:
            mae = train_and_eval(exp, SEED_OFFSET + s, epochs=2000)
            maes.append(mae)
            with open(csv_filename, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow({
                    'experiment_name': exp['name'],
                    'seed': s,
                    'mae': mae,
                    'success_strict': mae < 10.0,  # Boolean for easy filtering
                    'success_logic': mae < 80.0
                })

        mean_mae = np.mean(maes)
        median_mae = np.median(maes)         
        std_mae = np.std(maes)

        # Calculate both rates
        strict_success = sum(1 for m in maes if m < strict_threshold) / len(seeds) * 100
        lax_success = sum(1 for m in maes if m < lax_threshold) / len(seeds) * 100
        
        print(f"{exp['name']:<30} | {median_mae:<10.4f} (Med) | {mean_mae:<10.1f} (Mean) | {strict_success:.0f}% (Strict) / {lax_success:.0f}% (Logic)")

