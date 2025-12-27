import torch
import torch.nn as nn
import math
import random
import torch.optim as optim
from test_strat_learned_types import run_simple_tests, run_robust_evaluation

# --- 1. Flatland Embeddings (Unchanged) ---
class LearnableTypeEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.type_embeddings = nn.Embedding(3, 2) # 2 Dimensions
        nn.init.normal_(self.type_embeddings.weight, mean=0.0, std=0.02)

    def forward(self, token_ids, raw_values):
        types = self.type_embeddings(token_ids)
        d0 = raw_values.unsqueeze(-1)
        d_acc = torch.zeros_like(d0)
        x = torch.cat([d0, types, d_acc], dim=-1)
        return x

# --- 2. STRAT GLU (Unchanged) ---
class STRATGLU(nn.Module):
    def __init__(self, idx_ctrl, idx_data):
        super().__init__()
        self.idx_ctrl = idx_ctrl
        self.idx_data = idx_data
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

# --- 3. Multi-Head Attention (The Fix) ---
class STRAT_Flatland_MultiHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = 4
        self.embedding = LearnableTypeEmbeddings()
        
        self.idx_ctrl = [1, 2] # 2D Types
        self.idx_data = [0, 3]
        
        n_ctrl = len(self.idx_ctrl) # 2
        
        # MULTI-HEAD CONFIG
        self.num_heads = 2
        self.head_dim = 1 # 2 dims / 2 heads = 1 dim per head
        
        # We project 2D input -> 2D Q/K/V (which is 2 heads * 1 dim)
        self.W_q = nn.Linear(n_ctrl, n_ctrl, bias=False)
        self.W_k = nn.Linear(n_ctrl, n_ctrl, bias=False)
        self.W_v = nn.Linear(n_ctrl, n_ctrl, bias=False)
        
        self.ffn = STRATGLU(self.idx_ctrl, self.idx_data) 
        self.W_out = nn.Linear(4, 1, bias=False) 
        nn.init.xavier_uniform_(self.W_out.weight)

    def forward(self, token_ids, raw_values):
        x = self.embedding(token_ids, raw_values)
        batch_size, seq_len, _ = x.shape
        x_ctrl = x[:, :, self.idx_ctrl] # [B, L, 2]
        
        # 1. Project Q, K, V
        # Shape: [B, L, 2] -> [B, L, Heads(2), HeadDim(1)]
        Q = self.W_q(x_ctrl).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.W_k(x_ctrl).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.W_v(x_ctrl).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # 2. Transpose for Attention: [B, H, L, D]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # 3. Scaled Dot Product Attention
        # Scores: [B, H, L, L]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Positional Bias (Applied to both heads)
        indices = torch.arange(seq_len, device=x.device)
        dist = indices.view(-1, 1) - indices.view(1, -1)
        scores -= dist.float().abs()
        
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask == 1, float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Z: [B, H, L, D]
        Z_heads = torch.matmul(attn_weights, V)
        
        # 4. Concatenate/Flatten Heads
        # [B, H, L, D] -> [B, L, H, D] -> [B, L, 2]
        Z_ctrl = Z_heads.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        # Residual
        x_mixed = x.clone()
        x_mixed[:, :, self.idx_ctrl] = x[:, :, self.idx_ctrl] + Z_ctrl
        
        data_update = self.ffn(x_mixed)
        x_final = x_mixed.clone()
        x_final[:, :, self.idx_data] = x_mixed[:, :, self.idx_data] + data_update
        
        res = self.W_out(x_final.sum(dim=1))
        return res.squeeze(-1)

# --- Reuse Generator (N=24) ---
# (Assuming generate_batch_v2 is defined as before for N=24)
def generate_batch_v2(batch_size=24):
    if hasattr(generate_batch_v2, "fixed_batch"):
        return generate_batch_v2.fixed_batch
    # ... Same generator logic ...
    # Quick copy for safety in execution context:
    print(f"--- [GENERATOR] Creating RANDOM BATCH (N={batch_size}) ---")
    token_ids_list = []; raw_values_list = []; targets = []
    for _ in range(batch_size):
        length = random.randint(2, 5); ops = ['+', '-']; t_ids = []; vals = []
        curr = random.randint(1, 100); t_ids.append(0); vals.append(float(curr)); running_sum = curr
        for _ in range(length):
            op = random.choice(ops); num = random.randint(1, 100)
            if op == '+': t_ids.append(1); vals.append(0.0); running_sum += num
            else: t_ids.append(2); vals.append(0.0); running_sum -= num
            t_ids.append(0); vals.append(float(num))
        token_ids_list.append(torch.tensor(t_ids)); raw_values_list.append(torch.tensor(vals)); targets.append(float(running_sum))
    max_len = max([len(t) for t in token_ids_list]); padded_ids = torch.zeros(batch_size, max_len, dtype=torch.long); padded_vals = torch.zeros(batch_size, max_len, dtype=torch.float)
    for i in range(batch_size): l = len(token_ids_list[i]); padded_ids[i, :l] = token_ids_list[i]; padded_vals[i, :l] = raw_values_list[i]
    targets = torch.tensor(targets).unsqueeze(1)
    generate_batch_v2.fixed_batch = (padded_ids, padded_vals, targets)
    return padded_ids, padded_vals, targets

if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    
    model = STRAT_Flatland_MultiHead()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    print("--- Training STRAT_Flatland (2D, 2 HEADS) on N=24 ---")
    
    for epoch in range(2001):
        ids_batch, vals_batch, y_batch = generate_batch_v2(batch_size=40)
        optimizer.zero_grad()
        output = model(ids_batch, vals_batch) 
        loss = criterion(output.unsqueeze(1), y_batch) 
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if epoch % 200 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

    print("\n--- Final Verification ---")    
    run_simple_tests(model)
    run_robust_evaluation(model)