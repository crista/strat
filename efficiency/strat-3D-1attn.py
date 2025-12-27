import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import torch.optim as optim

from test_strat_learned_types import run_simple_tests, run_robust_evaluation

# --- 1. Compact Embeddings (3 Dimensions) ---
class LearnableTypeEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        # REDUCED: 3 Type Dimensions
        self.type_embeddings = nn.Embedding(3, 3) 
        nn.init.normal_(self.type_embeddings.weight, mean=0.0, std=0.02)

    def forward(self, token_ids, raw_values):
        # types: [Batch, Seq, 3]
        types = self.type_embeddings(token_ids)
        
        d0 = raw_values.unsqueeze(-1)      # [Batch, Seq, 1]
        d_acc = torch.zeros_like(d0)       # [Batch, Seq, 1]
        
        # Structure: [Val(1) | Type(3) | Acc(1)] = 5 Dimensions
        x = torch.cat([d0, types, d_acc], dim=-1)
        return x

# --- 2. VATS GLU (Router) ---
class VATSGLU(nn.Module):
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

# --- 3. The Compact Model (5D) ---
class VATS_Compact(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = 5 # Reduced from 6
        self.embedding = LearnableTypeEmbeddings()
        
        # New Index Map for 5D vector:
        # 0: Val
        # 1,2,3: Types
        # 4: Acc
        self.idx_ctrl = [1, 2, 3]
        self.idx_data = [0, 4]
        
        n_ctrl = len(self.idx_ctrl)
        self.W_q = nn.Linear(n_ctrl, n_ctrl, bias=False)
        self.W_k = nn.Linear(n_ctrl, n_ctrl, bias=False)
        self.W_v = nn.Linear(n_ctrl, n_ctrl, bias=False)
        
        self.ffn = VATSGLU(self.idx_ctrl, self.idx_data) 
        
        self.W_out = nn.Linear(5, 1, bias=False) # Readout from 5D
        nn.init.xavier_uniform_(self.W_out.weight)

    def forward(self, token_ids, raw_values):
        x = self.embedding(token_ids, raw_values)
        seq_len = x.shape[1]
        x_ctrl = x[:, :, self.idx_ctrl]
        
        Q = self.W_q(x_ctrl) 
        K = self.W_k(x_ctrl)
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

# --- Standard Random Generator (N=32) ---
def generate_batch_v2(batch_size=32):
    if hasattr(generate_batch_v2, "fixed_batch"):
        return generate_batch_v2.fixed_batch

    print(f"--- [GENERATOR] Creating RANDOM BATCH (N={batch_size}) ---")
    token_ids_list = []
    raw_values_list = []
    targets = []
    
    for _ in range(batch_size):
        length = random.randint(2, 5)
        ops = ['+', '-']
        t_ids = []
        vals = []
        curr = random.randint(1, 100)
        t_ids.append(0); vals.append(float(curr))
        running_sum = curr
        
        for _ in range(length):
            op = random.choice(ops)
            num = random.randint(1, 100)
            if op == '+': 
                t_ids.append(1); vals.append(0.0); running_sum += num
            else:
                t_ids.append(2); vals.append(0.0); running_sum -= num
            t_ids.append(0); vals.append(float(num))
            
        token_ids_list.append(torch.tensor(t_ids))
        raw_values_list.append(torch.tensor(vals))
        targets.append(float(running_sum))
        
    max_len = max([len(t) for t in token_ids_list])
    padded_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    padded_vals = torch.zeros(batch_size, max_len, dtype=torch.float)
    for i in range(batch_size):
        l = len(token_ids_list[i])
        padded_ids[i, :l] = token_ids_list[i]
        padded_vals[i, :l] = raw_values_list[i]
    targets = torch.tensor(targets).unsqueeze(1)
    generate_batch_v2.fixed_batch = (padded_ids, padded_vals, targets)
    return padded_ids, padded_vals, targets


def inspect_attention(model):
    print("\n[Inspection] Attention Sharpness")
    # Create a dummy batch: "10 - 5"
    # IDs: [0(10), 2(-), 0(5)]
    ids = torch.tensor([[0, 2, 0]])
    vals = torch.tensor([[10.0, 0.0, 5.0]])
    
    with torch.no_grad():
        # We need to hook into the forward pass or just repeat the logic briefly
        # Let's just manually run the attention logic from the class:
        x = model.embedding(ids, vals)
        x_ctrl = x[:, :, model.idx_ctrl]
        Q = model.W_q(x_ctrl)
        K = model.W_k(x_ctrl)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(4)
        
        # Pos bias
        indices = torch.arange(3).view(1, -1)
        dist = indices.view(-1, 1) - indices.view(1, -1)
        scores -= dist.float().abs()
        mask = torch.triu(torch.ones(3, 3), diagonal=1)
        scores = scores.masked_fill(mask == 1, float('-inf'))
        
        attn = torch.softmax(scores, dim=-1)
        print("Attention Matrix for '10 - 5':")
        print(attn[0].numpy())


if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    
    model = VATS_Compact() # Using the 5D model
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    print("--- Training VATS_Compact (3 Type Dims) on N=32 ---")
    
    for epoch in range(2001):
        ids_batch, vals_batch, y_batch = generate_batch_v2(batch_size=24)
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

        # Inspection of Learned Embeddings
    print("\n[Inspection] Learned Type Vectors (D1..D3):")
    embeds = model.embedding.type_embeddings.weight.detach()
    print(f"Literal (ID 0): {embeds[0].numpy()}")
    print(f"Plus    (ID 1): {embeds[1].numpy()}")
    print(f"Minus   (ID 2): {embeds[2].numpy()}")
    
    print("\nIf these vectors are distinct, the model has 'discovered' the types.")
    inspect_attention(model)