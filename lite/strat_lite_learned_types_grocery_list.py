import torch
import torch.nn as nn
import torch.optim as optim
import random
import math

# --- 1. The Model Architecture ---

class STRATGLU(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.idx_ctrl = [1, 2, 3, 4] 
        self.idx_data = [0, 5]       
        
        self.w_gate = nn.Linear(len(self.idx_ctrl), len(self.idx_data))
        self.w_val = nn.Linear(len(self.idx_data), len(self.idx_data), bias=False)
        self.act = nn.Sigmoid() 
        
        # Init: Gate Open, Value Identity
        nn.init.constant_(self.w_gate.bias, 1.0)
        nn.init.xavier_uniform_(self.w_gate.weight, gain=0.5)
        nn.init.normal_(self.w_val.weight, std=0.01)

    def forward(self, x):
        x_ctrl = x[:, :, self.idx_ctrl]
        x_data = x[:, :, self.idx_data]
        gate = self.act(self.w_gate(x_ctrl)) 
        val = self.w_val(x_data)             
        return gate * val

class STRAT12e_Generic(nn.Module):
    def __init__(self, vocab_size=15):
        super().__init__()
        self.d_model = 6
        
        # KEY DIFFERENCE: Larger Vocabulary mapping to same 4D Type Space
        self.type_embeddings = nn.Embedding(vocab_size, 4)
        # Init with noise so no category is distinct initially
        nn.init.normal_(self.type_embeddings.weight, mean=0.0, std=0.02)
        
        self.idx_ctrl = [1, 2, 3, 4]
        self.idx_data = [0, 5]
        
        n_ctrl = len(self.idx_ctrl)
        self.W_q = nn.Linear(n_ctrl, n_ctrl, bias=False)
        self.W_k = nn.Linear(n_ctrl, n_ctrl, bias=False)
        self.W_v = nn.Linear(n_ctrl, n_ctrl, bias=False)
        
        self.ffn = STRATGLU(6)
        
        # Readout
        self.W_out = nn.Linear(6, 1, bias=False)
        nn.init.xavier_uniform_(self.W_out.weight)

    def forward(self, token_ids, raw_values):
        # 1. Embed inputs (Batch, Seq, 4)
        types = self.type_embeddings(token_ids)
        
        # 2. Construct 6D stream [Val, T1, T2, T3, T4, Acc]
        d0 = raw_values.unsqueeze(-1)
        d5 = torch.zeros_like(d0)
        x = torch.cat([d0, types, d5], dim=-1)
        
        # 3. Attention (Control Only)
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
        
        # Residual Update (Control Only)
        x_mixed = x.clone()
        x_mixed[:, :, self.idx_ctrl] = x[:, :, self.idx_ctrl] + Z_ctrl
        
        # 4. FFN Router
        data_update = self.ffn(x_mixed)
        x_final = x_mixed.clone()
        x_final[:, :, self.idx_data] = x_mixed[:, :, self.idx_data] + data_update
        
        # 5. Readout
        res = self.W_out(x_final.sum(dim=1))
        return res.squeeze(-1)

# --- 2. Data Generator ---

def generate_grocery_batch(batch_size=64, n_categories=3, items_per_cat=5, min_val=1, max_val=20):
    """
    Task: Sum only items from Category 0.
    Items 0-4 are Target. Items 5-14 are Distractors.
    """
    target_cat = 0
    token_ids_list = []
    raw_values_list = []
    targets = []
    
    for _ in range(batch_size):
        length = random.randint(3, 8)
        t_ids = []
        vals = []
        running_sum = 0.0
        
        for _ in range(length):
            cat = random.randint(0, n_categories - 1)
            item_offset = random.randint(0, items_per_cat - 1)
            token_id = (cat * items_per_cat) + item_offset
            
            value = float(random.randint(min_val, max_val))
            
            t_ids.append(token_id)
            vals.append(value)
            
            if cat == target_cat:
                running_sum += value
                
        token_ids_list.append(torch.tensor(t_ids))
        raw_values_list.append(torch.tensor(vals))
        targets.append(float(running_sum))
        
    # Padding
    max_len = max([len(t) for t in token_ids_list])
    padded_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    padded_vals = torch.zeros(batch_size, max_len, dtype=torch.float)
    
    for i in range(batch_size):
        l = len(token_ids_list[i])
        padded_ids[i, :l] = token_ids_list[i]
        padded_vals[i, :l] = raw_values_list[i]
        
    targets = torch.tensor(targets).unsqueeze(1)
    return padded_ids, padded_vals, targets

# --- 3. Verification Suite ---

def run_grocery_evaluation(model, n_cat=3, items_per_cat=5):
    print("\n" + "="*50)
    print("   GROCERY LIST VERIFICATION (N=1000)")
    print("="*50)
    
    model.eval()
    
    # --- Test 1: In-Distribution (Costs 1-20) ---
    # This checks if the logic works when costs are small
    ids_in, vals_in, y_in = generate_grocery_batch(1000, n_cat, items_per_cat, 1, 20)
    
    with torch.no_grad():
        pred_in = model(ids_in, vals_in).unsqueeze(1)
        
    err_in = (pred_in - y_in).abs()
    mae_in = err_in.mean().item()
    
    # Solved if error is negligible
    acc_in = (err_in < 0.1).float().mean().item() * 100
    
    print(f"\n[In-Distribution] (Costs $1-$20)")
    print(f"MAE: {mae_in:.4f} | Solved (<0.1): {acc_in:.1f}%")
    
    # --- Test 2: Out-of-Distribution (Costs 500-1000) ---
    # This checks if massive values break the clustering (Interference Check)
    ids_ood, vals_ood, y_ood = generate_grocery_batch(1000, n_cat, items_per_cat, 500, 1000)
    
    with torch.no_grad():
        pred_ood = model(ids_ood, vals_ood).unsqueeze(1)
        
    err_ood = (pred_ood - y_ood).abs()
    mae_ood = err_ood.mean().item()
    
    # We relax tolerance for OOD due to sigmoid scaling, but check consistency
    acc_ood = (err_ood < 5.0).float().mean().item() * 100
    
    print(f"\n[Out-of-Distribution] (Costs $500-$1000)")
    print(f"MAE: {mae_ood:.4f} | Solved (<5.0): {acc_ood:.1f}%")
    
    # --- Test 3: Distractor Isolation ---
    # Feed ONLY distractors. Sum should be 0.
    # We manually construct a batch of only Category 1 & 2 items
    print("\n[Distractor Test] Feeding only non-target items...")
    
    distractor_ids = []
    distractor_vals = []
    for _ in range(100):
        # Cat 1 starts at index 5
        token_id = random.randint(5, 14) 
        distractor_ids.append(token_id)
        distractor_vals.append(100.0) # High value to ensure leakage is visible
        
    x_dist_ids = torch.tensor(distractor_ids).unsqueeze(0) # Batch 1, Seq 100
    x_dist_vals = torch.tensor(distractor_vals).unsqueeze(0)
    
    with torch.no_grad():
        res = model(x_dist_ids, x_dist_vals).item()
        
    print(f"Input: 100 Distractor Items (Cost 100 each). Total Cost: 10,000")
    print(f"Predicted Sum: {res:.4f} (Target: 0.0)")

import numpy as np

def analyze_latent_space(model):
    print("\n" + "="*60)
    print("      LATENT SPACE FORENSIC ANALYSIS (The 'Null Space')")
    print("="*60)
    
    # Extract the learned embedding matrix (Vocab Size x 4)
    # Shape: (15, 4)
    embeds = model.type_embeddings.weight.detach().numpy()
    
    categories = {
        0: "Target (Items 0-4)",
        1: "Distractor A (Items 5-9)",
        2: "Distractor B (Items 10-14)"
    }
    
    centroids = {}
    
    # --- 1. Calculate Centroids & Intra-Cluster Variance ---
    print(f"{'Category':<25} | {'Centroid Coords (D1..D4)':<35} | {'Intra-Cluster Var (Tightness)'}")
    print("-" * 95)
    
    for cat_idx in range(3):
        start = cat_idx * 5
        end = start + 5
        
        # Get all vectors for this category
        vectors = embeds[start:end] 
        
        # Calculate Centroid (Mean vector)
        centroid = np.mean(vectors, axis=0)
        centroids[cat_idx] = centroid
        
        # Calculate Variance: Average Euclidean distance of items from their own centroid
        # This is the "10^-4" number you are looking for.
        # It measures if the model treats them as synonyms.
        distances = np.linalg.norm(vectors - centroid, axis=1)
        avg_distance = np.mean(distances)
        
        # Formatting coords for display
        coords_str = f"[{centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f}, {centroid[3]:.2f}]"
        
        print(f"{categories[cat_idx]:<25} | {coords_str:<35} | {avg_distance:.6f}")

    # --- 2. Calculate Inter-Cluster Separation ---
    print("\n" + "-"*60)
    print(" INTER-CLUSTER SEPARATION (Distance between Centroids)")
    print("-" * 60)
    
    c0 = centroids[0] # Target
    c1 = centroids[1] # Distractor A
    c2 = centroids[2] # Distractor B
    
    d_target_distA = np.linalg.norm(c0 - c1)
    d_target_distB = np.linalg.norm(c0 - c2)
    d_distA_distB  = np.linalg.norm(c1 - c2)
    
    print(f"Target       <-> Distractor A : {d_target_distA:.4f}  (Should be LARGE, e.g. > 1.0)")
    print(f"Target       <-> Distractor B : {d_target_distB:.4f}  (Should be LARGE)")
    print(f"Distractor A <-> Distractor B : {d_distA_distB:.4f}   (Should be SMALL if collapsed)")

    # --- 3. The "Collapse" Verification ---
    # We check if Distractor A and Distractor B have merged into a single 'Null Space'
    # relative to how far they are from the Target.
    
    avg_separation = (d_target_distA + d_target_distB) / 2
    collapse_ratio = d_distA_distB / avg_separation
    
    print("\n[FORENSIC CONCLUSION]")
    if collapse_ratio < 0.1:
        print(f"SUCCESS: Distractors collapsed! (Ratio: {collapse_ratio:.4f})")
        print("The model successfully invented a single 'Ignore' type for all distractors.")
    else:
        print(f"FAILURE: Distractors are distinct. (Ratio: {collapse_ratio:.4f})")
        print("The model is still treating distinct distractors as different types.")

    
# --- 4. Main Execution Block ---
SEED = 1009
if __name__ == "__main__":
    torch.manual_seed(SEED)
    random.seed(SEED)
    
    vocab_size = 15 # 3 Categories * 5 Items
    model = STRAT12e_Generic(vocab_size=vocab_size)
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    print(f"--- Training Grocery List Model (Vocab={vocab_size}) ---")
    
    for epoch in range(2001):
        ids, vals, y = generate_grocery_batch(64, n_categories=3, items_per_cat=5)
        
        optimizer.zero_grad()
        output = model(ids, vals)
        
        # Unsqueeze output to match target shape [64, 1]
        loss = criterion(output.unsqueeze(1), y)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
            
    # Run the Verification Suite
    run_grocery_evaluation(model)
    analyze_latent_space(model)