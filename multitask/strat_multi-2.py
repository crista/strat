import torch
import torch.nn as nn
import torch.optim as optim
import random
import math

# --- 1. The Dual-Head Logic Router ---

class STRATGLU_MultiHead(nn.Module):
    def __init__(self, d_type_vec, d_func_vec, d_data):
        super().__init__()
        # The gate integrates two distinct logical streams
        self.d_ctrl_total = d_type_vec + d_func_vec
        
        self.w_gate = nn.Linear(self.d_ctrl_total, d_data)
        self.w_val = nn.Linear(d_data, d_data, bias=False)
        self.act = nn.Sigmoid() 
        
        # Init: Bias Open to encourage gradient flow
        nn.init.constant_(self.w_gate.bias, 1.0)
        nn.init.xavier_uniform_(self.w_gate.weight, gain=0.5)
        nn.init.normal_(self.w_val.weight, std=0.01)

    def forward(self, h_type, h_func, x_data):
        # Concatenate the two logical contexts
        # [Batch, Seq, TypeDim + FuncDim]
        ctrl_context = torch.cat([h_type, h_func], dim=-1)
        
        gate = self.act(self.w_gate(ctrl_context)) 
        val = self.w_val(x_data)             
        return gate * val

class STRAT_Multi(nn.Module):
    def __init__(self):
        super().__init__()
        
        # --- STRATA DEFINITION ---
        # D0: Value (Data)
        # D1-D4: Type Stratum (4D) -> Head 1
        # D5-D6: Function Stratum (2D) -> Head 2
        # D7: Accumulator (Data)
        
        self.d_model = 8 
        self.idx_data = [0, 7]           
        self.idx_type = [1, 2, 3, 4]     
        self.idx_func = [5, 6]           
        
        # --- EMBEDDINGS ---
        # Function: 0=Arithmetic, 1=Grocery
        self.func_embed = nn.Embedding(2, 2)
        nn.init.normal_(self.func_embed.weight, std=0.1)

        # Type: Global Vocab
        self.vocab_size = 18
        self.type_embed = nn.Embedding(self.vocab_size, 4)
        nn.init.normal_(self.type_embed.weight, std=0.02)
        
        # --- DUAL CONTROL HEADS ---
        # Head 1: The Predicate Head (Looks at Types)
        d_t = len(self.idx_type)
        self.W_q_t = nn.Linear(d_t, d_t, bias=False)
        self.W_k_t = nn.Linear(d_t, d_t, bias=False)
        self.W_v_t = nn.Linear(d_t, d_t, bias=False)

        # Head 2: The Context Head (Looks at Functions)
        # (Even though functions are constant per seq, this allows the mechanism 
        # to be theoretically consistent if context switched mid-stream)
        d_f = len(self.idx_func)
        self.W_q_f = nn.Linear(d_f, d_f, bias=False)
        self.W_k_f = nn.Linear(d_f, d_f, bias=False)
        self.W_v_f = nn.Linear(d_f, d_f, bias=False)
        
        # The GLU Router takes both streams
        self.ffn = STRATGLU_MultiHead(d_t, d_f, len(self.idx_data))
        self.W_out = nn.Linear(self.d_model, 1, bias=False)

    def forward(self, token_ids, func_ids, raw_values):
        batch_size, seq_len = token_ids.shape
        
        # 1. Assemble the Strata
        types = self.type_embed(token_ids)      # [B, S, 4]
        funcs = self.func_embed(func_ids)       # [B, S, 2]
        d0 = raw_values.unsqueeze(-1)           # [B, S, 1]
        d7 = torch.zeros_like(d0)               # [B, S, 1]
        
        # Full embedding for residual updates [Val | Type | Func | Acc]
        # Note the order change to match indices
        x = torch.cat([d0, types, funcs, d7], dim=-1)
        
        # 2. Parallel Attention Execution
        
        # --- HEAD 1: TYPE ATTENTION ---
        x_type = x[:, :, self.idx_type]
        Q_t = self.W_q_t(x_type)
        K_t = self.W_k_t(x_type)
        scores_t = torch.matmul(Q_t, K_t.transpose(-2, -1)) / math.sqrt(len(self.idx_type))
        
        # Masking (Standard Causal)
        indices = torch.arange(seq_len, device=x.device)
        dist = indices.view(-1, 1) - indices.view(1, -1)
        scores_t -= dist.float().abs()
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1)
        scores_t = scores_t.masked_fill(mask == 1, float('-inf'))
        
        attn_t = torch.softmax(scores_t, dim=-1)
        h_type = torch.matmul(attn_t, self.W_v_t(x_type)) # Result: Contextualized Types
        
        # --- HEAD 2: FUNCTION ATTENTION ---
        x_func = x[:, :, self.idx_func]
        Q_f = self.W_q_f(x_func)
        K_f = self.W_k_f(x_func)
        scores_f = torch.matmul(Q_f, K_f.transpose(-2, -1)) / math.sqrt(len(self.idx_func))
        # Masking same as above
        scores_f = scores_f.masked_fill(mask == 1, float('-inf'))
        
        attn_f = torch.softmax(scores_f, dim=-1)
        h_func = torch.matmul(attn_f, self.W_v_f(x_func)) # Result: Contextualized Function
        
        # 3. Residual Updates (Separate Strata)
        x_mixed = x.clone()
        x_mixed[:, :, self.idx_type] += h_type
        x_mixed[:, :, self.idx_func] += h_func
        
        # 4. GLU Logic Gate (The Convergence Point)
        # The gate decides based on BOTH streams
        x_data_in = x_mixed[:, :, self.idx_data]
        
        # Pass the separated heads to the gate
        # h_type contains "There is a minus sign"
        # h_func contains "We are doing Arithmetic"
        data_update = self.ffn(h_type, h_func, x_data_in)
        
        x_final = x_mixed.clone()
        x_final[:, :, self.idx_data] += data_update
        
        # 5. Readout
        res = self.W_out(x_final.sum(dim=1))
        return res.squeeze(-1)
    
# --- 2. Data Generators (Merged) ---

def get_arithmetic_batch(batch_size=32, min_val=1, max_val=100):
    # IDs: 0(Lit), 1(+), 2(-)
    # Func ID: 0
    ids, vals, y = [], [], []
    for _ in range(batch_size):
        length = random.randint(2, 5)
        t_ids, t_vals = [], []
        
        # Start with a number in the specified range
        curr = random.randint(min_val, max_val)
        t_ids.append(0); t_vals.append(float(curr)); running = curr
        
        for _ in range(length):
            op = random.choice(['+', '-'])
            num = random.randint(min_val, max_val)
            
            if op == '+': t_ids.append(1); running += num
            else: t_ids.append(2); running -= num
            
            t_vals.append(0.0)
            t_ids.append(0); t_vals.append(float(num))
        
        ids.append(torch.tensor(t_ids))
        vals.append(torch.tensor(t_vals))
        y.append(float(running))
        
    return pad_and_pack(ids, vals, y, func_id=0)

def get_grocery_batch(batch_size=32, min_val=1, max_val=20):
    # IDs: 3-17 (Offset by 3 from arithmetic)
    # Func ID: 1
    ids, vals, y = [], [], []
    for _ in range(batch_size):
        length = random.randint(3, 8)
        t_ids, t_vals = [], []
        running = 0.0
        target_cat = 0 # Items 3-7 are target
        
        for _ in range(length):
            cat = random.randint(0, 2)
            item = random.randint(0, 4)
            # Offset: 3 (Arith) + Cat*5 + Item
            token_id = 3 + (cat * 5) + item
            
            # Use specified range for costs
            val = float(random.randint(min_val, max_val))
            
            t_ids.append(token_id)
            t_vals.append(val)
            if cat == target_cat: running += val
            
        ids.append(torch.tensor(t_ids))
        vals.append(torch.tensor(t_vals))
        y.append(float(running))

    return pad_and_pack(ids, vals, y, func_id=1)

def pad_and_pack(id_list, val_list, y_list, func_id):
    batch_size = len(id_list)
    max_len = max(len(t) for t in id_list)
    p_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    p_vals = torch.zeros(batch_size, max_len, dtype=torch.float)
    p_funcs = torch.full((batch_size, max_len), func_id, dtype=torch.long)
    
    for i in range(batch_size):
        l = len(id_list[i])
        p_ids[i, :l] = id_list[i]
        p_vals[i, :l] = val_list[i]
        
    return p_ids, p_funcs, p_vals, torch.tensor(y_list).unsqueeze(1)

# --- 3. Training & Evaluation ---

def run_multitask_eval(model):
    print("\n" + "="*60)
    print("      STRAT-MULTI: CROSS-TASK EVALUATION (ID vs OOD)")
    print("="*60)
    model.eval()
    
    # --- TASK A: ARITHMETIC ---
    print("\n[Task A: Arithmetic (y = sum +/- x)]")
    
    # 1. In-Distribution (Range 1-100)
    ids_id, f_id, v_id, y_id = get_arithmetic_batch(500, min_val=1, max_val=100)
    with torch.no_grad(): pred_id = model(ids_id, f_id, v_id).unsqueeze(1)
    
    mae_id = (pred_id - y_id).abs().mean().item()
    # Logic < 2.0 (Distinguishes +x from -x)
    acc_id_logic = ((pred_id - y_id).abs() < 2.0).float().mean().item() * 100
    acc_id_strict = ((pred_id - y_id).abs() < 0.1).float().mean().item() * 100
    
    print(f"  ID  [1, 100]      | MAE: {mae_id:7.4f} | Logic Acc: {acc_id_logic:5.1f}% | Strict Acc: {acc_id_strict:5.1f}%")

    # 2. Out-of-Distribution (Range 2000-5000)
    ids_ood, f_ood, v_ood, y_ood = get_arithmetic_batch(500, min_val=2000, max_val=5000)
    with torch.no_grad(): pred_ood = model(ids_ood, f_ood, v_ood).unsqueeze(1)
    
    mae_ood = (pred_ood - y_ood).abs().mean().item()
    # Logic < 80.0 (Distinguishes +3500 from -3500)
    acc_ood_logic = ((pred_ood - y_ood).abs() < 80.0).float().mean().item() * 100
    
    print(f"  OOD [2000, 5000]  | MAE: {mae_ood:7.4f} | Logic Acc: {acc_ood_logic:5.1f}%")

    # --- TASK B: GROCERY ---
    print("\n[Task B: Grocery (y = sum targets)]")
    
    # 1. In-Distribution (Range 1-20)
    ids_id, f_id, v_id, y_id = get_grocery_batch(500, min_val=1, max_val=20)
    with torch.no_grad(): pred_id = model(ids_id, f_id, v_id).unsqueeze(1)
    
    mae_id = (pred_id - y_id).abs().mean().item()
    # Logic < 2.0 (Distinguishes "Add 10" from "Add 0")
    acc_id_logic = ((pred_id - y_id).abs() < 2.0).float().mean().item() * 100
    acc_id_strict = ((pred_id - y_id).abs() < 0.1).float().mean().item() * 100
    
    print(f"  ID  [1, 20]       | MAE: {mae_id:7.4f} | Logic Acc: {acc_id_logic:5.1f}% | Strict Acc: {acc_id_strict:5.1f}%")
    
    # 2. Out-of-Distribution (Range 500-1000)
    ids_ood, f_ood, v_ood, y_ood = get_grocery_batch(500, min_val=500, max_val=1000)
    with torch.no_grad(): pred_ood = model(ids_ood, f_ood, v_ood).unsqueeze(1)
    
    mae_ood = (pred_ood - y_ood).abs().mean().item()
    
    # Logic < 80.0. 
    # If the model failed logic, Error would be ~750.0. 
    # Error of 35.0 is definitely "Logic Success" (Leakage).
    acc_ood_logic = ((pred_ood - y_ood).abs() < 80.0).float().mean().item() * 100
    
    print(f"  OOD [500, 1000]   | MAE: {mae_ood:7.4f} | Logic Acc: {acc_ood_logic:5.1f}%")
    print("="*60 + "\n")

SEED = 1000
if __name__ == "__main__":
    torch.manual_seed(SEED)
    random.seed(SEED)
    
    model = STRAT_Multi()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    print("--- Training STRAT-Multi (Interleaved Tasks) ---")
    
    for epoch in range(3001):
        # Mixed Batch: 32 Arithmetic + 32 Grocery
        id1, f1, v1, y1 = get_arithmetic_batch(32)
        id2, f2, v2, y2 = get_grocery_batch(32)
        
        # Concat
        # Note: Pad to max length of both
        max_len = max(id1.shape[1], id2.shape[1])
        # (Padding logic simplified: assuming roughly equal lengths for demo)
        # We'll just train on them sequentially in the step to avoid complex padding code
        # Gradient accumulation handles the "Simultaneous" aspect
        
        optimizer.zero_grad()
        
        # Forward Pass 1 (Arithmetic)
        out1 = model(id1, f1, v1)
        loss1 = criterion(out1.unsqueeze(1), y1)
        loss1.backward()
        
        # Forward Pass 2 (Grocery)
        out2 = model(id2, f2, v2)
        loss2 = criterion(out2.unsqueeze(1), y2)
        loss2.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch} | Loss A: {loss1.item():.4f} | Loss G: {loss2.item():.4f}")
            
    run_multitask_eval(model)