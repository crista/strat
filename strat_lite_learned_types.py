import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import torch.optim as optim

from test_e import run_simple_tests, run_robust_evaluation

# --- 1. Learnable Embeddings (The "Meta-Learning" Component) ---
class LearnableTypeEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        # 3 Token Types: Literal (0), Plus (1), Minus (2)
        # 4 Type Dimensions: D1, D2, D3, D4
        self.type_embeddings = nn.Embedding(3, 4)
        
        # Initialize with small random noise to break symmetry
        # We start "blind" - the model doesn't know D2 is "Plus" yet.
        nn.init.normal_(self.type_embeddings.weight, mean=0.0, std=0.02)

    def forward(self, token_ids, raw_values):
        """
        token_ids: [Batch, Seq] (Integers 0, 1, 2)
        raw_values: [Batch, Seq] (Floats e.g. 5.0, 0.0, 12.0)
        """
        batch_size, seq_len = token_ids.shape
        
        # 1. Get the learned Type Vector (Batch, Seq, 4)
        # This corresponds to D1..D4
        types = self.type_embeddings(token_ids)
        
        # 2. Construct the full 6D residual stream
        # Structure: [D0, D1, D2, D3, D4, D5]
        
        # D0: The Raw Value (Payload)
        d0 = raw_values.unsqueeze(-1)  # [Batch, Seq, 1]
        
        # D5: The Accumulator (Initialized to 0)
        d5 = torch.zeros_like(d0)      # [Batch, Seq, 1]
        
        # Concatenate: [Val, Type1, Type2, Type3, Type4, Acc]
        x = torch.cat([d0, types, d5], dim=-1)
        
        return x

# --- 2. VATS GLU (The Router) ---
# Unchanged from VATS12, but now routing learned latent types
class VATSGLU(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        
        # SPLIT BRAIN INDICES
        self.idx_ctrl = [1, 2, 3, 4] # Learned Latent Flags
        self.idx_data = [0, 5]       # Data (D0=In, D5=Scratch)
        
        # 1. Gate: Looks at Control -> Controls Data flow
        self.w_gate = nn.Linear(len(self.idx_ctrl), len(self.idx_data))
        
        # 2. Value: Looks at Data -> Transforms Data
        self.w_val = nn.Linear(len(self.idx_data), len(self.idx_data), bias=False)
        
        self.act = nn.Sigmoid() 
        
        # --- INITIALIZATION ---
        # Same strategy: Gate Open, Value Identity
        nn.init.constant_(self.w_gate.bias, 1.0)
        nn.init.xavier_uniform_(self.w_gate.weight, gain=0.5)
        nn.init.normal_(self.w_val.weight, std=0.01)

    def forward(self, x):
        x_ctrl = x[:, :, self.idx_ctrl]
        x_data = x[:, :, self.idx_data]
        
        gate = self.act(self.w_gate(x_ctrl)) 
        val = self.w_val(x_data)             
        
        return gate * val

# --- 3. The VATS-12e Model (Learned Types) ---
class VATS12e(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = 6
        
        # The Embedding Layer (New!)
        self.embedding = LearnableTypeEmbeddings()
        
        self.idx_ctrl = [1, 2, 3, 4]
        self.idx_data = [0, 5]
        
        # ATTENTION: Control Only
        n_ctrl = len(self.idx_ctrl)
        self.W_q = nn.Linear(n_ctrl, n_ctrl, bias=False)
        self.W_k = nn.Linear(n_ctrl, n_ctrl, bias=False)
        self.W_v = nn.Linear(n_ctrl, n_ctrl, bias=False)
        
        self.ffn = VATSGLU(6) 
        
        # Readout
        self.W_out = nn.Linear(6, 1, bias=False)
        nn.init.xavier_uniform_(self.W_out.weight)

    def forward(self, token_ids, raw_values):
        # 1. Embed inputs using learned types
        x = self.embedding(token_ids, raw_values)
        seq_len = x.shape[1]
        
        # 2. Attention (Control Stream)
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
        
        # 3. FFN (The Router)
        data_update = self.ffn(x_mixed)
        
        # Update Data Stream
        x_final = x_mixed.clone()
        x_final[:, :, self.idx_data] = x_mixed[:, :, self.idx_data] + data_update
        
        # 4. Sum Reduce & Readout
        res = self.W_out(x_final.sum(dim=1))
        
        return res.squeeze(-1)

# --- HELPERS ---

def generate_batch_v2(batch_size=32):
    """
    Returns separate tensors for Token IDs and Raw Values.
    IDs: 0=Num, 1=Plus, 2=Minus
    """
    token_ids_list = []
    raw_values_list = []
    targets = []
    
    for _ in range(batch_size):
        length = random.randint(2, 5)
        ops = ['+', '-']
        
        t_ids = []
        vals = []
        
        # First Number
        curr = random.randint(1, 100)
        t_ids.append(0)       # Literal ID
        vals.append(float(curr))
        running_sum = curr
        
        for _ in range(length):
            op = random.choice(ops)
            num = random.randint(1, 100)
            
            # Operator
            if op == '+': 
                t_ids.append(1) # Plus ID
                vals.append(0.0)
                running_sum += num
            else:
                t_ids.append(2) # Minus ID
                vals.append(0.0)
                running_sum -= num
                
            # Number
            t_ids.append(0)     # Literal ID
            vals.append(float(num))
            
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

def generate_batch_fixed(batch_size=32): 
    """
    Returns A FIXED SINGLE BATCH.
    The first time it is called, it generates data and caches it.
    Every subsequent call returns the exact same tensors.
    """
    # Check if we have already generated the batch
    if hasattr(generate_batch_v2, "fixed_batch"):
        return generate_batch_v2.fixed_batch

    print(f"--- [GENERATOR] Creating the ONE TRUE BATCH (Size: {batch_size}) ---")
    
    token_ids_list = []
    raw_values_list = []
    targets = []
    
    for _ in range(batch_size):
        length = random.randint(2, 5)
        ops = ['+', '-']
        
        t_ids = []
        vals = []
        
        # First Number
        curr = random.randint(1, 100)
        t_ids.append(0)     # Literal ID
        vals.append(float(curr))
        running_sum = curr
        
        for _ in range(length):
            op = random.choice(ops)
            num = random.randint(1, 100)
            
            # Operator
            if op == '+': 
                t_ids.append(1) # Plus ID
                vals.append(0.0)
                running_sum += num
            else:
                t_ids.append(2) # Minus ID
                vals.append(0.0)
                running_sum -= num
                
            # Number
            t_ids.append(0)     # Literal ID
            vals.append(float(num))
            
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
    
    # Cache the batch as a function attribute
    generate_batch_v2.fixed_batch = (padded_ids, padded_vals, targets)
    
    return padded_ids, padded_vals, targets

def generate_orthogonal_batch(batch_size=32):
    """
    Generates a DETERMINISTIC dataset that covers ALL combinatorial
    operator patterns for lengths 2, 3, 4, and 5.
    
    Total patterns: 2 + 4 + 8 + 16 = 30 patterns.
    We pad with 2 random examples to reach N=32.
    """
    if hasattr(generate_orthogonal_batch, "fixed_batch"):
        return generate_orthogonal_batch.fixed_batch

    print(f"--- [GENERATOR] Creating ORTHOGONAL BASIS (N={batch_size}) ---")
    
    token_ids_list = []
    raw_values_list = []
    targets = []
    
    # Generate all binary patterns for varying numbers of operators
    # 0 = Plus (ID 1), 1 = Minus (ID 2)
    patterns = []
    
    # 1 Op (Length 2 sequence of numbers)
    for i in range(2): patterns.append(format(i, '01b'))
    # 2 Ops (Length 3 sequence of numbers)
    for i in range(4): patterns.append(format(i, '02b'))
    # 3 Ops (Length 4 sequence of numbers)
    for i in range(8): patterns.append(format(i, '03b'))
    # 4 Ops (Length 5 sequence of numbers)
    for i in range(16): patterns.append(format(i, '04b'))
    
    # We now have 30 patterns. 
    # If batch_size > 30, we'll just repeat the last few randomly.
    
    for p_idx, pattern_str in enumerate(patterns):
        if p_idx >= batch_size: break
        
        t_ids = []
        vals = []
        
        # Start with a number
        # We use consistent magnitudes to avoid distracting the model
        curr = 50.0 
        t_ids.append(0)
        vals.append(curr)
        running_sum = curr
        
        # Apply the pattern
        for char in pattern_str:
            op_val = random.randint(1, 20) # Small variability
            
            if char == '0': # Plus
                t_ids.append(1)
                vals.append(0.0)
                running_sum += op_val
            else: # Minus
                t_ids.append(2)
                vals.append(0.0)
                running_sum -= op_val
            
            # The number following the op
            t_ids.append(0)
            vals.append(float(op_val))
            
        token_ids_list.append(torch.tensor(t_ids))
        raw_values_list.append(torch.tensor(vals))
        targets.append(float(running_sum))
        
    # Fill remaining spots (if batch_size > 30) with random valid patterns
    while len(token_ids_list) < batch_size:
        # Just copy the first one (simple case) to fill the tensor
        token_ids_list.append(token_ids_list[0].clone())
        raw_values_list.append(raw_values_list[0].clone())
        targets.append(targets[0])

    # Padding
    max_len = max([len(t) for t in token_ids_list])
    padded_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    padded_vals = torch.zeros(batch_size, max_len, dtype=torch.float)
    
    for i in range(batch_size):
        l = len(token_ids_list[i])
        padded_ids[i, :l] = token_ids_list[i]
        padded_vals[i, :l] = raw_values_list[i]
        
    targets = torch.tensor(targets).unsqueeze(1)
    
    # Cache
    generate_orthogonal_batch.fixed_batch = (padded_ids, padded_vals, targets)
    
    return padded_ids, padded_vals, targets

def generate_balanced_batch(batch_size=32):
    """
    Generates a STRATEGICALLY BALANCED dataset.
    Instead of covering all complex 5-step paths, we focus on 
    grounding the 'Atoms' (Length 2) and 'Interactions' (Length 3).
    
    Structure (Total 32):
    - 8x Length 2 (Pure Addition/Subtraction) -> The Anchor
    - 8x Length 3 (Simple Chaining)
    - 8x Length 4 (Complex Chaining)
    - 8x Length 5 (Deep Chaining)
    """
    if hasattr(generate_balanced_batch, "fixed_batch"):
        return generate_balanced_batch.fixed_batch

    print(f"--- [GENERATOR] Creating BALANCED ATOMIC BATCH (N={batch_size}) ---")
    
    token_ids_list = []
    raw_values_list = []
    targets = []
    
    # We will generate 8 examples for each length from 2 to 5
    lengths = [2, 3, 4, 5]
    samples_per_len = batch_size // len(lengths) # 32 // 4 = 8
    
    for length in lengths:
        for _ in range(samples_per_len):
            t_ids = []
            vals = []
            
            # Start Number
            curr = float(random.randint(1, 100))
            t_ids.append(0)
            vals.append(curr)
            running_sum = curr
            
            # Generate Ops
            # We want to ensure we see both + and - frequently
            for _ in range(length - 1): # Length 2 has 1 op
                op_type = random.choice(['+', '-'])
                num = float(random.randint(1, 100))
                
                if op_type == '+':
                    t_ids.append(1) # Plus ID
                    vals.append(0.0)
                    running_sum += num
                else:
                    t_ids.append(2) # Minus ID
                    vals.append(0.0)
                    running_sum -= num
                
                # The Number
                t_ids.append(0)
                vals.append(num)
            
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
    
    # Cache
    generate_balanced_batch.fixed_batch = (padded_ids, padded_vals, targets)
    
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

SEED = 42
# --- EXECUTION ---
if __name__ == "__main__":
    torch.manual_seed(SEED)
    random.seed(SEED)
    
    # Instantiate the learned embeddings model
    model = VATS12e()
    
    # We use a slightly higher LR for the embeddings to help them separate quickly
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    print("--- Starting Training (VATS12e - Learned Types) ---")
    print("Initializing with random Type Embeddings...")
    
    for epoch in range(2001):
        # Get separated inputs
        ids_batch, vals_batch, y_batch = generate_batch_v2(batch_size=64)
        
        optimizer.zero_grad()
        
        # Pass both IDs and Values
        output = model(ids_batch, vals_batch) 
        
        loss = criterion(output.unsqueeze(1), y_batch) 
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

    print("\n--- Training Complete ---")
    
        # --- Verification ---
    print("\n--- Final Verification ---")    
    run_simple_tests(model)
    run_robust_evaluation(model)

    # Inspection of Learned Embeddings
    print("\n[Inspection] Learned Type Vectors (D1..D4):")
    embeds = model.embedding.type_embeddings.weight.detach()
    print(f"Literal (ID 0): {embeds[0].numpy()}")
    print(f"Plus    (ID 1): {embeds[1].numpy()}")
    print(f"Minus   (ID 2): {embeds[2].numpy()}")
    
    print("\nIf these vectors are distinct, the model has 'discovered' the types.")
    inspect_attention(model)