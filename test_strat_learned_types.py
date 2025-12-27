import torch
import random

# --- Robust Evaluation Helper ---
def generate_eval_batch_v2(num_samples, min_val, max_val):
    """
    Generates a batch of data with specific value ranges for VATS12e.
    Returns:
        padded_ids: [Batch, Seq] (LongTensor)
        padded_vals: [Batch, Seq] (FloatTensor)
        targets: [Batch, 1] (FloatTensor)
    """
    token_ids_list = []
    raw_values_list = []
    targets = []
    ops = ['+', '-']
    
    for _ in range(num_samples):
        length = random.randint(2, 5)
        
        t_ids = []
        vals = []
        
        # Start with a number
        curr = random.randint(min_val, max_val)
        t_ids.append(0)       # Literal ID
        vals.append(float(curr))
        running_sum = curr
        
        for _ in range(length):
            op = random.choice(ops)
            num = random.randint(min_val, max_val)
            
            # Operator Token
            if op == '+': 
                t_ids.append(1) # Plus ID
                vals.append(0.0)
                running_sum += num
            else:
                t_ids.append(2) # Minus ID
                vals.append(0.0)
                running_sum -= num
                
            # Number Token
            t_ids.append(0)     # Literal ID
            vals.append(float(num))
            
        token_ids_list.append(torch.tensor(t_ids))
        raw_values_list.append(torch.tensor(vals))
        targets.append(float(running_sum))
        
    # Padding
    max_len = max([len(t) for t in token_ids_list])
    
    padded_ids = torch.zeros(num_samples, max_len, dtype=torch.long)
    padded_vals = torch.zeros(num_samples, max_len, dtype=torch.float)
    
    for i in range(num_samples):
        l = len(token_ids_list[i])
        padded_ids[i, :l] = token_ids_list[i]
        padded_vals[i, :l] = raw_values_list[i]
        
    targets = torch.tensor(targets).unsqueeze(1)
    return padded_ids, padded_vals, targets

def run_robust_evaluation(model):
    print("\n" + "="*50)
    print("   ROBUST STATISTICAL EVALUATION (N=1000)")
    print("="*50)
    
    model.eval()
    
    # --- 1. In-Distribution Test (Range 1-100) ---
    x_ids_in, x_vals_in, y_in = generate_eval_batch_v2(1000, 1, 100)
    
    with torch.no_grad():
        # Pass separate streams
        pred_in = model(x_ids_in, x_vals_in).unsqueeze(1)
    
    err_in = (pred_in - y_in).abs()
    mae_in = err_in.mean().item()
    max_in = err_in.max().item()
    
    # "Solved" if error is effectively zero (floating point noise < 0.01)
    acc_in = (err_in < 1e-2).float().mean().item() * 100
    
    print(f"\n[In-Distribution] (Values 1-100)")
    print(f"MAE: {mae_in:.4f} | Max Err: {max_in:.4f} | Solved: {acc_in:.1f}%")

    # --- 2. Out-of-Distribution Test (Range 2000-5000) ---
    x_ids_ood, x_vals_ood, y_ood = generate_eval_batch_v2(1000, 2000, 5000)
    
    with torch.no_grad():
        pred_ood = model(x_ids_ood, x_vals_ood).unsqueeze(1)
    
    err_ood = (pred_ood - y_ood).abs()
    mae_ood = err_ood.mean().item()
    max_ood = err_ood.max().item()
    
    # Note: With OOD values (e.g. 5000), error might scale slightly due to Sigmoid Tax.
    # We relax the tolerance slightly or check relative error, but sticking to absolute here.
    acc_ood = (err_ood < 1.0).float().mean().item() * 100
    
    print(f"\n[Out-of-Distribution] (Values 2000-5000)")
    print(f"MAE: {mae_ood:.4f} | Max Err: {max_ood:.4f} | Solved (<1.0 err): {acc_ood:.1f}%")
    print("="*50 + "\n")

# --- Simple tests ---
def run_simple_tests(model):
    # Helper to construct single example manually
    def make_input(tokens_raw):
        ids = []
        vals = []
        for t in tokens_raw:
            if isinstance(t, (int, float)):
                ids.append(0)
                vals.append(float(t))
            elif t == '+':
                ids.append(1)
                vals.append(0.0)
            elif t == '-':
                ids.append(2)
                vals.append(0.0)
        return torch.tensor([ids]), torch.tensor([vals])

    # Test 1 (Simple)
    t1_raw = [12, '+', 3, '-', 5]
    x_ids_1, x_vals_1 = make_input(t1_raw)
    
    # Test 2 (Magnitude Generalization - OOD check)
    t2_raw = [2000, '+', 5000, '-', 3000]
    x_ids_2, x_vals_2 = make_input(t2_raw)
    
    model.eval()
    with torch.no_grad():
        res1 = model(x_ids_1, x_vals_1).item()
        res2 = model(x_ids_2, x_vals_2).item()
        
    print(f"Test 1 Input: 12 + 3 - 5       | Target: 10.0   | Learned: {res1:.4f}")
    print(f"Test 2 Input: 2000 + 5000 - 3000 | Target: 4000.0 | Learned: {res2:.4f}")