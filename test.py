import torch
import random
from embeddings import get_tensor_embedding

# --- Robust Evaluation Helper ---
def generate_eval_batch(num_samples, min_val, max_val):
    """Generates a batch of data with specific value ranges."""
    inputs = []
    targets = []
    ops = ['+', '-']
    
    for _ in range(num_samples):
        length = random.randint(2, 5)
        tokens = []
        
        # Start with a number in the specified range
        curr = random.randint(min_val, max_val)
        tokens.append(curr)
        running_sum = curr
        
        for _ in range(length):
            op = random.choice(ops)
            num = random.randint(min_val, max_val)
            tokens.append(op)
            tokens.append(num)
            
            if op == '+': running_sum += num
            else: running_sum -= num
            
        x = torch.stack([get_tensor_embedding(t) for t in tokens])
        inputs.append(x)
        targets.append(float(running_sum))
        
    # Padding
    max_len = max([x.shape[0] for x in inputs])
    padded_inputs = torch.zeros(num_samples, max_len, 6)
    for i, x in enumerate(inputs):
        padded_inputs[i, :x.shape[0], :] = x
        
    targets = torch.tensor(targets).unsqueeze(1)
    return padded_inputs, targets

def run_robust_evaluation(model):
    print("\n" + "="*50)
    print("   ROBUST STATISTICAL EVALUATION (N=1000)")
    print("="*50)
    
    model.eval()
    
    # --- 1. In-Distribution Test (Range 1-100) ---
    x_in, y_in = generate_eval_batch(1000, 1, 100)
    with torch.no_grad():
        pred_in = model(x_in).unsqueeze(1)
    
    err_in = (pred_in - y_in).abs()
    mae_in = err_in.mean().item()
    max_in = err_in.max().item()
    min_in = err_in.min().item()

    # "Solved" if error is effectively zero (floating point noise)
    acc_in = (err_in < 1e-2).float().mean().item() * 100
    
    print(f"\n[In-Distribution] (Values 1-100)")
    print(f"MAE: {mae_in:.4f} | Max Err: {max_in:.4f} | Min Err: {min_in:.4f} | Solved: {acc_in:.1f}%")

    # --- 2. Out-of-Distribution Test (Range 2000-5000) ---
    x_ood, y_ood = generate_eval_batch(1000, 2000, 5000)
    with torch.no_grad():
        pred_ood = model(x_ood).unsqueeze(1)
    
    err_ood = (pred_ood - y_ood).abs()
    mae_ood = err_ood.mean().item()
    max_ood = err_ood.max().item()
    min_ood = err_ood.min().item()
    acc_ood = (err_ood < 1e-2).float().mean().item() * 100
    
    print(f"\n[Out-of-Distribution] (Values 2000-5000)")
    print(f"MAE: {mae_ood:.4f} | Max Err: {max_ood:.4f} | Min Err: {min_ood:.4f} | Solved: {acc_ood:.1f}%")
    print("="*50 + "\n")

# --- Simple tests ---
def run_simple_tests(model):
    # Test 1 (Simple)
    tokens_1 = [12, '+', 3, '-', 5]
    x_1 = torch.stack([get_tensor_embedding(t) for t in tokens_1]).unsqueeze(0)
    
    # Test 2 (Magnitude Generalization - OOD check)
    # The learned model saw nums 1-100. Let's see if it handles 2000+ like the hand-coded one.
    tokens_2 = [2000, '+', 5000, '-', 3000]
    x_2 = torch.stack([get_tensor_embedding(t) for t in tokens_2]).unsqueeze(0)
    
    model.eval()
    with torch.no_grad():
        res1 = model(x_1).item()
        res2 = model(x_2).item()
        
    print(f"Test 1 Input: 12 + 3 - 5       | Target: 10.0   | Actual: {res1:.4f}")
    print(f"Test 2 Input: 2000 + 5000 - 3000 | Target: 4000.0 | Actual: {res2:.4f}")
    
    err1 = abs(res1 - 10.0)
    err2 = abs(res2 - 4000.0)

