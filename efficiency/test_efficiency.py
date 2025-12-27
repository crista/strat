import torch
import random

def generate_eval_batch_headless(num_samples, min_val, max_val):
    """
    Generates a batch of data for evaluation.
    Identical logic to your generate_eval_batch_v2, just helper for this file.
    """
    token_ids_list = []
    raw_values_list = []
    targets = []
    ops = ['+', '-']
    
    for _ in range(num_samples):
        length = random.randint(1, 5)
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

def run_robust_evaluation_headless(model, n_samples=1000, ood_range=(2000, 5000)):
    """
    Headless evaluator that returns metrics instead of printing them.
    """
    model.eval()
    
    # Generate OOD Test Set based on provided range
    x_ids, x_vals, y_true = generate_eval_batch_headless(n_samples, ood_range[0], ood_range[1])
    
    with torch.no_grad():
        # Pass separate streams
        y_pred = model(x_ids, x_vals).unsqueeze(1)
    
    # Calculate Metrics
    errors = (y_pred - y_true).abs()
    mae = errors.mean().item()
    max_err = errors.max().item()
    
    # "Solved" if error is < 1.0 (strictness for high magnitude numbers)
    solved_rate = (errors < 1.0).float().mean().item() * 100.0
    
    return {
        'ood_mae': mae,
        'ood_max_err': max_err,
        'ood_solved_rate': solved_rate
    }