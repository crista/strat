import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import random
import time

# --- IMPORTS ---
# Ensure these files exist in your directory
from learn_ConfigA import ConfigA
from learn_ConfigI import ConfigI
from learn_ConfigB import ConfigB
from learn_ConfigC import ConfigC
from learn_ConfigD import ConfigD
from learn_ConfigE import ConfigE
from learn_ConfigF import ConfigF
from learn_ConfigG import ConfigG
from learn_ConfigH import ConfigH
from learn_ConfigJ import ConfigJNorm
from learn_ConfigK import ConfigK
from learn_ConfigL import ConfigL
from strat_lite import STRATLite as ConfigJ

# --- GLOBAL SETTINGS ---
NUM_RUNS = 50
TOTAL_STEPS = 2000 
BATCH_SIZE = 64
SEED_OFFSET = 1000 

# --- DATA GENERATION HELPERS ---
def get_tensor_embedding(token):
    x = torch.zeros(6)
    if isinstance(token, (int, float)):
        x[0] = float(token); x[1] = 1.0
    elif token == '+':
        x[2] = 1.0; x[4] = 1.0
    elif token == '-':
        x[3] = 1.0; x[4] = 1.0
    return x

def generate_batch(batch_size=32, scale_target=1.0):
    inputs = []
    targets = []
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
            tokens.append(op)
            tokens.append(num)
            if op == '+': running_sum += num
            else: running_sum -= num
        
        x = torch.stack([get_tensor_embedding(t) for t in tokens])
        inputs.append(x)
        targets.append(float(running_sum) * scale_target)
        
    max_len = max([x.shape[0] for x in inputs])
    padded_inputs = torch.zeros(batch_size, max_len, 6)
    for i, x in enumerate(inputs):
        padded_inputs[i, :x.shape[0], :] = x
    targets = torch.tensor(targets).unsqueeze(1)
    return padded_inputs, targets

# --- EVALUATION HELPER ---
def evaluate(model, min_val, max_val, scale_output=1.0):
    model.eval()
    
    inputs = []
    targets = []
    for _ in range(1000):
        length = random.randint(2, 5)
        curr = random.randint(min_val, max_val)
        tokens = [curr]
        running_sum = curr
        for _ in range(length):
            op = random.choice(['+', '-'])
            num = random.randint(min_val, max_val)
            tokens.append(op); tokens.append(num)
            if op == '+': running_sum += num
            else: running_sum -= num
        x = torch.stack([get_tensor_embedding(t) for t in tokens])
        inputs.append(x)
        targets.append(running_sum)
        
    max_len = max([x.shape[0] for x in inputs])
    padded = torch.zeros(1000, max_len, 6)
    for i, x in enumerate(inputs):
        padded[i, :x.shape[0], :] = x
        
    with torch.no_grad():
        preds = model(padded).squeeze() * scale_output
        
    targets = torch.tensor(targets).float()
    errors = (preds - targets).abs()
    mae = errors.mean().item()
    solved = (errors < 1e-2).float().mean().item() * 100
    return mae, solved

# --- STATS HELPER ---
def calculate_mad(tensor):
    if tensor.numel() == 0: return 0.0
    mean_val = torch.mean(tensor)
    mad = torch.mean(torch.abs(tensor - mean_val))
    return mad.item()

# --- MAIN EXPERIMENT RUNNER ---
def run_single_experiment(config_name, model_class, seed, run_id):
    # 1. Setup
    torch.manual_seed(seed)
    random.seed(seed)
    model = model_class()
    
    params = filter(lambda p: p.requires_grad, model.parameters())
    # Note: VATS12 uses LR 0.005, others use 0.01. 
    # We can handle this logic here or just use 0.005 for all (others should handle it fine)
    # or check the class name.
    lr = 0.005 if config_name == 'J' else 0.01
    optimizer = optim.Adam(params, lr=lr)
    criterion = nn.MSELoss()
    
    use_ortho = config_name in ['G', 'H']
    scale_target = 0.01 if config_name == 'I' else 1.0
    scale_output = 100.0 if config_name == 'I' else 1.0
    lambda_reg = 10.0
    
    start_time = time.time()
    
    all_input_nums = []
    all_target_vals = []
    
    final_train_loss = 0.0
    
    # 2. Training Loop
    for step in range(TOTAL_STEPS + 1):
        model.train()
        x_batch, y_batch = generate_batch(BATCH_SIZE, scale_target=scale_target)
        
        # Collect Stats
        with torch.no_grad():
            mask = x_batch[:, :, 1] == 1.0
            flat_inputs = x_batch[:, :, 0][mask].cpu()
            flat_targets = (y_batch / scale_target).squeeze().cpu()
            
            all_input_nums.append(flat_inputs)
            all_target_vals.append(flat_targets)
        
        optimizer.zero_grad()
        output = model(x_batch)
        
        # Ensure output is [Batch, 1] for loss
        if output.dim() == 1:
            output = output.unsqueeze(1)
            
        mse_loss = criterion(output, y_batch)
        
        if use_ortho:
            gate_weights = model.ffn.w_gate.weight
            ortho_loss = (gate_weights[:, 0] ** 2).sum()
            loss = mse_loss + (lambda_reg * ortho_loss)
        else:
            loss = mse_loss
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step == TOTAL_STEPS:
            final_train_loss = mse_loss.item()
            
    # Compute MAD
    full_inputs = torch.cat(all_input_nums)
    full_targets = torch.cat(all_target_vals)
    in_mad = calculate_mad(full_inputs)
    out_mad = calculate_mad(full_targets)
            
    # 3. Evaluation
    id_mae, id_solved = evaluate(model, 1, 100, scale_output)
    ood_mae, ood_solved = evaluate(model, 2000, 5000, scale_output)
    
    duration = time.time() - start_time
    
    return {
        'Config': config_name,
        'Run_ID': run_id,
        'Seed': seed,
        'Train_MSE': final_train_loss,
        'Input_MAD': in_mad,
        'Target_MAD': out_mad,
        'ID_MAE': id_mae,
        'ID_Solved_Pct': id_solved,
        'OOD_MAE': ood_mae,
        'OOD_Solved_Pct': ood_solved,
        'Duration_Sec': duration
    }

# --- EXECUTION ---
if __name__ == "__main__":
    # FULL CONFIGURATION LIST
    configs_to_run = {
        'A': ConfigA, 
        'I': ConfigI,
        'B': ConfigB,
        'C': ConfigC,
        'D': ConfigD,
        'E': ConfigE,
        'F': ConfigF,
        'G': ConfigG,
        'H': ConfigH,
        'J': ConfigJNorm,
        'K': ConfigK,
        'L': ConfigL,
        'S': ConfigJ # The STRATLite
    }
    
    all_results = []
    
    print(f"Starting Campaign: {NUM_RUNS} runs per config...")
    print("-" * 110)
    print(f"{'Config':<10} | {'Run':<5} | {'In MAD':<10} | {'Tgt MAD':<10} | {'ID MAE':<10} | {'OOD MAE':<10} | {'Status'}")
    print("-" * 110)
    
    for i in range(NUM_RUNS):
        current_seed = SEED_OFFSET + i 

        for name, cls in configs_to_run.items():
            result = run_single_experiment(name, cls, current_seed, i+1)
            all_results.append(result)
            
            print(f"{name:<10} | {i+1:<5} | {result['Input_MAD']:<10.2f} | {result['Target_MAD']:<10.2f} | {result['ID_MAE']:<10.4f} | {result['OOD_MAE']:<10.4f} | Done")
            
    df = pd.DataFrame(all_results)
    df.to_csv("experiment_results_50x.csv", index=False)
    
    print("-" * 110)
    print("\nSummary Statistics (Mean +/- Std):")
    summary = df.groupby('Config').agg({
        'ID_MAE': ['mean', 'std'],
        'OOD_MAE': ['mean', 'std'],
        'Input_MAD': ['mean'],
        'Target_MAD': ['mean']
    })
    print(summary)