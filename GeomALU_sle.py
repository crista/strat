import torch
import torch.nn as nn

# --- 1. The Geometric Activation (STE) ---
# This remains the same: a hard binary step for logic.
class StepFunctionSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

# --- 2. Step-Gated Linear Unit (Dense / Standard) ---
# NOTE: We use a dense layer here.
# This proves that a Standard Transformer CAN represent the logic.
# The problem (revealed later) is that it can't LEARN it.
class StepGLU(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # Dense layers mixing Data and Control together
        self.w_gate = nn.Linear(d_model, d_model)
        self.w_val = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        # Gate: 0.0 or 1.0
        gate = StepFunctionSTE.apply(self.w_gate(x))
        # Val: Continuous magnitude
        val = self.w_val(x)
        # Gating
        return gate * val

# --- 3. The Model (Standard Architecture) ---
class GeometricTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = 6
        self.SCALE = 100.0
        
        # Dimensions (Mixed Vector):
        # D0: Value (Data)
        # D1: IsNum (Control)
        # D2: IsAdd (Control)
        # D3: IsSub (Control)
        # D4: IsOp  (Control)
        # D5: NegStore (Data Accumulator)

        # --- ATTENTION (Standard Dense) ---
        self.W_q = nn.Linear(6, 6, bias=False)
        self.W_k = nn.Linear(6, 6, bias=False)
        self.W_v = nn.Linear(6, 6, bias=False)
        
        # --- FFN (Dense StepGLU) ---
        self.ffn = StepGLU(6) 
        
        # --- AGGREGATION ---
        self.W_out = nn.Linear(6, 1, bias=False)

        self.hand_code_weights()

    def hand_code_weights(self):
        with torch.no_grad():
            # --- ATTENTION: Binding ---
            # Q: Number (D1) looks for Operator (D4)
            w_q = torch.zeros(6, 6); w_q[4, 1] = 2.0
            # K: Operator (D4) advertises itself
            w_k = torch.zeros(6, 6); w_k[4, 4] = 2.0
            
            self.W_q.weight.copy_(w_q)
            self.W_k.weight.copy_(w_k)
            
            # V: Fetch Op Flags (D2=Add, D3=Sub) -> D2, D3
            w_v = torch.zeros(6, 6)
            w_v[2, 2] = 1.0; w_v[3, 3] = 1.0
            self.W_v.weight.copy_(w_v)

            # --- FFN: The Hidden Logic ---
            # We prove here that a 6x6 matrix HAS the capacity to solve this.
            
            # W_GATE: Controls when channels open
            w_gate = torch.zeros(6, 6); b_gate = torch.zeros(6)
            
            # Logic: Open Channel 5 (NegStore) ONLY IF:
            # 1. It is a Number (D1=1)
            # 2. It has a Subtraction Flag (D3=1)
            # Threshold -1.5 requires both to be True.
            w_gate[5, 1] = 1.0; w_gate[5, 3] = 1.0; b_gate[5] = -1.5
            
            # W_VAL: The data flow
            w_val = torch.zeros(6, 6); b_val = torch.zeros(6)
            
            # Routing: Copy the Value (D0) into Channel 5 (NegStore)
            w_val[5, 0] = 1.0 
            
            self.ffn.w_gate.weight.copy_(w_gate)
            self.ffn.w_gate.bias.copy_(b_gate)
            self.ffn.w_val.weight.copy_(w_val)
            self.ffn.w_val.bias.copy_(b_val)
            
            # --- READOUT ---
            # Total = Sum(Pos) - 2 * Sum(Neg)
            w_out = torch.zeros(1, 6)
            w_out[0, 0] = 1.0
            w_out[0, 5] = -2.0
            self.W_out.weight.copy_(w_out)

    def forward(self, x):
        # 1. Normalization
        x_norm = x.clone()
        x_norm[:, :, 0] = x_norm[:, :, 0] / self.SCALE
        
        seq_len = x.shape[1]
        
        # 2. Attention (Dense)
        Q = self.W_q(x_norm); K = self.W_k(x_norm)
        scores = torch.matmul(Q, K.transpose(-2, -1))
        
        # Locality Bias
        indices = torch.arange(seq_len, device=x.device)
        dist = indices.view(-1, 1) - indices.view(1, -1)
        scores -= dist.float().abs()

        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask == 1, float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)
        V = self.W_v(x_norm)
        Z = torch.matmul(attn_weights, V)
        
        # 3. FFN (Dense)
        combined = x_norm + Z
        ffn_out = self.ffn(combined)
        x_final = combined + ffn_out
        
        # 4. Readout
        res = self.W_out(x_final.sum(dim=1))
        
        # 5. Denormalize
        return res.squeeze(-1) * self.SCALE

# Helpers
def get_tensor_embedding(token):
    x = torch.zeros(6)
    if isinstance(token, (int, float)):
        x[0] = float(token); x[1] = 1.0
    elif token == '+':
        x[2] = 1.0; x[4] = 1.0
    elif token == '-':
        x[3] = 1.0; x[4] = 1.0
    return x

if __name__ == "__main__":
    model = GeometricTransformer()
    
    # Test 1: 12 + 3 - 5 = 10
    tokens = [12, '+', 3, '-', 5]
    x = torch.stack([get_tensor_embedding(t) for t in tokens]).unsqueeze(0)
    
    # Test 2: 8000 - 1000 + 500 = 7500
    tokens_big = [8000, '-', 1000, '+', 500]
    x_big = torch.stack([get_tensor_embedding(t) for t in tokens_big]).unsqueeze(0)

    with torch.no_grad():
        res1 = model(x)
        res2 = model(x_big)
        
    print("--- Hand-Coded Dense Model Verification ---")
    print(f"Input: 12 + 3 - 5    | Output: {res1.item():.2f} (Target: 10.0)")
    print(f"Input: 8000 - 1000 + 500 | Output: {res2.item():.2f} (Target: 7500.0)")