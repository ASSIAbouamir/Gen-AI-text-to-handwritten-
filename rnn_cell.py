import torch
import torch.nn as nn
import torch.nn.functional as F

# Importe drawing pour MAX_CHAR_LEN (assure-toi d'avoir drawing.py)
import drawing

class LSTMAttentionCell(nn.Module):
    """LSTM cell with attention mechanism - VERSION PYTORCH"""
    
    def __init__(
        self,
        lstm_size,
        num_attn_mixture_components,
        alphabet_size,  # Passé pour cohérence (utilisé dans le modèle pour one-hot), mais pas dans la cell
        num_output_mixture_components,  # Passé pour cohérence, mais pas utilisé dans la cell
        bias_size=None  # Non utilisé dans l'original
    ):
        super(LSTMAttentionCell, self).__init__()
        self.lstm_size = lstm_size
        self.num_attn_mixture_components = num_attn_mixture_components
        self.alphabet_size = alphabet_size
        self.num_output_mixture_components = num_output_mixture_components
        
        # LSTM cell: input_size=6 (inputs[3] + attn_proj[3])
        self.lstm_cell = nn.LSTMCell(6, lstm_size)
        
        # Projection attention: summary [batch, 2] -> [batch, 3] avec tanh
        self.attn_proj = nn.Linear(2, 3)
        
        # Linear pour params attention: lstm_output [batch, lstm_size] -> [batch, 3*K]
        self.attn_params_linear = nn.Linear(lstm_size, 3 * num_attn_mixture_components)
        
        # Bias partagé pour params attention
        self.attn_bias = nn.Parameter(torch.zeros(3 * num_attn_mixture_components))

    @property
    def state_size(self):
        # Équivalent TF: tuple pour compatibilité (h,c ; attention ; kappa)
        return (
            (self.lstm_size, self.lstm_size),  # (h, c) pour LSTM
            drawing.MAX_CHAR_LEN,  # attention [batch, seq_len]
            self.num_attn_mixture_components  # kappa [batch, K]
        )

    @property
    def output_size(self):
        return self.lstm_size

    def zero_state(self, batch_size, device='cpu', dtype=torch.float32):
        # Équivalent TF zero_state
        h = torch.zeros(batch_size, self.lstm_size, dtype=dtype, device=device)
        c = torch.zeros(batch_size, self.lstm_size, dtype=dtype, device=device)
        lstm_zero = (h, c)
        return (
            lstm_zero,
            torch.zeros(batch_size, drawing.MAX_CHAR_LEN, dtype=dtype, device=device),
            torch.zeros(batch_size, self.num_attn_mixture_components, dtype=dtype, device=device)
        )

    def forward(self, inputs, state, attention_values, attention_values_lengths):
        """
        inputs: [batch, 3] (dx, dy, eos)
        state: tuple (lstm_state=(h,c), prev_attention [batch, seq_len], prev_kappa [batch, K])
        attention_values: [batch, seq_len, alphabet_size] (one-hot chars, pour cohérence)
        attention_values_lengths: [batch] (longueurs réelles des séquences chars)
        """
        lstm_state, prev_attention, prev_kappa = state
        h, c = lstm_state
        
        batch_size = inputs.size(0)
        
        # === PROJECTION D'ATTENTION ===
        mean_attn = torch.mean(prev_attention, dim=1)  # [batch]
        max_attn = torch.max(prev_attention, dim=1)[0]  # [batch]
        summary = torch.stack([mean_attn, max_attn], dim=1)  # [batch, 2]
        attn_proj = torch.tanh(self.attn_proj(summary))  # [batch, 3]
        
        # === INPUT LSTM : [3 + 3 = 6] ===
        lstm_input = torch.cat([inputs, attn_proj], dim=1)  # [batch, 6]
        
        # === LSTM STEP ===
        new_h, new_c = self.lstm_cell(lstm_input, (h, c))
        new_lstm_state = (new_h, new_c)
        
        # === ATTENTION COMPUTATION ===
        attention, new_kappa = self._compute_attention(new_h, prev_kappa, attention_values_lengths)
        
        return new_h, (new_lstm_state, attention, new_kappa)

    def _compute_attention(self, lstm_output, prev_kappa, attention_values_lengths):
        K = self.num_attn_mixture_components
        seq_len = drawing.MAX_CHAR_LEN
        batch_size = lstm_output.size(0)
        device = lstm_output.device
        
        # Paramètres d'attention
        attn_params = self.attn_params_linear(lstm_output) + self.attn_bias  # [batch, 3K]
        alpha, beta, kappa_delta = torch.split(attn_params, K, dim=1)
        
        # Activations + accumulation
        alpha = F.softmax(alpha, dim=1)  # [batch, K]
        beta = F.softplus(beta) + 1e-4   # [batch, K]
        kappa = prev_kappa + kappa_delta * 0.1  # [batch, K] (stabilité sans exp)
        
        # Position encoding
        positions = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(0)  # [1, seq_len]
        
        # Gaussiennes [batch, K, seq_len]
        kappa_exp = kappa.unsqueeze(2)      # [batch, K, 1]
        beta_exp = beta.unsqueeze(2)        # [batch, K, 1]
        alpha_exp = alpha.unsqueeze(2)      # [batch, K, 1]
        # Broadcast: positions.unsqueeze(1) [1, 1, seq_len]
        diff = (kappa_exp - positions.unsqueeze(1)) ** 2  # [batch, K, seq_len]
        gaussians = alpha_exp * torch.exp(-beta_exp * diff)
        phi = torch.sum(gaussians, dim=1)   # [batch, seq_len]
        
        # Masquage + normalisation
        arange = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, seq_len]
        mask = (arange < attention_values_lengths.unsqueeze(1)).float()  # [batch, seq_len]
        phi = phi * mask
        phi = phi / (torch.sum(phi, dim=1, keepdims=True) + 1e-8)
        
        return phi, kappa

    def output_function(self, state):
        """Input neutre pour le timestep suivant : [batch, 3] (0,0,1 pour eos)"""
        _, attention, _ = state
        batch = attention.size(0)
        device = attention.device
        return torch.cat([
            torch.zeros(batch, 2, dtype=torch.float32, device=device),
            torch.ones(batch, 1, dtype=torch.float32, device=device)
        ], dim=1)

    def termination_condition(self, state):
        """Toujours False (pas de terminaison early par défaut)"""
        _, attention, _ = state
        batch = attention.size(0)
        device = attention.device
        return torch.zeros(batch, dtype=torch.bool, device=device)


# Test rapide (lance-le pour vérifier)
if __name__ == "__main__":
    batch_size = 4
    seq_len = drawing.MAX_CHAR_LEN
    lstm_size = 256
    K = 5
    alphabet_size = len(drawing.alphabet)  # 75 typique
    num_out_comp = 20

    cell = LSTMAttentionCell(lstm_size, K, alphabet_size, num_out_comp)
    device = torch.device('cpu')  # Ou 'cuda' si GPU
    inputs = torch.randn(batch_size, 3, device=device)
    attention_values = torch.randn(batch_size, seq_len, alphabet_size, device=device)
    lengths = torch.randint(10, seq_len, (batch_size,), device=device)
    state = cell.zero_state(batch_size, device)

    output, new_state = cell(inputs, state, attention_values, lengths)

    print("Input shape:", inputs.shape)
    print("Output shape:", output.shape)
    print("New state types:", [type(s) for s in new_state])
    print("LSTM state shape:", new_state[0][0].shape, new_state[0][1].shape)
    print("Attention shape:", new_state[1].shape)
    print("Kappa shape:", new_state[2].shape)
    print("Test passed! (Shapes OK, no errors)")