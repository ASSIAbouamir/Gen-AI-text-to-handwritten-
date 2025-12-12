import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Remplacez drawing, DataFrame et LSTMAttentionCell selon l'implémentation spécifique
import drawing

# Remplacez par votre propre code pour la cellule LSTM avec attention
class LSTMAttentionCell(nn.Module):
    def __init__(self, lstm_size, attention_mixture_components, output_mixture_components):
        super(LSTMAttentionCell, self).__init__()
        self.lstm_size = lstm_size
        self.attention_mixture_components = attention_mixture_components
        self.output_mixture_components = output_mixture_components
        self.lstm = nn.LSTM(input_size=3, hidden_size=lstm_size, batch_first=True)  # Exemple avec 3 entrées

    def forward(self, x, state):
        out, state = self.lstm(x, state)
        return out, state

    def zero_state(self, batch_size, dtype=torch.float32):
        return (torch.zeros(batch_size, self.lstm_size, dtype=dtype),
                torch.zeros(batch_size, self.lstm_size, dtype=dtype))


class DataReader:
    def __init__(self, data_dir):
        data_cols = ['x', 'x_len', 'c', 'c_len']
        try:
            data = [np.load(os.path.join(data_dir, '{}.npy'.format(i))) for i in data_cols]
            print(f"Données chargées depuis {data_dir}")
        except FileNotFoundError as e:
            print(f"Fichiers de données non trouvés dans {data_dir}")
            print("Création de données synthétiques pour test...")
            data = self.create_synthetic_data()

        self.test_df = DataFrame(columns=data_cols, data=data)
        self.train_df, self.val_df = self.test_df.train_test_split(train_size=0.95, random_state=2018)

        print('train size', len(self.train_df))
        print('val size', len(self.val_df))
        print('test size', len(self.test_df))

    def create_synthetic_data(self):
        print("Génération de 1000 échantillons synthétiques...")
        
        num_samples = 1000
        max_seq_len = 100
        max_char_len = 20
        
        x = np.random.randn(num_samples, max_seq_len, 3).astype(np.float32)
        x[:, :, 2] = (np.random.rand(num_samples, max_seq_len) > 0.9).astype(np.float32)
        
        x_len = np.random.randint(20, max_seq_len, size=num_samples).astype(np.int32)
        
        c = np.random.randint(0, min(60, len(drawing.alphabet)), size=(num_samples, max_char_len)).astype(np.int32)
        c_len = np.random.randint(5, max_char_len, size=num_samples).astype(np.int32)
        
        print("Données synthétiques créées avec succès!")
        return [x, x_len, c, c_len]

    def train_batch_generator(self, batch_size):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.train_df,
            shuffle=True,
            num_epochs=1000,
            mode='train'
        )

    def val_batch_generator(self, batch_size):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.val_df,
            shuffle=True,
            num_epochs=1000,
            mode='val'
        )

    def test_batch_generator(self, batch_size):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.test_df,
            shuffle=False,
            num_epochs=1,
            mode='test'
        )

    def batch_generator(self, batch_size, df, shuffle=True, num_epochs=1000, mode='train'):
        gen = df.batch_generator(
            batch_size=batch_size,
            shuffle=shuffle,
            num_epochs=num_epochs,
            allow_smaller_final_batch=(mode == 'test')
        )
        for batch in gen:
            batch['x_len'] = batch['x_len'] - 1
            max_x_len = np.max(batch['x_len'])
            max_c_len = np.max(batch['c_len'])

            batch['y'] = batch['x'][:, 1:max_x_len + 1, :]
            batch['x'] = batch['x'][:, :max_x_len, :]
            batch['c'] = batch['c']

            batch['x'] = batch['x'].astype(np.float32)
            batch['y'] = batch['y'].astype(np.float32)
            if batch['x'].shape[-1] != 3:
                raise ValueError(f"Expected x dim 3, got {batch['x'].shape}")
            if batch['y'].shape[-1] != 3:
                raise ValueError(f"Expected y dim 3, got {batch['y'].shape}")

            yield batch


class RNN(nn.Module):
    def __init__(self, lstm_size, output_mixture_components, attention_mixture_components):
        super(RNN, self).__init__()
        self.lstm_size = lstm_size
        self.output_mixture_components = output_mixture_components
        self.output_units = self.output_mixture_components * 6 + 1
        self.attention_mixture_components = attention_mixture_components
        
        self.cell = LSTMAttentionCell(lstm_size, attention_mixture_components, output_mixture_components)
        self.gmm_layer = nn.Linear(lstm_size, self.output_units)

    def parse_parameters(self, z, eps=1e-8, sigma_eps=1e-4):
        K = self.output_mixture_components
        pis, sigmas, rhos, mus, es = torch.split(z, [K, 2 * K, K, 2 * K, 1], dim=-1)
        pis = F.softmax(pis, dim=-1)
        sigmas = torch.clamp(torch.exp(sigmas), min=sigma_eps)
        rhos = torch.clamp(torch.tanh(rhos), min=eps - 1.0, max=1.0 - eps)
        es = torch.clamp(torch.sigmoid(es), min=eps, max=1.0 - eps)
        return pis, mus, sigmas, rhos, es

    def nll(self, y, lengths, pis, mus, sigmas, rhos, es, eps=1e-8):
        sigma_1, sigma_2 = torch.split(sigmas, 1, dim=-1)
        y_1, y_2, y_3 = torch.split(y, 1, dim=-1)
        mu_1, mu_2 = torch.split(mus, 1, dim=-1)

        norm = 1.0 / (2 * np.pi * sigma_1 * sigma_2 * torch.sqrt(1 - rhos**2 + eps))
        Z = ((y_1 - mu_1) / sigma_1)**2 + ((y_2 - mu_2) / sigma_2)**2 - 2 * rhos * ((y_1 - mu_1) * (y_2 - mu_2)) / (sigma_1 * sigma_2)
        exp = -0.5 * Z / (1 - rhos**2 + eps)
        gauss_lik = torch.exp(exp) * norm
        gmm_lik = torch.sum(pis * gauss_lik, dim=-1)
        gmm_lik = torch.clamp(gmm_lik, min=eps)

        bern_lik = torch.where(y_3 == 1, es, 1 - es).squeeze(-1)
        bern_lik = torch.clamp(bern_lik, min=eps)

        nll = -(torch.log(gmm_lik) + torch.log(bern_lik))
        T = y.shape[1]
        seq_mask = torch.arange(T, device=y.device).unsqueeze(0) < lengths.unsqueeze(1)
        nll = nll.masked_fill(~seq_mask, 0.0)
        num_valid = seq_mask.sum(dim=1, dtype=torch.float32)

        sequence_loss = nll.sum(dim=1) / torch.clamp(num_valid, min=1.0)
        element_loss = nll.sum() / torch.clamp(num_valid.sum(), min=1.0)
        return sequence_loss.mean(), element_loss.mean()

    def forward(self, x, c, c_len, x_len, y):
        B, T, _ = x.shape
        
        attention_values = F.one_hot(c, len(drawing.alphabet)).float()
        initial_state = self.cell.zero_state(B, dtype=torch.float32)
        
        outputs, _ = self._rnn_free_run(x, x_len, initial_state, attention_values, c_len, sample_tsteps=T)
        
        params = self.gmm_layer(outputs)
        pis, mus, sigmas, rhos, es = self.parse_parameters(params)
        
        seq_loss, elem_loss = self.nll(y, x_len - 1, pis[:, :-1], mus[:, :-1], sigmas[:, :-1], rhos[:, :-1], es[:, :-1])
        return seq_loss

    def _rnn_free_run(self, x, seq_len, initial_state, attention_values, c_len, sample_tsteps=10):
        B, T, _ = x.shape
        outputs = []
        state = initial_state

        for t in range(sample_tsteps):
            input_t = x[:, t, :].unsqueeze(1)  # [B, 1, 3]

            output_t, state = self.cell(input_t, state)
            outputs.append(output_t)

        outputs = torch.cat(outputs, dim=1)  # [B, T, lstm_size]
        return outputs, state


if __name__ == '__main__':
    print("="*60)
    print("DÉMARRAGE DE L'ENTRAÎNEMENT DU MODÈLE RNN")
    print("="*60)
    
    # Initialisation du lecteur de données
    dr = DataReader(data_dir='data/processed/')
    
    print("\n" + "="*60)
    print("CONFIGURATION DU MODÈLE")
    print("="*60)
    
    # Création et entraînement du modèle
    nn = RNN(
        lstm_size=400,
        output_mixture_components=20,
        attention_mixture_components=10
    )
    
    print("\n" + "="*60)
    print("LANCEMENT DE L'ENTRAÎNEMENT")
    print("="*60 + "\n")
    
    # Lancer l'entraînement (implémentation d'entraînement nécessaire)
    # nn.fit()
    
    print("\n" + "="*60)
    print("ENTRAÎNEMENT TERMINÉ!")
    print("="*60)
