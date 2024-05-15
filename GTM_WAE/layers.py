import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len: int = 50):
        """
        PositionalEncoding implemented with the PyTorch lightning colab tutorial
        Args
            d_model: Hidden dimensionality of the input.
            max_len: Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


class SelfAttnBlock(nn.Module):
    """
    Transfomer implemented with Pytorch lightning course
    args:
        input_dim: dimensionality of the input
        dim_feedforward: dimensionality of the hidden layer in the MLP
        num_heads: number of heads to use in the attention block
        dropout: dropout probability to use in dropout layers
    """

    def __init__(self, vector_dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            embed_dim=vector_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.linear_net = nn.Sequential(
            nn.Linear(vector_dim, vector_dim * 4),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(vector_dim * 4, vector_dim)
        )

        self.layer_norm1 = nn.LayerNorm(vector_dim)
        self.layer_norm2 = nn.LayerNorm(vector_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask=None):
        x = self.layer_norm1(inputs)
        attn_output, _ = self.self_attention(x, x, x, attn_mask=mask)
        x = x + self.dropout(attn_output)
        x = self.layer_norm2(x)
        x = x + self.dropout(self.linear_net(x))
        return x


class EncoderAttention(nn.Module):
    """
    TransfomerEncoder implemented with Pytorch lightning course
    """

    def __init__(self, num_layers=6, vector_dim=256, num_heads=8, dropout=0.0):
        super().__init__()
        self.mha = nn.ModuleList([SelfAttnBlock(vector_dim, num_heads, dropout) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.mha:
            x = layer(x)
        lv = torch.sum(x, -2)
        return lv


class DecoderAttention(nn.Module):
    def __init__(self, lv_dim, output_dim, num_layers=6, num_heads=8, dropout=0.0):
        super().__init__()
        self.self_attn_layers = nn.ModuleList(
            [SelfAttnBlock(lv_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.linear = nn.Linear(lv_dim, output_dim)

    def forward(self, x, attn_mask=None):
        for layer in self.self_attn_layers:
            x = layer(x, attn_mask)
        logits = self.linear(x)
        return logits


class DecoderGRU(nn.Module):
    def __init__(self, lv_dim, output_dim):
        super().__init__()
        self.rnn = nn.GRU(2 * lv_dim, lv_dim, batch_first=True)
        self.dense = nn.Linear(lv_dim, output_dim)
        self.act = nn.ReLU()

    def forward(self, latent_vecs, embeddings):
        _, seq_len, _ = embeddings.shape
        embeddings = torch.cat([latent_vecs.unsqueeze(1), embeddings], 1)
        expanded_latent_vecs = latent_vecs.unsqueeze(1).expand(-1, seq_len + 1, -1)
        dec_inputs = torch.cat([embeddings, expanded_latent_vecs], 2)
        # with torch.backends.cudnn.flags(enabled=False):
        rnn_out, _ = self.rnn(dec_inputs, latent_vecs.unsqueeze(0))
        rnn_out = rnn_out[:, :-1, :]
        logits = self.dense(rnn_out)
        return logits

    def predict_char(self, embedding, latent_vector, hid_state):
        # embedding (1 x embedding_dim)
        # latent_vector (1 x decoder_dim)
        # hid_state (1 x 1 x decoder_dim)
        dec_input = torch.cat([embedding, latent_vector], 1)
        output, hid_state_new = self.rnn(dec_input, hid_state)
        logits = self.dense(output)
        return logits, hid_state_new
