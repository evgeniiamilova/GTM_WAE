import torch
import torch.distributions.normal as tdist
import torch.nn as nn
import torch.nn.functional as F
import torch_optimizer as optim
from pytorch_lightning import LightningModule
from typing import Optional
import numpy as np
from src.gtm_wae.layers import DecoderGRU, EncoderAttention, PositionalEncoding
from src.gtm_wae.metrics import recrate, kl_gaussianprior, wae_mmd_gaussianprior, kl_gaussian_sharedmu


class GTM_WAE(LightningModule):
    def __init__(self, vocab_size, max_len, num_heads=8, num_layers=6, dropout=0.0, lv_dim=256, lr=0.0005,
                 vae=True, vae_loss_type="mmdrf", vae_beta=0.05, lambda_logvar_l1=1e-3, lambda_logvar_kl=1e-3,
                 task='train'):
        super().__init__()
        self.task = task
        self.save_hyperparameters()
        self.lr = lr
        self.vocab_size = vocab_size  # Redone by K 020223
        self.max_seq_len = max_len
        self.lv_dim = lv_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.vae = vae
        self.vae_loss_type = vae_loss_type  # 'kl', 'mmd', 'mmdrf'
        self.vae_beta = vae_beta
        self.lambda_logvar_l1 = lambda_logvar_l1
        self.lambda_logvar_kl = lambda_logvar_kl

        self.embeddings = nn.Embedding(self.vocab_size, lv_dim, padding_idx=0)
        self.positional_encoding = PositionalEncoding(d_model=lv_dim, max_len=self.max_seq_len)
        self.encoder = EncoderAttention(
            num_layers=num_layers,
            vector_dim=lv_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        self.decoder = DecoderGRU(lv_dim, self.vocab_size)
        self.size_predictor = nn.Linear(lv_dim, 1)

        if vae:
            self.q_mu = nn.Linear(lv_dim, lv_dim)
            self.q_logvar = nn.Linear(lv_dim, lv_dim)

    def forward(self, inputs):
        if self.task in ['train']:
            return self.reconstruct(inputs)
        elif self.task == 'encode':
            return self.encode(inputs)

    def reconstruct(self, inputs):
        embeds = self.embeddings(inputs.long())
        x = self.positional_encoding(embeds)
        latent_vecs = self.encoder(x)
        logits = self.decoder(latent_vecs)
        return logits

    @torch.no_grad()
    def encode(self, inputs):
        inputs = inputs.long()
        embeds = self.embeddings(inputs)
        x = self.positional_encoding(embeds)
        latent_vecs = self.encoder(x)
        if self.vae:
            latent_vecs = self.q_mu(latent_vecs)
        return latent_vecs

    @torch.no_grad()
    def sample(self, latent_vec):
        latent_vec = latent_vec.unsqueeze(0)
        size = self.size_predictor(latent_vec)
        size = int(torch.round(size).item())
        embedding = h = latent_vec
        result = torch.zeros(self.max_seq_len)
        for aai in range(size):
            logits, h = self.decoder.predict_char(embedding, latent_vec, h)
            max_idx = torch.argmax(logits)
            embedding = torch.unsqueeze(self.embeddings(max_idx), 0)
            result[aai] = max_idx
        return result

    def sample_z_prior(self):  # Random sampling from normal distribution
        """
        Sample z ~ p(z) = N(0, I)
        """
        z = torch.randn(self.lv_dim).to(self.device)
        # print("z type", type(z), z, type(z[0]))
        return z


    def sample_from_node(self, loc: np.ndarray, scale: float = 1.0, seed: Optional[int] = None) -> torch.Tensor:
        """
        Sample z ~ N(loc, scale) with an optional seed for reproducibility
        on this call only.
    
        :param loc: Location of distribution, expected to be a NumPy array
        :param scale: Scale (standard deviation) of the distribution (default = 1.0)
        :param seed: Optional seed to fix RNG state for this call
        :return: Sampled tensor
        """
    
        # 1) If a seed is provided, store and reset RNG state
        if seed is not None:
            # Store current CPU RNG state
            cpu_rng_state = torch.get_rng_state()
    
            # If using a single GPU, store its RNG state as well
            if torch.cuda.is_available():
                gpu_rng_state = torch.cuda.get_rng_state(device=self.device)
    
            # Set a new, temporary seed
            torch.manual_seed(seed)
            # np.random.seed(seed)  # Uncomment if NumPy-based randomness is used
            # random.seed(seed)      # Uncomment if Python's random module is used
    
        # 2) Convert inputs to PyTorch tensors and sample from Normal distribution
        loc_tensor = torch.from_numpy(loc).float()
        scale_tensor = torch.tensor([scale], dtype=torch.float32)
        z = tdist.Normal(loc_tensor, scale_tensor)
        lv = z.sample().float().to(self.device)
    
        # 3) If a seed was set, restore the old RNG state(s) to avoid affecting the rest of the program
        if seed is not None:
            torch.set_rng_state(cpu_rng_state)
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(gpu_rng_state, device=self.device)
    
        return lv


    def _get_loss(self, batch, batch_idx):
        inputs = batch.long()
        true_one_hot, mask = self._masked_one_hot(inputs)
        true_size = torch.squeeze(mask).sum(-1).float()
        embeds = self.embeddings(inputs)
        x = self.positional_encoding(embeds)
        latent_vecs = self.encoder(x)

        if self.vae:
            mu = self.q_mu(latent_vecs)
            logvar = self.q_logvar(latent_vecs)
            z = self._sample_z(latent_vecs, mu, logvar)
            preds = self.decoder(z, embeds)
            pred_size = self.size_predictor(z)
        else:
            preds = self.decoder(latent_vecs, embeds)
            # TODO showmodif substitute z by latent_vecs below
            pred_size = self.size_predictor(latent_vecs)

        pred_size = torch.squeeze(pred_size)

        bce_loss = F.binary_cross_entropy_with_logits(preds, true_one_hot)

        size_loss = F.mse_loss(pred_size, true_size)

        if self.vae:
            wae_l = self._wae_loss(z, mu, logvar)
            loss = bce_loss + wae_l + torch.tensor(0.25) * size_loss
        else:
            loss = bce_loss + torch.tensor(0.25) * size_loss

        rec_rate = recrate(preds * mask, inputs)
        metrics = {"loss": loss, "rec_rate": rec_rate}
        if self.vae:
            metrics["wae_loss"] = wae_l
        return metrics

    def _masked_one_hot(self, inputs):
        inputs = inputs.long()
        mask = torch.where(inputs > 0, 1, 0)
        mask = torch.unsqueeze(mask, -1)
        true_one_hot = F.one_hot(inputs, self.vocab_size)
        true_one_hot = true_one_hot * mask
        true_one_hot = true_one_hot.float()
        return true_one_hot, mask

    def _sample_z(self, latent, mu, logvar):
        """
        Reparameterization trick: z = mu + std*eps; eps ~ N(0, I)
        """
        eps = torch.randn((mu.size(0), self.lv_dim), device=mu.device)
        return mu + torch.exp(logvar / 2) * eps

    def _wae_loss(self, z, z_mu, z_logvar):
        kl_loss = kl_gaussianprior(z_mu, z_logvar)
        wae_mmd_loss = wae_mmd_gaussianprior(z, method='full_kernel')
        wae_mmdrf_loss = wae_mmd_gaussianprior(z, method='rf')
        z_regu_losses = {'kl': kl_loss, 'mmd': wae_mmd_loss, 'mmdrf': wae_mmdrf_loss}
        z_regu_loss = z_regu_losses[self.vae_loss_type]
        z_logvar_l1 = z_logvar.abs().sum(1).mean(0)  # L1 in z-dim, mean over mb.
        z_logvar_kl_penalty = kl_gaussian_sharedmu(z_mu, z_logvar)
        loss = (self.vae_beta * z_regu_loss + self.lambda_logvar_l1 * z_logvar_l1
                + self.lambda_logvar_kl * z_logvar_kl_penalty)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdaBelief(
            self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-16,
            weight_decay=1e-4,
            amsgrad=False,
            weight_decouple=True,
            fixed_decay=False,
            rectify=True
        )
        return optimizer

    # def backward(self, loss, *agrs, **kwargs):
    #     loss.backward(create_graph=True)

    def training_step(self, batch, batch_idx):
        metrics = self._get_loss(batch, batch_idx)
        for metric_name, metric_value in metrics.items():
            self.log(f"train_{metric_name}", metric_value)
        return metrics["loss"]

    def validation_step(self, batch, batch_idx):
        metrics = self._get_loss(batch, batch_idx)
        for metric_name, metric_value in metrics.items():
            self.log(f"val_{metric_name}", metric_value)
