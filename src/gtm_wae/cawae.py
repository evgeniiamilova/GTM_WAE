import torch
import torch.distributions.normal as tdist
import torch.nn as nn
import torch.nn.functional as F
import torch_optimizer as optim
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb

#from CAWAE.layers import EncoderAttention, PositionalEncoding, DecoderAttention
#from CAWAE.metrics import recrate, kl_gaussianprior, wae_mmd_gaussianprior, kl_gaussian_sharedmu

from src.gtm_wae.layers import EncoderAttention, PositionalEncoding, DecoderAttention
from src.gtm_wae.metrics import recrate, kl_gaussianprior, wae_mmd_gaussianprior, kl_gaussian_sharedmu


class CAWAE(LightningModule):
    """
    Conditional Attention-based Wasserstein AutoEncoder implemented with Pytorch Lightning.

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary, which provides the number of classes.
    max_len : int
        Maximum length of a sequence.
    num_heads : int, optional
        Number of attention heads. Default is 8.
    num_layers : int, optional
        Number of layers in the encoder and decoder. Default is 6.
    dropout : float, optional
        Dropout rate. Default is 0.0.
    lv_dim : int, optional
        Dimensionality of the latent space. Default is 256.
    lr : float, optional
        Learning rate. Default is 0.0005.
    vae : bool, optional
        Whether to use a Variational AutoEncoder. Default is True.
    vae_loss_type : str, optional
        Type of loss to use for the VAE. Options are 'kl', 'mmd', 'mmdrf'. Default is 'mmdrf'.
    vae_beta : float, optional
        Beta value for the VAE loss. Default is 0.05.
    lambda_logvar_l1 : float, optional
        Lambda value for L1 regularization on log variance. Default is 1e-3.
    lambda_logvar_kl : float, optional
        Lambda value for KL divergence regularization on log variance. Default is 1e-3.
    task : str, optional
        Task for the module. Options are 'train', 'encode'. Default is 'train'.

    Attributes
    ----------
    embeddings : torch.nn.Embedding
        Embedding layer.
    positional_encoding : PositionalEncoding
        Positional encoding layer.
    encoder : EncoderAttention
        Encoder module.
    decoder : DecoderAttention
        Decoder module.
    size_predictor : torch.nn.Linear
        Linear layer for size prediction.
    q_mu : torch.nn.Linear
        Linear layer for mean of q distribution.
    q_logvar : torch.nn.Linear
        Linear layer for log variance of q distribution.

    """

    def __init__(self, vocab_size, max_len, num_heads=8, num_layers=6, dropout=0.0, lv_dim=256, lr=0.0005,
                 vae=True, vae_loss_type="mmdrf", vae_beta=0.05, lambda_logvar_l1=1e-3, lambda_logvar_kl=1e-3,
                 task='train', debug=False):
        super().__init__()
        self.task = task
        self.debug = debug
        self.save_hyperparameters()
        self.lr = lr
        self.vocab_size = vocab_size
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
        self.decoder = DecoderAttention(lv_dim, self.vocab_size)
        self.size_predictor = nn.Linear(lv_dim, 1)

        if vae:
            self.q_mu = nn.Linear(lv_dim, lv_dim)
            self.q_logvar = nn.Linear(lv_dim, lv_dim)

        # self.apply(self.init_weights) #init weigths
    def forward(self, inputs):
        """
        Forward pass through the model.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor from the reconstruction or encoding, depending on the task.

        """
        if self.task in ['train']:
            return self.reconstruct(inputs)
        elif self.task == 'encode':
            return self.encode(inputs)

    def reconstruct(self, inputs):
        embeds = self.embeddings(inputs.long())
        x = self.positional_encoding(embeds)
        latent_vecs = self.encoder(x, inputs)
        logits = self.decoder(latent_vecs)
        return logits

    @torch.no_grad()
    def encode(self, inputs):
        """
        Encode the input sequence into the latent space.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor.

        Returns
        -------
        latent_vecs : torch.Tensor
            Latent vectors representing the inputs in the latent space.
        """
        inputs = inputs.long()
        embeds = self.embeddings(inputs)
        x = self.positional_encoding(embeds)
        latent_vecs = self.encoder(x, inputs)
        if self.vae:
            latent_vecs = self.q_mu(latent_vecs)
        return latent_vecs

    @torch.no_grad()
    def sample(self, latent_vec):
        """
        Generate a novel sequence using the provided latent vector.

        Parameters
        ----------
        latent_vec : torch.Tensor
            Latent vector used as input for the generation process.

        Returns
        -------
        result : torch.Tensor
            A tensor representing the generated sequence.
        """
        latent_vec = latent_vec.unsqueeze(0)
        size = self.size_predictor(latent_vec)
        size = int(torch.round(size).item())

        zeros_mtx = torch.zeros((self.max_seq_len, latent_vec.shape[-1]))
        pure_pos_enc_mtx = self.positional_encoding(zeros_mtx).squeeze(0)
        dec_inp = torch.concat([latent_vec.squeeze(1), zeros_mtx], dim=0)
        result = torch.zeros(self.max_seq_len)
        for i, aai in enumerate(range(size)):
            logits = self.decoder(dec_inp)
            max_idx = torch.argmax(logits[i])
            result[aai] = max_idx
            dec_inp[i + 1, :] = self.embeddings(max_idx) + pure_pos_enc_mtx[i, :]
        return result


    def sample_from_node(self, loc, scale: float = 0.1):  # Sampling from the node
        """
        Sample from a normal distribution with provided location and scale.

        Parameters
        ----------
        loc : numpy.ndarray
            Location parameter (mean) of the normal distribution.
        scale : float, optional
            Scale parameter (standard deviation) of the normal distribution. Default is 0.1.
            PyTorch applies broadcasting to the scale tensor. Since scale is of shape (1,),
            it's automatically expanded to match the shape of loc ((3,)) during operations.
            Each element in loc is paired with the single value in scale, effectively using
            0.1 as the standard deviation for each mean value in loc.

        Returns
        -------
        lv : torch.Tensor
            A sample drawn from the normal distribution.

        """
        loc = torch.from_numpy(loc) #TODO: place this function out of this class since it does not have a self
        scale = torch.tensor([scale])
        z = tdist.Normal(loc, scale)
        lv = z.sample().float() # this sample() is function of pytorch, not the one defined in class
        # print("z type", type(z), z)
        return lv

    def sample_z_prior(self):
        """
        Sample from a standard normal distribution.

        Returns
        -------
        z : torch.Tensor
            Sampled latent vector from a standard normal distribution. Dimension of the vector is `self.lv_dim`.
        """
        z = torch.randn(self.lv_dim).to(self.device)
        return z

    def _make_attn_mask(self, embeds):
        """
        Create an attention mask for the self-attention mechanism.

        Parameters
        ----------
        embeds : torch.Tensor
            Input embeddings tensor with shape [batch_size, seq_len, embed_dim].

        Returns
        -------
        attn_mask : torch.Tensor
            Boolean tensor representing the attention mask, with `False` values where attention is allowed and `True` values where it is blocked. Shape is [seq_len + 1, seq_len + 1].
        """
        batch_size, seq_len, embed_dim = embeds.shape
        attn_mask = torch.triu(torch.ones((seq_len + 1, seq_len + 1)), diagonal=1).bool()  # batch_size, num_heads,
        attn_mask = attn_mask.to(self.device)
        return attn_mask

    def _get_loss(self, batch, batch_idx):
        inputs = batch.long()
        if self.debug:
            print("INPUTS: ", inputs)
        true_one_hot, mask = self._masked_one_hot(inputs)
        if self.debug:
            print("TRUE_ONE_HOT+mask: ", true_one_hot.sum, "\n", mask)
        true_size = torch.squeeze(mask).sum(-1).float()
        if self.debug:
            print("EMBEDS: ", embeds)
        embeds = self.embeddings(inputs)
        # x = self.input_net(embeds) #commented on purpose
        x = self.positional_encoding(embeds)
        latent_vecs = self.encoder(x, inputs)

        if self.vae:
            mu = self.q_mu(latent_vecs)
            logvar = self.q_logvar(latent_vecs)
            z = self._reparametrization_trick(latent_vecs, mu, logvar)
            attn_mask = self._make_attn_mask(x)
            dec_inp = torch.concat([z.unsqueeze(1), x], dim=1)
            preds = self.decoder(dec_inp, attn_mask)
            preds = preds[:, :-1, :]
            pred_size = self.size_predictor(z)
        else:
            preds = self.decoder(latent_vecs, embeds)
            # TODO showmodif substitute z by latent_vecs below
            pred_size = self.size_predictor(latent_vecs)

        pred_size = torch.squeeze(pred_size)

        #print("preds+inputs", preds.shape, inputs.shape)
        #print(preds.shape, true_one_hot.shape)

        bce_loss = F.binary_cross_entropy_with_logits(preds, true_one_hot)

        size_loss = F.mse_loss(pred_size, true_size)

        if self.vae:
            wae_l = self._wae_loss(z, mu, logvar)
            loss = bce_loss + wae_l + torch.tensor(0.20) * size_loss
        else:
            loss = bce_loss + torch.tensor(0.20) * size_loss

        rec_rate = recrate(preds * mask, inputs)
        metrics = {
            "loss": loss,
            "rec_loss": bce_loss,
            "rec_rate": rec_rate,
            "size_loss": size_loss,
        }
        if self.vae:
            metrics["wae_loss"] = wae_l
        return metrics

    def _masked_one_hot(self, inputs):
        inputs = inputs.long()
        mask = torch.where(inputs > 0, 1, 0)
        #print("inputs", inputs.shape)
        mask = torch.unsqueeze(mask, -1)
        #print("mask", mask.shape)
        true_one_hot = F.one_hot(inputs, self.vocab_size)
        #print("true_one_hot", true_one_hot.shape)
        #print("vocab", self.vocab_size)
        true_one_hot = true_one_hot * mask
        true_one_hot = true_one_hot.float()
        return true_one_hot, mask

    def _reparametrization_trick(self, latent, mu, logvar): # was called _sample_z previously
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
        z_logvar_l1 = z_logvar.abs().sum(1).mean(0)  # L1 in z-dim, mean over mini batch of data
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
        lr_scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.8, min_lr=5e-5, verbose=True)
        scheduler = {
            'scheduler': lr_scheduler,
            'reduce_on_plateau': True,
            'monitor': 'val_loss'
        }

        return [optimizer], [scheduler]

    # def backward(self, loss, *agrs, **kwargs):
    #     loss.backward(create_graph=True)

    def update_beta(self, beta_value, epoch):
        """Update the beta value for the KL divergence term."""
        self.vae_beta = beta_value
        print("Epoch {}: current_beta: {}".format(epoch, beta_value))
        wandb.log({'epoch': epoch, 'current_beta': beta_value})

    def training_step(self, batch, batch_idx):
        metrics = self._get_loss(batch, batch_idx)
        for metric_name, metric_value in metrics.items():
            self.log(f"train_{metric_name}", metric_value)
        return metrics["loss"]

    def validation_step(self, batch, batch_idx):
        metrics = self._get_loss(batch, batch_idx)
        for metric_name, metric_value in metrics.items():
            self.log(f"val_{metric_name}", metric_value)
