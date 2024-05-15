import math
import torch


# A majority of functions were taken from https://github.com/IBM/controlled-peptide-generation


def recrate(preds: torch.Tensor, target: torch.Tensor):
    preds = preds.argmax(-1)
    correct = torch.sum(torch.all(preds == target, dim=-1))
    total = target.shape[0]
    return correct.float() / total


def kl_gaussianprior(mu, logvar):
    """ analytically compute kl divergence with unit gaussian. """
    return torch.mean(0.5 * torch.sum((logvar.exp() + mu ** 2 - 1 - logvar), 1))


def kl_gaussian_sharedmu(mu, logvar):
    """ analytically compute kl divergence N(mu,sigma) with N(mu, I). """
    return torch.mean(0.5 * torch.sum((logvar.exp() - 1 - logvar), 1))


def wae_mmd_gaussianprior(z, method='full_kernel'):
    """ compute MMD with samples from unit gaussian.
    MMD parametrization from cfg loaded here."""
    z_prior = torch.randn_like(z)  # shape and device
    if method == 'full_kernel':
        mmd_kwargs = {'sigma': 7.0, 'kernel': 'gaussian'}
        return mmd_full_kernel(z, z_prior, **mmd_kwargs)
    else:
        mmd_kwargs = {"sigma": 7.0,  # ~ O( sqrt(z_dim) )
                      "kernel": 'gaussian',
                      "rf_dim": 500,  # for method = rf
                      "rf_resample": False}
        return mmd_rf(z, z_prior, **mmd_kwargs)


def mmd_full_kernel(z1, z2, **mmd_kwargs):
    K11 = compute_mmd_kernel(z1, z1, **mmd_kwargs)
    K22 = compute_mmd_kernel(z2, z2, **mmd_kwargs)
    K12 = compute_mmd_kernel(z1, z2, **mmd_kwargs)
    N = z1.size(0)
    assert N == z2.size(0), 'expected matching sizes z1 z2'
    H = K11 + K22 - K12 * 2  # gretton 2012 eq (4)
    H = H - torch.diag(H)  # unbiased: delete diagonal. Makes MMD^2_u negative! (typically)
    loss = 1. / (N * (N - 1)) * H.sum()
    return loss


def mmd_rf(z1, z2, **mmd_kwargs):
    mu1 = compute_mmd_mean_rf(z1, **mmd_kwargs)
    mu2 = compute_mmd_mean_rf(z2, **mmd_kwargs)
    loss = ((mu1 - mu2) ** 2).sum()
    return loss


rf = {}


def compute_mmd_mean_rf(z, sigma, kernel, rf_dim, rf_resample=False):
    # random features approx of gaussian kernel mmd.
    # rf_resample: keep fixed base of RF? or resample RF every time?
    # Then just loss = |mu_real - mu_fake|_H
    global rf
    if kernel == 'gaussian':
        if kernel not in rf or rf_resample:
            # sample rf if it's the first time or we want to resample every time
            rf_w = torch.randn((z.shape[1], rf_dim), device=z.device)
            rf_b = math.pi * 2 * torch.rand((rf_dim,), device=z.device)
            rf['gaussian'] = (rf_w, rf_b)
        else:
            rf_w, rf_b = rf['gaussian']
            assert rf_w.shape == (z.shape[1], rf_dim), 'not expecting z dim or rf_dim to change'
        z_rf = compute_gaussian_rf(z, rf_w, rf_b, sigma, rf_dim)
    else:  # kernel xxx
        raise ValueError('todo implement rf for kernel ' + kernel)
    mu_rf = z_rf.mean(0, keepdim=False)
    return mu_rf


def compute_gaussian_rf(z, rf_w, rf_b, sigma, rf_dim):
    z_emb = (z @ rf_w) / sigma + rf_b
    z_emb = torch.cos(z_emb) * (2. / rf_dim) ** 0.5
    return z_emb


def compute_mmd_kernel(x, y, sigma, kernel):
    """ x: (Nxd) y: (Mxd). sigma: kernel width """
    # adapted from https://discuss.pytorch.org/t/error-when-implementing-rbf-kernel-bandwidth-differentiation-in-pytorch/13542
    x_i = x.unsqueeze(1)
    y_j = y.unsqueeze(0)
    xmy = ((x_i - y_j) ** 2).sum(2)
    if kernel == "gaussian":
        K = torch.exp(- xmy / sigma ** 2)
    elif kernel == "laplace":
        K = torch.exp(- torch.sqrt(xmy + (sigma ** 2)))
    elif kernel == "energy":
        K = torch.pow(xmy + (sigma ** 2), -.25)
    return K

# class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
#     def __init__(self, optimizer, warmup, max_iters):
#         self.warmup = warmup
#         self.max_num_iters = max_iters
#         super().__init__(optimizer)

#     def get_lr(self):
#         lr_factor = self.get_lr_factor(epoch=self.last_epoch)
#         return [base_lr * lr_factor for base_lr in self.base_lrs]

#     def get_lr_factor(self, epoch):
#         lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
#         if epoch <= self.warmup:
#             lr_factor *= epoch * 1.0 / self.warmup
#         return lr_factor
