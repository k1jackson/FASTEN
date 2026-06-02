from .common import torch, nn, F, np
from .data import Dataset
from .model import Model
from torch.distributions import NegativeBinomial, Binomial
from torch.distributions.kl import register_kl


MAX_STEPS = 100000
MAX_SIZE = 4 * 1024**2

@register_kl(NegativeBinomial, NegativeBinomial)
def KL_negative_binomial(p, q): 
    log_p, log_q = F.logsigmoid(p.logits), F.logsigmoid(q.logits)
    log_neg_p, log_neg_q = F.logsigmoid(-p.logits), F.logsigmoid(-q.logits)
    d_log, d_log_neg = log_p - log_q, log_neg_p - log_neg_q
    mean_p = p.total_count * torch.exp(p.logits)
    kld_exact = mean_p * d_log + p.total_count * d_log_neg
    log_mean_p = torch.log(p.total_count) + p.logits
    std_p = torch.exp((log_mean_p - log_neg_p) / 2)
    return approximate_KL(p, q, mean_p, std_p, kld_exact)

@register_kl(Binomial, Binomial)
def KL_binomial(p, q): 
    log_p, log_q = F.logsigmoid(p.logits), F.logsigmoid(q.logits)
    log_neg_p, log_neg_q = F.logsigmoid(-p.logits), F.logsigmoid(-q.logits)
    d_log, d_log_neg = log_p - log_q, log_neg_p - log_neg_q
    mean_p = p.total_count * torch.sigmoid(p.logits)
    kld_exact = mean_p * d_log + (p.total_count - mean_p) * d_log_neg
    std_p = torch.sqrt(mean_p * (1 - torch.sigmoid(p.logits)))
    return approximate_KL(p, q, mean_p, std_p, kld_exact)

def approximate_KL(p, q, mean_p, std_p, kld_exact, n_stds = 6):
    comparable = torch.isclose(p.total_count, q.total_count)
    if comparable.all(): return kld_exact

    max_k = torch.ceil(mean_p + n_stds * std_p)
    min_k = torch.clamp(torch.floor(mean_p - n_stds * std_p), min = 0.0)
    zeros, ones = torch.zeros_like(std_p), torch.ones_like(std_p)
    total_size = torch.where(comparable, zeros, max_k - min_k)
    integer_size = torch.floor(total_size) + 1
    n_steps = int(max(1, min(integer_size.max().item(), MAX_STEPS)))

    max_width = total_size / max(n_steps - 1, 1)
    width = torch.where(total_size > n_steps, max_width, ones)
    full = torch.full_like(total_size, n_steps)
    valid = torch.where(total_size > n_steps, full, integer_size)
    size = int(max(1, min(MAX_SIZE, n_steps)))

    kld_approx = torch.zeros_like(mean_p)
    min_k, valid = min_k.unsqueeze(0), valid.unsqueeze(0)
    width = width.unsqueeze(0)
    for i in range(0, n_steps, size):
        j = min(i + size, n_steps)
        shape = [-1] + [1] * len(mean_p.shape)
        delta = torch.arange(i, j, device = p.logits.device).view(shape)
        k = torch.where(delta < valid, min_k + delta * width, min_k)
        log_prob_p, log_prob_q = p.log_prob(k), q.log_prob(k)        
        kld = torch.exp(log_prob_p) * (log_prob_p - log_prob_q) * width
        kld = torch.where(delta < valid, kld, torch.zeros_like(kld))
        kld_approx += torch.sum(kld, dim = 0)
    return torch.where(comparable, kld_exact, kld_approx)


class Partition(): 
    def __init__(self, dataset: Dataset, loss_func: str):
        self.dataset: Dataset = dataset
        self.dataloader: torch.utils.data.DataLoader = None
        self.by_sample: bool = (loss_func == "NLL")
        self.loss: list = []

    def load(self, batch_size: int, device: torch.device): 
        data = self.dataset.samples if self.by_sample else self.dataset.stats
        input_tensor = torch.tensor(data.inputs.values).to(device)
        output_tensor = torch.tensor(data.outputs.values).to(device)
        data_tensor = torch.utils.data.TensorDataset(input_tensor, output_tensor)
        self.dataloader = torch.utils.data.DataLoader(data_tensor, batch_size, shuffle = True)
        self.loss = []


class EarlyStop:
    def __init__(self, patience = 20, min_delta = 0, multiplier = 0.1):
        self.multiplier: float = multiplier
        self.patience: float = patience
        self.min_delta: float = min_delta
        self.counter: int = 0
        self.best_loss: float = float('inf')
        self.avg_loss: float = None

    def __call__(self, loss):
        if self.avg_loss is not None: 
            self.avg_loss *= 1 - self.multiplier
            self.avg_loss += self.multiplier * loss
        else: self.avg_loss = loss
        if self.avg_loss < self.best_loss - self.min_delta:
            if self.avg_loss < self.best_loss:
                self.best_loss = self.avg_loss
            self.counter = 0
        else: self.counter += 1
        return (self.counter >= self.patience)
    

class Loss(nn.Module):
    def __init__(self, model: Model):
        super().__init__()
        self.loss_func = model.args.loss_func
        self.mean_squared_error = nn.MSELoss(reduction = "none")
        self.register_buffer("min", torch.from_numpy(model.param_scaler.min))
        self.register_buffer("range", torch.from_numpy(model.param_scaler.range))
        self.load_params(model)

    def load_params(self, model: Model):
        self.num_outputs = len(model.outputs)
        self.dists = [output.dist.base for output in model.outputs.values()]
        masks = torch.zeros((self.num_outputs, len(model.params)), dtype = bool)
        for i, output in enumerate(model.outputs.values()): 
            loop = (param in output.dist.params for param in model.params)
            mask = np.fromiter(loop, dtype = bool, count = len(model.params))
            masks[i] = torch.tensor(mask, dtype = bool)
        self.register_buffer("outputs", masks)

    def forward(self, pred: torch.Tensor, true: torch.Tensor):
        pred_scaled, true_scaled = pred, true
        pred_unscaled = pred * self.range + self.min
        if self.loss_func != "NLL":
            true_unscaled = true * self.range + self.min
        match self.loss_func:
            case "MSE": return self.mean_squared_error(pred_scaled, true_scaled).mean()
            case "KLD": return self.kl_divergence(pred_unscaled, true_unscaled).mean()
            case "NLL": return self.neg_log_likelihood(pred_unscaled, true_scaled).mean()

    def evaluate(self, pred: Dataset, true: Dataset, dependent: bool) -> list[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred_stats_unscaled = torch.tensor(pred.stats.outputs.values)
        true_stats_unscaled = torch.tensor(true.stats.outputs.values)
        pred_stats_scaled = (pred_stats_unscaled - self.min) / self.range
        true_stats_scaled = (true_stats_unscaled - self.min) / self.range
        mse = self.mean_squared_error(pred_stats_scaled, true_stats_scaled)
        kld = self.kl_divergence(pred_stats_unscaled, true_stats_unscaled)
        if dependent: return mse, kld, None
        sample_groups = true.samples.outputs.groupby(true.samples.group)
        nll = torch.zeros((sample_groups.ngroups, pred.samples.outputs.shape[1]), dtype = float)
        for i, sample_group in sample_groups:
            pred_stats = pred_stats_unscaled[i].unsqueeze(0).repeat(len(sample_group), 1)
            true_samples = torch.tensor(sample_group.values)
            nll[i] = self.neg_log_likelihood(pred_stats, true_samples)
        return mse, kld, nll
    
    def neg_log_likelihood(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        neg_log_likelihoods = []
        for i in range(self.num_outputs):
            pred_params = pred[:, self.outputs[i]]
            pred_fit = self.dists[i](*pred_params.unbind(dim = 1))
            loss = -1 * pred_fit.log_prob(true[:,i])
            neg_log_likelihoods.append(loss.mean())
        return torch.column_stack(neg_log_likelihoods)

    def kl_divergence(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        kl_divergences = []
        for i in range(self.num_outputs):
            pred_params = pred[:, self.outputs[i]]
            true_params = true[:, self.outputs[i]]
            pred_fit = self.dists[i](*pred_params.unbind(dim = 1))
            true_fit = self.dists[i](*true_params.unbind(dim = 1))
            loss = torch.distributions.kl_divergence(true_fit, pred_fit)
            kl_divergences.append(loss)
        return torch.column_stack(kl_divergences)