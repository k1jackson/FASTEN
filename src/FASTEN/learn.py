from .common import torch, nn, F, np
from .data import Dataset
from .model import Model
from torch.distributions import NegativeBinomial, Binomial
from torch.distributions.kl import register_kl

@register_kl(NegativeBinomial, NegativeBinomial)
def KL_negative_binomial(p, q, n_stds = 3, memory_chunk = int(1e5)): 
    comparable = torch.isclose(p.total_count, q.total_count)
    log_p = F.logsigmoid(p.logits)
    log_q = F.logsigmoid(q.logits)
    log_neg_p = F.logsigmoid(-p.logits)
    log_neg_q = F.logsigmoid(-q.logits)
    prob_p = torch.sigmoid(p.logits)
    mean_p = p.total_count * prob_p / (1 - prob_p)
    term = mean_p * (log_p - log_q)
    term.nan_to_num_(nan = 0.0)
    kld_exact = term + p.total_count * (log_neg_p - log_neg_q)
    if comparable.all(): return kld_exact

    var_p = mean_p / (1 - prob_p)
    std_p = torch.sqrt(var_p)
    max_k = torch.ceil(mean_p + n_stds * std_p).nan_to_num_(posinf = 1e20)
    min_k = torch.clamp(torch.floor(mean_p - n_stds * std_p), min = 0.0)
    max_steps = int((max_k - min_k).max().item()) + 1
    max_k, min_k = max_k.unsqueeze(0), min_k.unsqueeze(0)

    kld_approx = torch.zeros_like(mean_p)
    for i in range(0, max_steps, memory_chunk):
        chunk_end = min(i + memory_chunk, max_steps)
        delta_chunk = torch.arange(i, chunk_end, device = p.logits.device, dtype = p.logits.dtype)
        delta_expanded = delta_chunk.view(*([-1] + [1] * len(p.batch_shape)))
        k_expanded = min_k + delta_expanded
        step_range = (k_expanded <= max_k)
        k_valid = torch.where(step_range, k_expanded, min_k)

        log_prob_p = p.log_prob(k_valid)
        log_prob_q = q.log_prob(k_valid)
        kld = torch.exp(log_prob_p) * (log_prob_p - log_prob_q)
        kld = torch.where(step_range, kld, torch.zeros_like(kld))
        kld.nan_to_num_(nan = 0.0)
        kld_approx += torch.sum(kld, dim = 0)
    return torch.where(comparable, kld_exact, kld_approx)

@register_kl(Binomial, Binomial)
def KL_binomial(p, q, n_stds = 3, memory_chunk = int(1e5)): 
    comparable = torch.isclose(p.total_count, q.total_count)
    log_p = F.logsigmoid(p.logits)
    log_q = F.logsigmoid(q.logits)
    log_neg_p = F.logsigmoid(-p.logits)
    log_neg_q = F.logsigmoid(-q.logits)
    prob_p = torch.sigmoid(p.logits)
    mean_p = p.total_count * prob_p
    term = mean_p * (log_p - log_q)
    term.nan_to_num_(nan = 0.0)
    kld_exact = term + (p.total_count - mean_p) * (log_neg_p - log_neg_q)
    if comparable.all(): return kld_exact

    var_p = mean_p * (1 - prob_p)
    std_p = torch.sqrt(var_p)

    max_k = torch.clamp(torch.ceil(mean_p + n_stds * std_p), max = p.total_count)
    min_k = torch.clamp(torch.floor(mean_p - n_stds * std_p), min = 0.0)
    max_steps = int((max_k - min_k).max().item()) + 1
    max_k, min_k = max_k.unsqueeze(0), min_k.unsqueeze(0)
    
    kld_approx = torch.zeros_like(mean_p)
    for i in range(0, max_steps, memory_chunk):
        chunk_end = min(i + memory_chunk, max_steps)
        delta_chunk = torch.arange(i, chunk_end, device = p.logits.device, dtype = p.logits.dtype)
        delta_expanded = delta_chunk.view(*([-1] + [1] * len(p.batch_shape)))
        k_expanded = min_k + delta_expanded
        step_range = (k_expanded <= max_k)
        k_p_expanded = torch.min(torch.where(step_range, k_expanded, min_k), p.total_count)
        k_q_expanded = torch.min(torch.where(step_range, k_expanded, min_k), q.total_count)
        
        log_prob_p = p.log_prob(k_p_expanded)
        log_prob_q = q.log_prob(k_q_expanded)
        kld = torch.exp(log_prob_p) * (log_prob_p - log_prob_q)
        kld = torch.where(k_expanded <= p.total_count, kld, torch.zeros_like(kld))
        kld = torch.where(step_range, kld, torch.zeros_like(kld))
        kld.nan_to_num_(nan = 0.0)
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
        pred_stats_unscaled = torch.from_numpy(pred.stats.outputs.values)
        true_stats_unscaled = torch.from_numpy(true.stats.outputs.values)
        pred_stats_scaled = (pred_stats_unscaled - self.min) / self.range
        true_stats_scaled = (true_stats_unscaled - self.min) / self.range
        mse = self.mean_squared_error(pred_stats_scaled, true_stats_scaled)
        kld = self.kl_divergence(pred_stats_unscaled, true_stats_unscaled)
        if dependent: return mse, kld, None
        sample_groups = true.samples.outputs.groupby(true.samples.group)
        nll = torch.zeros((sample_groups.ngroups, pred.samples.outputs.shape[1]), dtype = float)
        for i, sample_group in sample_groups:
            pred_stats = pred_stats_unscaled[i].unsqueeze(0).repeat(len(sample_group), 1)
            true_samples = torch.from_numpy(sample_group.values)
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