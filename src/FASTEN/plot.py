from .common import torch, pd, np, os
import matplotlib.pyplot as plt
import optuna

COLORS = {"logits": "C6", "total_count": "C4", "rate": "C3",
          "concentration": "C2", "loc": "C0", "scale": "C1",
          "df": "C7", "alpha": "C5", "high": "C8", "low": "C9"}

def plot_train(trainer, figure_dir):
    if not os.path.exists(figure_dir): os.mkdir(figure_dir)
    plt.figure(dpi = 200, figsize = (6, 4))
    plt.plot(range(len(trainer.train.loss)), trainer.train.loss, 
        color = "blue", alpha = 0.5, label = "Training", zorder = 2)
    max_val = max(trainer.train.loss)
    if trainer.valid: 
        plt.plot(range(len(trainer.valid.loss)), trainer.valid.loss, 
            color = "red", alpha = 0.5, label = "Validation", zorder = 2)
        max_val = max(max_val, max(trainer.valid.loss))
    if max_val > 10: plt.yscale("log")
    plt.xlabel("Epoch")
    plt.legend()
    plt.ylabel("Average Loss")
    for i in range(2, 4):
        plt.subplot(1, 3, i)
        plt.plot([], [])
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{figure_dir}/loss_curve_plot.png")
    plt.close()

def plot_tune(tuner, figure_dir):
    plt.figure(figsize = (4,4))
    rank = optuna.importance.get_param_importances(tuner.study)
    importances = np.array(list(rank.values()))
    params = np.array(list(rank.keys()))
    index = np.argsort(importances)
    plt.barh(params[index], importances[index])
    plt.xlabel("Importance")
    plt.ylabel("Hyperparameter")
    plt.tight_layout()
    plt.savefig(f"{figure_dir}/importance_plot.png", dpi = 200)
    plt.close()

    trials = tuner.study.trials_dataframe()
    trials = trials.loc[trials["state"] == "COMPLETE"].reset_index(drop = True)
    plt.figure(figsize = (3 * len(params), 3))
    for i, param in enumerate(params):
        plt.subplot(1, len(params), i+1)
        for value, group in trials.groupby(f"params_{param}"):
            x, y = np.repeat(str(value), len(group)), group["value"]
            plt.scatter(x, y, alpha = 0.5, linewidths = 0, color = "C0")
        plt.xlabel(param)
        plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(f"{figure_dir}/slices_plot.png", dpi = 200)
    plt.close()

    best = [trials.loc[:i+1, "value"].min() for i in range(len(trials))]
    plt.figure(figsize = (8, 4))
    plt.scatter(range(len(trials)), trials["value"], color = "C0")
    plt.plot(range(len(trials)), best, color = "C3")
    plt.xlabel("Trial")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(f"{figure_dir}/convergence_plot.png", dpi = 200)
    plt.close()


def plot_predict(predictor, figure_dir): 
    if predictor.true is None: return
    predictor.execute()
    mse, kld, nll = predictor.evaluate()
    if not os.path.exists(figure_dir): 
        os.mkdir(figure_dir)
    plot_statistics(predictor, figure_dir)
    plot_kl_divergence(predictor, kld, figure_dir)
    plot_mean_squares(predictor, mse, figure_dir)
    plot_samples(predictor, figure_dir)
    return mse, kld, nll

def plot_samples(predictor, figure_dir, points = 10000, n_samples = 10):
    sample_dir = f"{figure_dir}/samples"
    if not os.path.exists(sample_dir): os.mkdir(sample_dir) 
    groups = predictor.true.samples.outputs.groupby(predictor.true.samples.group)
    n_samples = min(n_samples, len(groups) // 5)
    clusters = predictor.true.stats.cluster_data(n_clusters = n_samples)
    samples = pd.Series(np.arange(len(groups))).groupby(clusters).sample()
    rows, cols = n_samples, len(predictor.model.outputs)
    true = torch.empty((cols, points, rows)), torch.empty((cols, points, rows))
    pred = torch.empty((cols, points, rows)), torch.empty((cols, points, rows))
    (x_true, y_true), (x_pred, y_pred) = true, pred
    
    for j, output in enumerate(predictor.model.outputs.values()):
        mask = [param in output.dist.params for param in predictor.model.params]
        for label, (x, y) in {"true": true, "pred": pred}.items():
            dataset = getattr(predictor, label)
            stats = dataset.stats.outputs.iloc[samples,mask]
            params = torch.from_numpy(stats.values)
            fit = output.dist.base(*params.unbind(dim = 1))
            lower = output.dist.support.get_bound("lower", output.dist.params, params, fit)
            upper = output.dist.support.get_bound("upper", output.dist.params, params, fit)
            dtype = int if output.dist.support.discrete else float
            x[j] = torch.lerp(lower, upper, torch.linspace(0, 1, points)).to(dtype).t()
            y[j] = torch.exp(fit.log_prob(x[j]))
    
    for i, sample in enumerate(samples):
        plt.figure(dpi = 200, figsize = (2.5 * cols + 0.5, 2))
        group = groups.get_group(sample)
        for j, output in enumerate(predictor.model.outputs.values()):
            plt.subplot(1, cols, j + 1)
            plt.hist(group.iloc[:,j], density = True, color = "darkgray")
            real_true, real_pred = ~torch.isinf(y_true[j,:,i]), ~torch.isinf(y_pred[j,:,i])
            mask_true, mask_pred = real_true.clone(), real_pred.clone()
            mask_true[real_true] = y_true[j,real_true,i] >= 0.01 * max(y_true[j,real_true,i])
            mask_pred[real_pred] = y_pred[j,real_pred,i] >= 0.01 * max(y_pred[j,real_pred,i])
            plt.plot(x_true[j,mask_true,i], y_true[j,mask_true,i], color = "green", 
                alpha = 1, linewidth = 2, label = f"Emprical (N = {len(group)})")
            plt.plot(x_pred[j,mask_pred,i], y_pred[j,mask_pred,i], color = "orange", 
                alpha = 1, linewidth = 2, linestyle = "dashed", label = "Predicted")
            if j == cols - 1: plt.legend(bbox_to_anchor = (1.05, 1), loc = "upper left")
            plt.xlabel(output.name)
            plt.ylabel("Density")
            plt.xticks(fontsize = 8)
            plt.yticks(fontsize = 8)
        plt.tight_layout()
        plt.savefig(f"{sample_dir}/sample_plot_{sample}.png")
        plt.close()

def plot_statistics(predictor, figure_dir):
    cols, rows = len(predictor.model.outputs), 2
    groups = predictor.true.samples.outputs.groupby(predictor.true.samples.group)
    plt.figure(figsize = (3 * cols, 3 * rows))

    for j, output in enumerate(predictor.model.outputs.values()):
        mask = [param in output.dist.params for param in predictor.model.params]
        stats = predictor.pred.stats.outputs.iloc[:,mask]
        params = torch.from_numpy(stats.values)
        fit = output.dist.base(*params.unbind(dim = 1))
        pred_means, pred_vars = fit.mean, fit.variance
        true_means = torch.zeros(groups.ngroups)
        true_vars = torch.zeros(groups.ngroups)
        for i, group in groups:
            true_means[i] = group[output.label].mean()
            true_vars[i] = group[output.label].var()
        
        plt.subplot(rows, cols, j + 1) 
        plt.scatter(true_means, pred_means, color = "C0")
        max_val = max(pred_means.max().max(), true_means.max().max())
        min_val = min(pred_means.min().min(), true_means.min().min())
        delta = (max_val - min_val) * 0.1
        max_val, min_val = max_val + delta, min_val - delta
        plt.axline((min_val, min_val), slope = 1, color = "darkgray", linestyle = "--")
        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)
        plt.ylabel(f"Predicted Mean")
        plt.xlabel(f"Empirical Mean")
        plt.title(output.name, pad = 10, fontweight = "bold")

        plt.subplot(rows, cols, cols + j + 1) 
        plt.scatter(true_vars, pred_vars, color = "C1")
        max_val = max(pred_vars.max().max(), true_vars.max().max())
        min_val = min(pred_vars.min().min(), true_vars.min().min())
        delta = (max_val - min_val) * 0.1
        max_val, min_val = max_val + delta, min_val - delta
        plt.axline((min_val, min_val), slope = 1, color = "darkgray", linestyle = "--")
        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)
        plt.ylabel(f"Predicted Variance")
        plt.xlabel(f"Empirical Variance")
    plt.tight_layout(rect = [0, 0, 0.99, 1])
    plt.savefig(f"{figure_dir}/statistics_plot.png", dpi = 300)
    plt.close()
    
def plot_kl_divergence(predictor, kld, figure_dir):
    names = [output.name for output in predictor.model.outputs.values()]
    positions = range(1, kld.shape[1] + 1)
    plt.figure(dpi = 200, figsize = (6, 6))
    bplot = plt.boxplot(kld, whis = [0, 100], patch_artist = True)
    for patch in bplot["boxes"]: patch.set_facecolor("darkgray")
    for patch in bplot["medians"]: patch.set_color("black")
    plt.ylim(-0.5, kld.max().max() * 5)
    plt.ylabel("KL Divergence")
    plt.xticks(positions, names, rotation = 45, ha = "right")
    plt.yscale("symlog")
    plt.tight_layout()
    plt.savefig(f"{figure_dir}/kl_divergence_plot.png")
    plt.close()

def plot_mean_squares(predictor, mse, figure_dir):
    names = [output.name for output in predictor.model.outputs.values()]
    colors = [COLORS[param.base] for param in predictor.model.params.values()]
    labels = [param.base for param in predictor.model.params.values()]
    positions = [0.5*i for i in range(mse.shape[1])]
    ticks = [0 for i in range(len(predictor.model.outputs))]
    for i, output in enumerate(predictor.model.outputs.values()):
        for j, param in enumerate(predictor.model.params):
            if param in output.dist.params: 
                positions[j] += 0.5*i + 1
                ticks[i] += positions[j] / len(output.dist.params)
    
    plt.figure(dpi = 200, figsize = (10, 6))
    bplot = plt.boxplot(mse, whis = [0, 100], widths = 0.5, positions = positions, patch_artist = True, label = labels)
    for patch, color in zip(bplot["boxes"], colors): patch.set_facecolor(color)
    for patch in bplot["medians"]: patch.set_color("black")
    plt.ylabel("Mean Squared Error")
    plt.xticks(ticks, names, rotation = 45, ha = "right")
    plt.yscale("log")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor = (1.02, 1), loc = "upper left")
    plt.tight_layout()
    plt.savefig(f"{figure_dir}/mean_squares_plot.png")
    plt.close()