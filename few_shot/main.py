import argparse
import copy
import os
import pickle
import random
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

import gpt
import utils


parser = argparse.ArgumentParser(description="Finite-task regression pretraining with few-shot fine-tuning eval")
parser.add_argument("--experiment_name", default="default_exp", type=str, help="grouping name for experiments")
parser.add_argument("--data", default="./data", help="unused legacy arg kept for compatibility")
parser.add_argument("--cache", default="./cache", help="unused legacy arg kept for compatibility")
parser.add_argument("--savedir", default=".", type=str, help="directory where results are written")
parser.add_argument("--fileprefix", default="regression", type=str, help="experiment name prefix")
parser.add_argument("--seed", default=0, type=int, help="random seed")
parser.add_argument("--no-cuda", action="store_true", default=False, help="disable CUDA")
parser.add_argument("-b", "--batch-size", default=64, type=int, help="pretraining batch size")
parser.add_argument("--fewshot_batch_size", default=None, type=int, help="adaptation batch size; defaults to batch-size")
parser.add_argument("--num_iters", default=50000, type=int, help="number of pretraining batches")
parser.add_argument("--eval_every", default=1000, type=int, help="evaluate every N pretraining batches")
parser.add_argument("--lr", default=1e-3, type=float, help="pretraining learning rate")
parser.add_argument("--fewshot_lr", default=5e-4, type=float, help="adaptation learning rate")
parser.add_argument("--wd", "--weight-decay", default=0.0, type=float, dest="weight_decay", help="weight decay")
parser.add_argument("--optimizer", default="Adam", choices=["Adam", "SGD"], help="optimizer")
parser.add_argument("--arch", default="gpt", choices=["gpt", "OneLayerAttention"], help="model architecture")
parser.add_argument("--len_context", default=32, type=int, help="number of examples per sequence")
parser.add_argument("--input_dim", default=16, type=int, help="dimension of Gaussian inputs")
parser.add_argument("--noise_std", default=0.1, type=float, help="target noise std")
parser.add_argument("--K", default=1000, type=int, help="number of training tasks in the finite pool")
parser.add_argument(
    "--task_distribution",
    default="zipf",
    choices=["uniform", "zipf"],
    help="rank-frequency distribution over the finite task pool",
)
parser.add_argument("--zipf_alpha", default=1.0, type=float, help="Zipf skew parameter")
parser.add_argument("--num_eval_seen_tasks", default=256, type=int, help="number of seen tasks used in capacity/interference eval")
parser.add_argument("--eval_task_stride", type=int, default=1, help="stride for evaluating seen tasks")
parser.add_argument("--num_eval_heldout_tasks", default=256, type=int, help="number of held-out tasks used in generalization eval")
parser.add_argument("--num_rank_buckets", default=5, type=int, help="number of rank buckets for seen-task summaries")
parser.add_argument("--num_hidden_features", default=32, type=int, help="per-head hidden width")
parser.add_argument("--num_layers", default=4, type=int, help="number of transformer layers")
parser.add_argument("--num_heads", default=4, type=int, help="number of attention heads")
parser.add_argument("--dropout", default=0.0, type=float, help="dropout")
parser.add_argument("--gpt_bias", default="True", type=str, help="whether linears/layer norms use bias")
parser.add_argument("--num_eval_batches", default=4, type=int, help="number of eval batches for aggregate MSE")
parser.add_argument(
    "--fewshot_source",
    default="heldout",
    choices=["heldout", "seen"],
    help="where fine-tuning tasks come from",
)
parser.add_argument("--fewshot_num_batches", default=50, type=int, help="number of adaptation batches")
parser.add_argument("--fewshot_mix_probability", default=12.0/256.0, type=float, help="probability that an item in adaptation batch comes from the fine-tuning stream")
parser.add_argument("--num_finetune_tasks", default=50, type=int, help="number of fixed fine-tuning tasks")
parser.add_argument("--steps_per_epoch", default=1000, type=int, help="number of pretraining gradient steps per epoch")
parser.add_argument("--fewshot_mse_threshold", default=0.05, type=float, help="few-shot success threshold on fine-tuning eval MSE")
parser.add_argument("--fewshot_eval_num_tasks", default=64, type=int, help="number of fixed tasks used in fine-tuning evaluation")
parser.add_argument("--fewshot_eval_batches", default=2, type=int, help="number of evaluation batches per fine-tuning checkpoint")


if utils.is_interactive():
    jupyter_args = (
        "--savedir ./cache --experiment_name debug_group --fileprefix debug_regression --seed 0 --batch-size 32 "
        "--num_iters 200 --eval_every 50 --lr 1e-3 --len_context 16 --input_dim 8 --K 128 "
        "--task_distribution zipf --zipf_alpha 1.0 --num_eval_seen_tasks 64 --num_eval_heldout_tasks 64 "
        "--num_hidden_features 8 --num_layers 2 --num_heads 2 --fewshot_num_batches 10 "
        "--fewshot_mix_probability 0.5 --fewshot_eval_num_tasks 16"
    ).split()
    args = parser.parse_args(jupyter_args)
else:
    args = parser.parse_args()

if args.experiment_name in ["gpt", "gpt_K_100000"]:
    args.arch = "gpt"
elif args.experiment_name in ["OneLayer", "OneLayer_K_100000"]:
    args.arch = "OneLayerAttention"

def parse_bool(value):
    return str(value).lower() in {"1", "true", "yes", "y"}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def bucketize_indices(num_items, num_buckets):
    num_buckets = max(1, min(num_buckets, num_items))
    splits = np.array_split(np.arange(num_items), num_buckets)
    return [bucket.astype(int) for bucket in splits if len(bucket) > 0]


class FiniteTaskPool:
    def __init__(self, num_tasks, input_dim, noise_std, distribution, zipf_alpha, seed):
        self.num_tasks = num_tasks
        self.input_dim = input_dim
        self.noise_std = noise_std
        self.distribution = distribution
        self.zipf_alpha = zipf_alpha
        self.generator = torch.Generator().manual_seed(seed)
        self.task_weights = torch.randn(num_tasks, input_dim, generator=self.generator)
        self.task_bias = torch.randn(num_tasks, 1, generator=self.generator)
        self.probabilities = self._build_probabilities()

    def _build_probabilities(self):
        if self.distribution == "uniform":
            return torch.full((self.num_tasks,), 1.0 / self.num_tasks)
        ranks = torch.arange(1, self.num_tasks + 1, dtype=torch.float32)
        probs = ranks.pow(-self.zipf_alpha)
        return probs / probs.sum()

    def sample_task_ids(self, batch_size):
        return torch.multinomial(self.probabilities, batch_size, replacement=True, generator=self.generator)

    def sample_batch(self, batch_size, len_context, device, task_ids=None):
        if task_ids is None:
            task_ids = self.sample_task_ids(batch_size)
        else:
            task_ids = task_ids.to(dtype=torch.long)
        xs = torch.randn(batch_size, len_context, self.input_dim, generator=self.generator)
        weights = self.task_weights[task_ids]
        bias = self.task_bias[task_ids].unsqueeze(1)
        ys = (xs * weights.unsqueeze(1)).sum(dim=-1, keepdim=True) + bias
        noise = self.noise_std * torch.randn(batch_size, len_context, 1, generator=self.generator)
        ys = ys + noise
        return xs.to(device), ys.to(device), task_ids.to(device)


class HeldoutTaskSampler:
    def __init__(self, input_dim, noise_std, num_tasks, seed):
        self.input_dim = input_dim
        self.noise_std = noise_std
        self.num_tasks = num_tasks
        self.generator = torch.Generator().manual_seed(seed)
        self.task_weights = torch.randn(num_tasks, input_dim, generator=self.generator)
        self.task_bias = torch.randn(num_tasks, 1, generator=self.generator)

    def sample_batch(self, batch_size, len_context, device, task_ids=None):
        if task_ids is None:
            task_ids = torch.randint(self.num_tasks, (batch_size,), generator=self.generator)
        else:
            task_ids = task_ids.to(dtype=torch.long)
        xs = torch.randn(batch_size, len_context, self.input_dim, generator=self.generator)
        weights = self.task_weights[task_ids]
        bias = self.task_bias[task_ids].unsqueeze(1)
        ys = (xs * weights.unsqueeze(1)).sum(dim=-1, keepdim=True) + bias
        noise = self.noise_std * torch.randn(batch_size, len_context, 1, generator=self.generator)
        ys = ys + noise
        return xs.to(device), ys.to(device), task_ids.to(device)


@dataclass
class FixedEvalSet:
    xs: torch.Tensor
    ys: torch.Tensor
    task_ids: torch.Tensor
    task_ranks: Optional[torch.Tensor] = None


def make_seen_eval_set(task_pool, num_tasks, len_context, device, stride=1):
    num_tasks = min(num_tasks, task_pool.num_tasks)
    if task_pool.distribution == "zipf":
        if stride > 1:
            task_ids = torch.arange(0, num_tasks, stride, dtype=torch.long)
        else:
            weights = task_pool.probabilities.cpu().numpy()
            sampled = np.random.choice(task_pool.num_tasks, size=num_tasks, replace=False, p=weights / weights.sum())
            task_ids = torch.as_tensor(np.sort(sampled), dtype=torch.long)
    else:
        task_ids = torch.arange(0, num_tasks, stride, dtype=torch.long)
    xs, ys, task_ids = task_pool.sample_batch(len(task_ids), len_context, device, task_ids=task_ids)
    return FixedEvalSet(xs=xs, ys=ys, task_ids=task_ids, task_ranks=task_ids.detach().cpu())


def make_heldout_eval_set(task_sampler, num_tasks, len_context, device):
    task_ids = torch.arange(min(num_tasks, task_sampler.num_tasks), dtype=torch.long)
    xs, ys, task_ids = task_sampler.sample_batch(len(task_ids), len_context, device, task_ids=task_ids)
    return FixedEvalSet(xs=xs, ys=ys, task_ids=task_ids)


def iter_eval_batches(eval_set, batch_size):
    total = eval_set.xs.shape[0]
    for start in range(0, total, batch_size):
        end = min(total, start + batch_size)
        yield eval_set.xs[start:end], eval_set.ys[start:end], eval_set.task_ids[start:end]


def compute_sequence_mse(predictions, targets):
    return ((predictions - targets) ** 2).mean(dim=(1, 2))


@torch.no_grad()
def evaluate_fixed_set(model, eval_set, batch_size, num_rank_buckets):
    model.eval()
    task_ids = []
    per_task_mse = []
    for xs, ys, batch_task_ids in iter_eval_batches(eval_set, batch_size):
        predictions = model(xs, ys)
        task_ids.append(batch_task_ids.detach().cpu())
        per_task_mse.append(compute_sequence_mse(predictions, ys).detach().cpu())
    task_ids = torch.cat(task_ids).numpy()
    per_task_mse = torch.cat(per_task_mse).numpy()
    metrics = {
        "overall_mse": float(per_task_mse.mean()),
        "per_task_mse": per_task_mse,
        "task_ids": task_ids,
    }
    if eval_set.task_ranks is not None:
        ranks = eval_set.task_ranks.numpy()
        bucket_metrics = []
        for bucket in bucketize_indices(len(ranks), num_rank_buckets):
            bucket_metrics.append(
                {
                    "rank_start": int(ranks[bucket[0]]) + 1,
                    "rank_end": int(ranks[bucket[-1]]) + 1,
                    "mse": float(per_task_mse[bucket].mean()),
                }
            )
        metrics["rank_buckets"] = bucket_metrics
    return metrics


def build_model(args, device):
    bias = parse_bool(args.gpt_bias)
    if args.arch == "gpt":
        config = gpt.GPTConfig(
            block_size=args.len_context,
            input_dim=args.input_dim,
            output_dim=1,
            n_layer=args.num_layers,
            n_head=args.num_heads,
            n_embd=args.num_heads * args.num_hidden_features,
            dropout=args.dropout,
            bias=bias,
        )
        return gpt.GPT(config).to(device)
    if args.arch == "OneLayerAttention":
        return gpt.OneLayerAttention(
            len_context=args.len_context,
            input_dim=args.input_dim,
            output_dim=1,
            num_heads=args.num_heads,
            num_hidden_features=args.num_hidden_features,
            num_mlp_layers=args.num_layers,
            dropout=args.dropout,
            bias=bias,
        ).to(device)
    raise ValueError(f"Unsupported architecture: {args.arch}")


def build_optimizer(args, model, lr=None):
    lr = args.fewshot_lr if lr is None else lr
    if args.optimizer == "SGD":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)


def pretrain_step(model, optimizer, xs, ys):
    optimizer.zero_grad()
    predictions = model(xs, ys)
    loss = F.mse_loss(predictions, ys)
    loss.backward()
    optimizer.step()
    return float(loss.item())


def evaluate_pretraining(model, seen_eval_set, heldout_eval_set, batch_size, num_rank_buckets):
    seen_metrics = evaluate_fixed_set(model, seen_eval_set, batch_size, num_rank_buckets)
    heldout_metrics = evaluate_fixed_set(model, heldout_eval_set, batch_size, num_rank_buckets)
    heldout_metrics["generalization_gap"] = heldout_metrics["overall_mse"] - seen_metrics["overall_mse"]
    return seen_metrics, heldout_metrics


def run_fewshot_adaptation(args, device, base_model, train_pool, finetune_task_pool):
    adapted_model = copy.deepcopy(base_model).to(device)
    optimizer = build_optimizer(args, adapted_model, lr=args.fewshot_lr)
    
    num_tasks = finetune_task_pool.num_tasks
    loss_trace = np.full((num_tasks, args.fewshot_num_batches), np.nan)
    
    for batch_idx in range(args.fewshot_num_batches):
        adapted_model.train()
        
        num_finetune = np.random.binomial(args.fewshot_batch_size, args.fewshot_mix_probability)
        num_pretrain = args.fewshot_batch_size - num_finetune
        
        xs_list, ys_list = [], []
        tids_f = None
        
        if num_pretrain > 0:
            xs_p, ys_p, _ = train_pool.sample_batch(num_pretrain, args.len_context, device)
            xs_list.append(xs_p)
            ys_list.append(ys_p)
            
        if num_finetune > 0:
            xs_f, ys_f, tids_f = finetune_task_pool.sample_batch(num_finetune, args.len_context, device)
            xs_list.append(xs_f)
            ys_list.append(ys_f)
            
        xs = torch.cat(xs_list, dim=0)
        ys = torch.cat(ys_list, dim=0)
        
        optimizer.zero_grad()
        predictions = adapted_model(xs, ys)
        
        # Calculate loss per sequence: shape (batch_size,)
        per_seq_loss = ((predictions - ys) ** 2).mean(dim=(1, 2))
        loss = per_seq_loss.mean()
        loss.backward()
        optimizer.step()
        
        if num_finetune > 0:
            ft_losses = per_seq_loss[-num_finetune:].detach().cpu().numpy()
            tids = tids_f.cpu().numpy()
            for t_idx in np.unique(tids):
                mask = (tids == t_idx)
                loss_trace[t_idx, batch_idx] = ft_losses[mask].mean()
                
    return {
        "loss_trace": loss_trace
    }


def main():
    fewshot_batch_size = args.fewshot_batch_size or args.batch_size
    args.fewshot_batch_size = fewshot_batch_size

    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_pool = FiniteTaskPool(
        num_tasks=args.K,
        input_dim=args.input_dim,
        noise_std=args.noise_std,
        distribution=args.task_distribution,
        zipf_alpha=args.zipf_alpha,
        seed=args.seed,
    )
    seen_eval_set = make_seen_eval_set(train_pool, args.num_eval_seen_tasks, args.len_context, device, stride=args.eval_task_stride)
    heldout_eval_sampler = HeldoutTaskSampler(
        input_dim=args.input_dim,
        noise_std=args.noise_std,
        num_tasks=max(args.num_eval_heldout_tasks, args.fewshot_eval_num_tasks),
        seed=args.seed + 1,
    )
    heldout_eval_set = make_heldout_eval_set(heldout_eval_sampler, args.num_eval_heldout_tasks, args.len_context, device)

    finetune_task_pool = HeldoutTaskSampler(
        input_dim=args.input_dim,
        noise_std=args.noise_std,
        num_tasks=args.num_finetune_tasks,
        seed=args.seed + 100,
    )

    model = build_model(args, device)
    num_parameters = model.get_num_params()
    optimizer = build_optimizer(args, model, lr=args.lr)
    train_loss_meter = utils.AverageMeter("train_mse", ":.4e")

    experiment_group_dir = os.path.join(args.savedir, "cache", args.experiment_name)
    os.makedirs(experiment_group_dir, exist_ok=True)
    
    timestamp = int(time.time())
    fileprefix = f"{args.experiment_name}_{args.fileprefix}_K_{args.K}_L_{args.len_context}_H_{args.num_heads}_D_{args.num_layers}_{timestamp}"
    dirprefix = os.path.join(experiment_group_dir, fileprefix)
    os.makedirs(dirprefix, exist_ok=True)
    
    exp_path = os.path.join(dirprefix, f"{fileprefix}.pkl")
    record = {
        "args": vars(args),
        "num_parameters": num_parameters,
        "pretraining_logs": [],
        "capacity_metrics": [],
        "generalization_metrics": [],
        "fewshot_adaptation": [],
    }

    print("device", device)
    print("num_parameters", num_parameters)
    print("saving_to", exp_path)

    for step in range(args.num_iters):
        model.train()
        xs, ys, _ = train_pool.sample_batch(args.batch_size, args.len_context, device)
        train_loss = pretrain_step(model, optimizer, xs, ys)
        train_loss_meter.update(train_loss, n=xs.shape[0])

        should_eval = step == 0 or (step + 1) % args.eval_every == 0 or step == args.num_iters - 1
        if should_eval:
            seen_metrics, heldout_metrics = evaluate_pretraining(
                model,
                seen_eval_set,
                heldout_eval_set,
                args.batch_size,
                args.num_rank_buckets,
            )
            record["pretraining_logs"].append(
                {
                    "step": step,
                    "train_mse": train_loss_meter.avg,
                    "seen_mse": seen_metrics["overall_mse"],
                    "heldout_mse": heldout_metrics["overall_mse"],
                    "generalization_gap": heldout_metrics["generalization_gap"],
                }
            )
            record["capacity_metrics"].append({"step": step, **seen_metrics})
            record["generalization_metrics"].append({"step": step, **heldout_metrics})
            print(
                f"step {step:6d} train_mse {train_loss_meter.avg:.6f} "
                f"seen_mse {seen_metrics['overall_mse']:.6f} "
                f"heldout_mse {heldout_metrics['overall_mse']:.6f}"
            )
            with open(exp_path, "wb") as handle:
                pickle.dump(record, handle)

        if (step + 1) % args.steps_per_epoch == 0:
            fewshot_res = run_fewshot_adaptation(
                args,
                device,
                model,
                train_pool,
                finetune_task_pool,
            )
            record["fewshot_adaptation"].append({
                "epoch": (step + 1) // args.steps_per_epoch,
                "step": step,
                **fewshot_res
            })
            with open(exp_path, "wb") as handle:
                pickle.dump(record, handle)

    print("finished")


if __name__ == "__main__":
    main()
