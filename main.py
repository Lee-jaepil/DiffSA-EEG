import argparse
import importlib
import os
import random
import numpy as np
import torch
import gc
from tqdm.auto import tqdm

from models.utils import (
    get_all_groups_dataloader,
    seed_worker,
    process_epoch_EEGNet,
    process_epoch_ChronoNet,
    process_epoch_BDTCN,
    process_epoch_default,
    process_epoch_Deep4Net,
    save_metrics,
    initialize_default,
    initialize_EEGNet,
    initialize_ChronoNet,
    initialize_BDTCN,
    initialize_Deep4Net,
)

from evaluation import load_evaluation_dataset

import warnings
warnings.filterwarnings("ignore", message="enable_nested_tensor is True")

import json
from datetime import datetime

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set as {seed}")

def run_experiment(args, run_idx):
    print(f"\nStarting experiment run {run_idx + 1}/{args.num_runs}")
    device = torch.device(args.device)
    set_seed(args.seed + run_idx)  # Use different seed for each run

    # Create per-run directory
    run_dir = f'{args.model_save_dir}/epoch_{args.num_epochs}_{args.dataset}_{args.model}/run_{run_idx + 1}'
    os.makedirs(run_dir, exist_ok=True)

    # Load datasets
    train_loader = get_all_groups_dataloader(args.data_dir, args.start_group, args.end_group, args.batch_size, args.seed + run_idx)
    if train_loader is None:
        print("Failed to create DataLoader. Exiting...")
        return None

    eval_data, eval_labels, file_paths = load_evaluation_dataset(args.eval_data_path)
    eval_dataset = torch.utils.data.TensorDataset(eval_data, eval_labels)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model based on the selected model type
    channels = 20 if args.dataset in ["TCP", "TCPx3"] else 21
    args.train_loader = train_loader

    if args.model == "EEGNet":
        model, optimizer, criterion, scheduler = initialize_EEGNet(args, device, channels)
        process_epoch = process_epoch_EEGNet
    elif args.model == "ChronoNet":
        model, optimizer, criterion, scheduler = initialize_ChronoNet(args, device, channels)
        process_epoch = process_epoch_ChronoNet
    elif args.model == "BDTCN":
        model, optimizer, criterion, scheduler = initialize_BDTCN(args, device, channels)
        process_epoch = process_epoch_BDTCN
    elif args.model == "Deep4Net":
        model, optimizer, criterion, scheduler = initialize_Deep4Net(args, device, channels)
        process_epoch = process_epoch_Deep4Net
    else:
        # For DiffSA-EEG models (base, SSDA_attn_SF_CBAM, etc.)
        model, optimizers, criterions, schedulers, fc_ema = initialize_default(args, device, channels)
        process_epoch = process_epoch_default

    best_metrics = {"accuracy": 0, "f1": 0, "recall": 0, "precision": 0, "auc": 0, "specificity": 0, "loss": float('inf')}
    all_epochs_metrics = []

    for epoch in range(args.num_epochs):
        if args.model in ["EEGNet", "ChronoNet", "BDTCN", "Deep4Net"]:
            train_metrics, train_lr = process_epoch(epoch, args, model, optimizer, criterion, scheduler, train_loader, is_training=True)
            eval_metrics, _ = process_epoch(epoch, args, model, optimizer, criterion, scheduler, eval_loader, is_training=False)
            lr = train_lr[-1] if isinstance(train_lr, list) else train_lr
        else:
            train_metrics, train_lr1, train_lr2 = process_epoch(epoch, args, model, optimizers, criterions, schedulers, fc_ema, train_loader, is_training=True)
            eval_metrics, _, _ = process_epoch(epoch, args, model, optimizers, criterions, schedulers, fc_ema, eval_loader, is_training=False)
            lr = [train_lr1[-1], train_lr2[-1]]

        print(f"\nEpoch {epoch+1}/{args.num_epochs}:")
        print(f"  Train - Accuracy: {train_metrics['accuracy']:.4f}, Recall: {train_metrics['recall']:.4f}, Specificity: {train_metrics['specificity']:.4f}, Loss: {train_metrics['loss']:.4f}")
        print(f"  Eval  - Accuracy: {eval_metrics['accuracy']:.4f}, Recall: {eval_metrics['recall']:.4f}, Specificity: {eval_metrics['specificity']:.4f}")

        # Save current epoch metrics
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_metrics['loss'],
            "train_accuracy": train_metrics['accuracy'],
            "train_f1": train_metrics['f1'],
            "train_recall": train_metrics['recall'],
            "train_precision": train_metrics['precision'],
            "train_auc": train_metrics['auc'],
            "train_specificity": train_metrics['specificity'],
            "eval_loss": eval_metrics['loss'],
            "eval_accuracy": eval_metrics['accuracy'],
            "eval_f1": eval_metrics['f1'],
            "eval_recall": eval_metrics['recall'],
            "eval_precision": eval_metrics['precision'],
            "eval_auc": eval_metrics['auc'],
            "eval_specificity": eval_metrics['specificity'],
            "lr": lr
        }
        all_epochs_metrics.append(epoch_metrics)

        # Update best metrics and save best model for this run
        if eval_metrics['accuracy'] > best_metrics['accuracy']:
            best_metrics = eval_metrics.copy()
            if args.model not in ["ChronoNet", "BDTCN", "EEGNet", "Deep4Net"]:
                torch.save(model[1].state_dict(), f'{run_dir}/best_model.pth')
                torch.save(fc_ema.state_dict(), f'{run_dir}/best_model_ema.pth')
            else:
                torch.save(model.state_dict(), f'{run_dir}/best_model.pth')

        # Clear memory
        torch.cuda.empty_cache()
        gc.collect()

    # Save metrics for this run
    with open(f'{run_dir}/metrics.json', 'w') as f:
        json.dump({
            'best_metrics': best_metrics,
            'all_epochs_metrics': all_epochs_metrics
        }, f, indent=4)

    # Also save as .npy for backward compatibility
    np.save(f'{run_dir}/all_epochs_metrics.npy', all_epochs_metrics)
    print(f"All metrics and learning rate data saved to {run_dir}/all_epochs_metrics.npy")

    return best_metrics

def aggregate_results(args):
    # Directory containing all experiment results
    base_dir = f'{args.model_save_dir}/epoch_{args.num_epochs}_{args.dataset}_{args.model}'

    # Collect results
    all_results = []
    metric_names = ['accuracy', 'f1', 'recall', 'precision', 'auc', 'specificity', 'loss']

    for run_idx in range(args.num_runs):
        with open(f'{base_dir}/run_{run_idx + 1}/metrics.json', 'r') as f:
            results = json.load(f)
            all_results.append(results['best_metrics'])

    # Compute statistics (convert NumPy types to Python native types)
    stats = {metric: {
        'mean': float(np.mean([run[metric] for run in all_results])),
        'std': float(np.std([run[metric] for run in all_results])),
        'best': float(max([run[metric] for run in all_results])) if metric != 'loss' else float(min([run[metric] for run in all_results]))
    } for metric in metric_names}

    # Find best performing run
    best_run_idx = int(np.argmax([run['accuracy'] for run in all_results]))

    # Save results
    stats['best_run'] = best_run_idx + 1
    with open(f'{base_dir}/aggregate_results.json', 'w') as f:
        json.dump(stats, f, indent=4)

    # Copy best model
    best_run_dir = f'{base_dir}/run_{best_run_idx + 1}'
    if args.model not in ["ChronoNet", "BDTCN", "EEGNet", "Deep4Net"]:
        os.system(f'cp {best_run_dir}/best_model.pth {base_dir}/best_overall_model.pth')
        os.system(f'cp {best_run_dir}/best_model_ema.pth {base_dir}/best_overall_model_ema.pth')
    else:
        os.system(f'cp {best_run_dir}/best_model.pth {base_dir}/best_overall_model.pth')

    return stats

def main(args):
    print("Starting multiple runs training")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.model_save_dir = os.path.join(args.model_save_dir, timestamp)
    os.makedirs(args.model_save_dir, exist_ok=True)

    # Run each experiment
    all_best_metrics = []
    for run_idx in range(args.num_runs):
        best_metrics = run_experiment(args, run_idx)
        if best_metrics is not None:
            all_best_metrics.append(best_metrics)

    # Aggregate results
    if all_best_metrics:
        stats = aggregate_results(args)

        # Print results
        print("\nAggregate Results:")
        for metric in stats:
            if metric != 'best_run':
                print(f"\n{metric.upper()}:")
                print(f"  Mean: {stats[metric]['mean']:.4f}")
                print(f"  Std:  {stats[metric]['std']:.4f}")
                print(f"  Best: {stats[metric]['best']:.4f}")
        print(f"\nBest Run: {stats['best_run']}")
    else:
        print("No successful runs completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate DiffSA-EEG with multiple runs")

    # Basic arguments
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (default: cuda:0)")
    parser.add_argument("--dataset", type=str, choices=["RAW", "TRA", "TRAx3", "TCP", "TCPx3", "TUSZ", "TUEP", "TUEP_NoSMOTE"], default="TRA", help="Dataset to use (default: TRA)")
    parser.add_argument("--model", type=str, default="SSDA_attn_SF_CBAM", help="Model to use (e.g., base, SSDA_attn_SF_CBAM, EEGNet, ChronoNet, BDTCN, Deep4Net)")
    parser.add_argument("--model_save_dir", type=str, default="./model_result", help="Directory to save the model")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes")
    parser.add_argument("--alpha", type=float, default=0.1, help="Weight for classification loss")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--start_group", type=int, default=1, help="Start group number")
    parser.add_argument("--end_group", type=int, default=9, help="End group number")

    # Multi-run arguments
    parser.add_argument("--num_runs", type=int, default=10, help="Number of runs to perform")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")

    # BDTCN specific arguments
    parser.add_argument("--n_filters", type=int, default=30, help="Number of filters in TCN")
    parser.add_argument("--n_blocks", type=int, default=3, help="Number of temporal blocks in TCN")
    parser.add_argument("--kernel_size", type=int, default=5, help="Kernel size in TCN")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout probability in TCN")

    # EEGNet specific arguments
    parser.add_argument('--F1', type=int, default=8, help='Number of temporal filters')
    parser.add_argument('--D', type=int, default=2, help='Depth multiplier')
    parser.add_argument('--F2', type=int, default=16, help='Number of pointwise filters')

    args = parser.parse_args()

    # Dynamically set data paths
    data_root = os.environ.get("DATA_ROOT", "./Preprocessed_Data")
    args.data_dir = os.path.join(data_root, args.dataset)
    args.eval_data_path = os.path.join(data_root, f"{args.dataset}_Evaluation", "evaluation_dataset.npz")

    main(args)
