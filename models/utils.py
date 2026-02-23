import numpy as np
import random
import torch
import mne
import os
import gc
import importlib
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ema_pytorch import EMA
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score,
    confusion_matrix
)

from evaluation import calculate_metrics

# Suppress MNE info messages
mne.set_log_level('WARNING')

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def zscore_norm(data):
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    elif not isinstance(data, torch.Tensor):
        raise TypeError("Input data must be a numpy array or a torch Tensor.")

    mean = torch.mean(data, dim=(1, 2), keepdim=True)
    std = torch.std(data, dim=(1, 2), keepdim=True)
    std = torch.where(std == 0, torch.ones_like(std), std)  # Prevent division by zero
    return (data - mean) / std

class OptimizedNpzDataset(Dataset):
    def __init__(self, npz_file):
        self.npz_file = npz_file
        with np.load(self.npz_file, mmap_mode='r') as data:
            self.data = data['data']
            self.labels = data['labels']
            self.file_paths = data['file_paths'].tolist()
        self.length = len(self.labels)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        file_path = self.file_paths[idx]

        x = np.nan_to_num(x, nan=0.0)
        y = np.nan_to_num(y, nan=0.0)

        x = torch.from_numpy(x).float()
        y = torch.tensor(y, dtype=torch.long)

        return x, y, file_path

def get_all_groups_dataloader(data_dir, start_group, end_group, batch_size, seed):
    datasets = []
    for group in range(start_group, end_group + 1):
        npz_file = os.path.join(data_dir, f"Group_{group}", f"group_{group}.npz")
        if os.path.exists(npz_file):
            try:
                dataset = OptimizedNpzDataset(npz_file)
                datasets.append(dataset)
            except Exception as e:
                print(f"Error loading {npz_file}: {e}")
        else:
            print(f"File not found: {npz_file}")

    if len(datasets) == 0:
        print("No datasets found!")
        return None

    combined_dataset = ConcatDataset(datasets)

    # Generator for reproducibility
    g = torch.Generator()
    g.manual_seed(seed)

    dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=8,
                            worker_init_fn=seed_worker,
                            generator=g,
                            pin_memory=True,
                            persistent_workers=False)

    return dataloader

def process_epoch_ChronoNet(epoch, args, model, optimizer, criterion, scheduler, loader, is_training):
    device = args.device
    num_classes = args.num_classes

    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0
    all_y = []
    all_y_hat = []
    all_y_hat_proba = []

    scaler = torch.cuda.amp.GradScaler()
    lr_history = []

    with tqdm(total=len(loader), desc=f"{'Training' if is_training else 'Evaluating'} Epoch {epoch+1}") as pbar:
        for batch in loader:
            if len(batch) == 3:
                x, y, _ = batch
            elif len(batch) == 2:
                x, y = batch
            else:
                raise ValueError(f"Unexpected batch size: {len(batch)}")

            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                output = model(x)
                loss = criterion(output, y)

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                lr_history.append(scheduler.get_last_lr()[0])

            total_loss += loss.item()

            y_pred_proba = F.softmax(output, dim=1).detach().cpu().numpy()
            y_pred = y_pred_proba.argmax(axis=1)
            all_y.extend(y.cpu().numpy())
            all_y_hat.extend(y_pred)
            all_y_hat_proba.extend(y_pred_proba)

            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    metrics = calculate_metrics(all_y, all_y_hat, np.array(all_y_hat_proba), num_classes=args.num_classes)
    metrics['loss'] = avg_loss

    return metrics, lr_history

def process_epoch_BDTCN(epoch, args, model, optimizer, criterion, scheduler, loader, is_training):
    device = args.device
    num_classes = args.num_classes

    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0
    all_y = []
    all_y_hat = []
    all_y_hat_proba = []

    scaler = torch.cuda.amp.GradScaler()
    lr_history = []

    with tqdm(total=len(loader), desc=f"{'Training' if is_training else 'Evaluating'} Epoch {epoch+1}") as pbar:
        for batch in loader:
            if len(batch) == 3:
                x, y, _ = batch
            elif len(batch) == 2:
                x, y = batch
            else:
                raise ValueError(f"Unexpected batch size: {len(batch)}")

            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                output = model(x)
                loss = criterion(output, y)

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                lr_history.append(scheduler.get_last_lr()[0])

            total_loss += loss.item()

            y_pred_proba = F.softmax(output, dim=1).detach().cpu().numpy()
            y_pred = y_pred_proba.argmax(axis=1)
            all_y.extend(y.cpu().numpy())
            all_y_hat.extend(y_pred)
            all_y_hat_proba.extend(y_pred_proba)

            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    metrics = calculate_metrics(all_y, all_y_hat, np.array(all_y_hat_proba), num_classes=args.num_classes)
    metrics['loss'] = avg_loss

    return metrics, lr_history

def process_epoch_default(epoch, args, models, optimizers, criterions, schedulers, fc_ema, loader, is_training):
    """Process one epoch for DDPM-based models (DiffSA-EEG and variants)."""
    ddpm, diffe = models
    optim1, optim2 = optimizers
    criterion, criterion_class = criterions
    scheduler1, scheduler2 = schedulers
    device = args.device
    num_classes = args.num_classes
    alpha = args.alpha

    if is_training:
        ddpm.train()
        diffe.train()
    else:
        ddpm.eval()
        diffe.eval()

    total_loss = 0
    all_y = []
    all_y_hat = []
    all_y_hat_proba = []

    lr_history1 = []
    lr_history2 = []

    with tqdm(total=len(loader), desc=f"{'Training' if is_training else 'Evaluating'} Epoch {epoch+1}") as pbar:
        for batch in loader:
            if len(batch) == 3:
                x, y, _ = batch
            elif len(batch) == 2:
                x, y = batch
            else:
                raise ValueError(f"Unexpected batch size: {len(batch)}")

            x, y = x.to(device), y.type(torch.LongTensor).to(device)
            y_cat = F.one_hot(y, num_classes=num_classes).type(torch.FloatTensor).to(device)

            if is_training:
                # Train DDPM
                optim1.zero_grad(set_to_none=True)
                x_hat, down, up, noise, t = ddpm(x)
                loss_ddpm = F.l1_loss(x_hat, x, reduction="none")
                loss_ddpm.mean().backward()
                optim1.step()

                # Detach DDPM outputs
                with torch.no_grad():
                    x_hat_detached = x_hat.detach()
                    down_detached = [d.detach() for d in down]
                    up_detached = [u.detach() for u in up]
                    ddpm_out = (x_hat_detached, down_detached, up_detached, t)
                    loss_ddpm_detached = loss_ddpm.detach()

                # Train Diff-E
                optim2.zero_grad(set_to_none=True)
                decoder_out, fc_out = diffe(x, ddpm_out)
                loss_gap = criterion(decoder_out, loss_ddpm_detached)
                loss_c = criterion_class(fc_out, y_cat)
                loss = loss_gap + alpha * loss_c
                loss.backward()
                optim2.step()

                scheduler1.step()
                scheduler2.step()
                fc_ema.update()

                lr_history1.append(scheduler1.get_last_lr()[0])
                lr_history2.append(scheduler2.get_last_lr()[0])
            else:
                with torch.no_grad():
                    x_hat, down, up, noise, t = ddpm(x)
                    ddpm_out = (x_hat, down, up, t)
                    decoder_out, fc_out = diffe(x, ddpm_out)
                    loss_ddpm = F.l1_loss(x_hat, x, reduction="none")
                    loss_gap = criterion(decoder_out, loss_ddpm)
                    loss_c = criterion_class(fc_out, y_cat)
                    loss = loss_gap + alpha * loss_c

            total_loss += loss.item()

            y_pred_proba = F.softmax(fc_out, dim=1).detach().cpu().numpy()
            y_pred = y_pred_proba.argmax(axis=1)
            all_y.extend(y.cpu().numpy())
            all_y_hat.extend(y_pred)
            all_y_hat_proba.extend(y_pred_proba)

            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    metrics = calculate_metrics(all_y, all_y_hat, np.array(all_y_hat_proba), num_classes=num_classes)
    metrics['loss'] = avg_loss

    return metrics, lr_history1, lr_history2

def process_epoch_EEGNet(epoch, args, model, optimizer, criterion, scheduler, loader, is_training):
    device = args.device
    num_classes = args.num_classes

    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0
    all_y = []
    all_y_hat = []
    all_y_hat_proba = []

    scaler = torch.cuda.amp.GradScaler()
    lr_history = []

    with tqdm(total=len(loader), desc=f"{'Training' if is_training else 'Evaluating'} Epoch {epoch+1}") as pbar:
        for batch in loader:
            if len(batch) == 3:
                x, y, _ = batch
            elif len(batch) == 2:
                x, y = batch
            else:
                raise ValueError(f"Unexpected batch size: {len(batch)}")

            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                output = model(x)
                loss = criterion(output, y)

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                lr_history.append(scheduler.get_last_lr()[0])

            total_loss += loss.item()

            y_pred_proba = F.softmax(output, dim=1).detach().cpu().numpy()
            y_pred = y_pred_proba.argmax(axis=1)
            all_y.extend(y.cpu().numpy())
            all_y_hat.extend(y_pred)
            all_y_hat_proba.extend(y_pred_proba)

            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    metrics = calculate_metrics(all_y, all_y_hat, np.array(all_y_hat_proba), num_classes=args.num_classes)
    metrics['loss'] = avg_loss

    return metrics, lr_history

def process_epoch_Deep4Net(epoch, args, model, optimizer, criterion, scheduler, loader, is_training):
    device = args.device
    num_classes = args.num_classes

    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0
    all_y = []
    all_y_hat = []
    all_y_hat_proba = []

    scaler = torch.cuda.amp.GradScaler()
    lr_history = []

    with tqdm(total=len(loader), desc=f"{'Training' if is_training else 'Evaluating'} Epoch {epoch+1}") as pbar:
        for batch in loader:
            if len(batch) == 3:
                x, y, _ = batch
            elif len(batch) == 2:
                x, y = batch
            else:
                raise ValueError(f"Unexpected batch size: {len(batch)}")

            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                output = model(x)
                loss = criterion(output, y)

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                lr_history.append(scheduler.get_last_lr()[0])

            total_loss += loss.item()

            y_pred_proba = F.softmax(output, dim=1).detach().cpu().numpy()
            y_pred = y_pred_proba.argmax(axis=1)
            all_y.extend(y.cpu().numpy())
            all_y_hat.extend(y_pred)
            all_y_hat_proba.extend(y_pred_proba)

            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    metrics = calculate_metrics(all_y, all_y_hat, np.array(all_y_hat_proba), num_classes=args.num_classes)
    metrics['loss'] = avg_loss

    return metrics, lr_history

def save_metrics(epoch_dir, all_epochs_metrics):
    np.save(f'{epoch_dir}/all_epochs_metrics.npy', all_epochs_metrics)
    print(f"All metrics and learning rate data saved to {epoch_dir}/all_epochs_metrics.npy")

def initialize_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        if isinstance(m, nn.Conv1d) and m.kernel_size[0] == 1:
            # Xavier initialization for 1x1 convolutions
            nn.init.xavier_normal_(m.weight)
        else:
            # He initialization (suitable for ReLU, PReLU)
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.PReLU):
        nn.init.constant_(m.weight, 0.25)  # Default initial value for PReLU

def initialize_default(args, device, channels):
    """Initialize DDPM-based models (DiffSA-EEG and Diff-E variants)."""
    model_module = importlib.import_module(f"models.models_{args.model}")
    ConditionalUNet = getattr(model_module, 'ConditionalUNet')
    DDPM = getattr(model_module, 'DDPM')
    Encoder = getattr(model_module, 'Encoder')
    Decoder = getattr(model_module, 'Decoder')
    LinearClassifier = getattr(model_module, 'LinearClassifier')
    DiffE = getattr(model_module, 'DiffE')
    base_lr, max_lr = 1e-5, 4e-3

    ddpm_dim = 128
    encoder_dim = 256
    fc_dim = 512
    n_T = 1000

    ddpm_model = ConditionalUNet(in_channels=channels, n_feat=ddpm_dim).to(device)
    ddpm_model.apply(initialize_weights)
    ddpm = DDPM(nn_model=ddpm_model, betas=(1e-6, 1e-2), n_T=n_T, device=device).to(device)

    encoder = Encoder(in_channels=channels, dim=encoder_dim).to(device)
    encoder.apply(initialize_weights)

    decoder = Decoder(in_channels=channels, n_feat=ddpm_dim, encoder_dim=encoder_dim).to(device)
    decoder.apply(initialize_weights)

    fc = LinearClassifier(encoder_dim, fc_dim, emb_dim=args.num_classes).to(device)
    fc.apply(initialize_weights)

    diffe = DiffE(encoder, decoder, fc).to(device)

    criterion = nn.SmoothL1Loss()
    criterion_class = nn.CrossEntropyLoss()

    optim1 = optim.AdamW(ddpm.parameters(), lr=base_lr, weight_decay=1e-4)
    optim2 = optim.AdamW(diffe.parameters(), lr=base_lr, weight_decay=1e-4)

    fc_ema = EMA(diffe.fc, beta=0.95, update_after_step=100, update_every=10)

    total_steps_per_epoch = len(args.train_loader)
    step_size = total_steps_per_epoch * 1
    scheduler1 = optim.lr_scheduler.CyclicLR(optimizer=optim1, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size, mode="exp_range", cycle_momentum=False, gamma=0.9998)
    scheduler2 = optim.lr_scheduler.CyclicLR(optimizer=optim2, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size, mode="exp_range", cycle_momentum=False, gamma=0.9998)

    model = (ddpm, diffe)
    optimizers = (optim1, optim2)
    criterions = (criterion, criterion_class)
    schedulers = (scheduler1, scheduler2)

    return model, optimizers, criterions, schedulers, fc_ema

def initialize_ChronoNet(args, device, channels):
    from models.models_ChronoNet import ChronoNet

    model = ChronoNet(input_channels=channels, sequence_length=250).to(device)

    criterion = nn.CrossEntropyLoss()

    base_lr, max_lr = 1e-5, 4e-3
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)

    total_steps_per_epoch = len(args.train_loader)
    step_size = total_steps_per_epoch * 1
    scheduler = optim.lr_scheduler.CyclicLR(optimizer=optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size, mode="exp_range", cycle_momentum=False, gamma=0.9998)

    return model, optimizer, criterion, scheduler

def initialize_BDTCN(args, device, channels):
    from models.models_BDTCN import TCN

    model = TCN(
        n_outputs=args.num_classes,
        n_chans=channels,
        n_times=250,
        n_filters=args.n_filters,
        n_blocks=args.n_blocks,
        kernel_size=args.kernel_size,
        drop_prob=args.dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    base_lr, max_lr = 1e-5, 4e-3
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)

    total_steps_per_epoch = len(args.train_loader)
    step_size = total_steps_per_epoch * 1
    scheduler = optim.lr_scheduler.CyclicLR(optimizer=optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size, mode="exp_range", cycle_momentum=False, gamma=0.9998)

    return model, optimizer, criterion, scheduler

def initialize_EEGNet(args, device, channels):
    from models.models_EEGNet import EEGNetv4

    model = EEGNetv4(
        n_outputs=args.num_classes,
        n_chans=channels,
        n_times=250,
        F1=args.F1,
        D=args.D,
        F2=args.F2,
        drop_prob=args.dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    base_lr, max_lr = 1e-5, 4e-3
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)

    total_steps_per_epoch = len(args.train_loader)
    step_size = total_steps_per_epoch * 1
    scheduler = optim.lr_scheduler.CyclicLR(optimizer=optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size, mode="exp_range", cycle_momentum=False, gamma=0.9998)

    return model, optimizer, criterion, scheduler

def initialize_Deep4Net(args, device, channels):
    from models.models_Deep4Net import Deep4Net

    sample_data = next(iter(args.train_loader))[0]
    input_window_samples = sample_data.shape[3]

    model = Deep4Net(
        n_chans=channels,
        n_outputs=args.num_classes,
        input_window_samples=input_window_samples,
        n_filters_time=25,
        n_filters_spat=25,
        filter_time_length=10,
        pool_time_length=3,
        pool_time_stride=3,
        n_filters_2=50,
        n_filters_3=100,
        n_filters_4=200,
        drop_prob=0.5
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0]).to(device))

    base_lr, max_lr = 1e-5, 4e-3
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)

    total_steps_per_epoch = len(args.train_loader)
    step_size = total_steps_per_epoch * 1
    scheduler = optim.lr_scheduler.CyclicLR(optimizer=optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size, mode="exp_range", cycle_momentum=False, gamma=0.9998)

    return model, optimizer, criterion, scheduler
