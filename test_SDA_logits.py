from torchvision.models import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import argparse
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from data_loader import get_loader, get_loader_corrupt
from utils import accuracy, Tracker
from coral import coral
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix
from timm.loss import LabelSmoothingCrossEntropy
import random
import os
from torch.optim.lr_scheduler import _LRScheduler
from model_SDA import ResNetWithSD, MultiSDAdapter

# --- 1. Environment Setup ---
os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# --- 2. Utility Functions ---

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def distillation_loss(student_logits, teacher_logits, T):
    """
    Logits Self-Distillation (KL Divergence):
    The shallow layers learn to mimic the probability distribution
    produced by the deepest layer.
    """
    p_s = F.log_softmax(student_logits / T, dim=1)
    p_t = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(p_s, p_t, reduction='batchmean') * (T**2)

class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        return [base_lr * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0) for base_lr in self.base_lrs]

    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None: self.after_scheduler.step(None)
            else: self.after_scheduler.step(epoch - self.total_epoch)
        else: return super(GradualWarmupScheduler, self).step(epoch)

def make_scheduler(optimizer, epoch):
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch)
    return GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=epoch//10, after_scheduler=cosine_scheduler)

def plot_tsne(model, src_loader, tgt_loader, device, save_path):
    model.eval()
    src_features, tgt_features = [], []
    with torch.no_grad():
        for i, (img, _) in enumerate(src_loader):
            img = img.to(device)
            _, feats = model(img)
            feat = torch.flatten(F.adaptive_avg_pool2d(feats[3], (1, 1)), 1)
            src_features.append(feat.cpu().numpy())
            if i > 15: break
        for i, (img, _) in enumerate(tgt_loader):
            img = img.to(device)
            _, feats = model(img)
            feat = torch.flatten(F.adaptive_avg_pool2d(feats[3], (1, 1)), 1)
            tgt_features.append(feat.cpu().numpy())
            if i > 15: break

    src_features = np.concatenate(src_features, axis=0)
    tgt_features = np.concatenate(tgt_features, axis=0)
    tsne = TSNE(n_components=2, random_state=42)
    all_features = np.vstack([src_features, tgt_features])
    tsne_results = tsne.fit_transform(all_features)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(tsne_results[:len(src_features), 0], tsne_results[:len(src_features), 1], 
                label='Source (ISIC)', alpha=0.6, c='blue', s=10)
    plt.scatter(tsne_results[len(src_features):, 0], tsne_results[len(src_features):, 1], 
                label='Target (Turbid)', alpha=0.6, c='red', s=10)
    plt.legend()
    plt.title(f"t-SNE Visualization ({os.path.basename(save_path)})")
    plt.savefig(save_path)
    plt.close()

# --- 3. Core Training Logic ---
def train(model, adapter, optimizer, src_loader, tgt_loader, tracker, args, criterion, scheduler, epoch):
    model.train()
    adapter.train()
    
    # Disable all auxiliary losses during the last two epochs
    # to encourage stable final convergence.
    is_final_phase = epoch >= (args.epochs - 2)
    curr_alpha = args.lambda_sd_logits if not is_final_phase else 0.0
    curr_l_coral = args.lambda_coral if not is_final_phase else 0.0
    curr_l_feat = args.lambda_sd_feat if not is_final_phase else 0.0

    tr_cls = tracker.track('classification_loss', tracker.MovingMeanMonitor(momentum=0.99))
    tr_coral = tracker.track('CORAL_loss', tracker.MovingMeanMonitor(momentum=0.99))
    tr_sd = tracker.track('SD_loss', tracker.MovingMeanMonitor(momentum=0.99))

    min_batches = min(len(src_loader), len(tgt_loader)) if tgt_loader else len(src_loader)
    src_iter, tgt_iter = iter(src_loader), iter(tgt_loader) if tgt_loader else None

    tq = tqdm(range(min_batches), desc=f'E{epoch:03d} Training', ncols=0)
    for _ in tq:
        src_img, src_lbl = next(src_iter)
        src_img, src_lbl = src_img.to(args.device), src_lbl.to(args.device)
        
        optimizer.zero_grad()
        src_logits, src_feats = model(src_img)
        
        # [STEP 1] Source Domain Training (GT + Logits SD)
        loss_cls = criterion(src_logits[3], src_lbl) # L4 Teacher
        teacher_logits_src = src_logits[3].detach()
        for i in range(3): # L1, L2, L3 Students
            loss_cls += (1 - curr_alpha) * criterion(src_logits[i], src_lbl)
            loss_cls += curr_alpha * distillation_loss(src_logits[i], teacher_logits_src, args.temperature)

        if tgt_iter:
            tgt_img, tgt_lbl = next(tgt_iter)
            tgt_img, tgt_lbl = tgt_img.to(args.device), tgt_lbl.to(args.device)
            tgt_logits, tgt_feats = model(tgt_img)
            
            
            # [STEP 2] Target Domain Training (Ground Truth + Logits Self-Distillation)
            loss_cls += criterion(tgt_logits[3], tgt_lbl)
            teacher_logits_tgt = tgt_logits[3].detach()
            for i in range(3):
                loss_cls += (1 - curr_alpha) * criterion(tgt_logits[i], tgt_lbl)
                loss_cls += curr_alpha * distillation_loss(tgt_logits[i], teacher_logits_tgt, args.temperature)
            
            # [STEP 3] Deep CORAL (Feature Alignment)
            s_l4 = torch.flatten(F.adaptive_avg_pool2d(src_feats[3], (1, 1)), 1)
            t_l4 = torch.flatten(F.adaptive_avg_pool2d(tgt_feats[3], (1, 1)), 1)
            loss_coral = coral(s_l4, t_l4)
            
            # [STEP 4] Feature SD (Hint Loss / Layer-wise)
            loss_sd_feat = 0
            teacher_feat_tgt = t_l4.detach()
            for i in range(3):
                student_feat = adapter(i, tgt_feats[i])
                loss_sd_feat += F.mse_loss(student_feat, teacher_feat_tgt)
            
            total_loss = loss_cls + (curr_l_coral * loss_coral) + (curr_l_feat * loss_sd_feat)
            tr_coral.append(loss_coral.item())
            tr_sd.append(loss_sd_feat.item())
        else:
            total_loss = loss_cls

        total_loss.backward()
        optimizer.step()
        tr_cls.append(loss_cls.item())
        tq.set_postfix(cls=f"{tr_cls.mean.value:.3f}", coral=f"{tr_coral.mean.value if tgt_iter else 0:.3f}")
    
    scheduler.step()

# --- 4. Evaluation Function ---

def evaluate(model, loader, dataset_name, tracker, args):
    model.eval()
    corrects = {f'L{i+1}': 0 for i in range(4)}
    corrects['Ensemble'] = 0
    total = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for data, label in tqdm(loader, desc=f'Eval {dataset_name}'):
            data, label = data.to(args.device), label.to(args.device)
            logits, _ = model(data)
            
            for i in range(4):
                corrects[f'L{i+1}'] += (logits[i].argmax(1) == label).sum().item()
            
            probs = torch.stack([F.softmax(l, dim=1) for l in logits])
            ens_pred = probs.mean(0).argmax(1)
            corrects['Ensemble'] += (ens_pred == label).sum().item()
            
            all_preds.extend(ens_pred.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            total += label.size(0)

    results = {k: v/total for k, v in corrects.items()}
    print(f"\n[{dataset_name}] Ensemble OA: {results['Ensemble']:.4f}")
    return np.array(all_preds), np.array(all_labels), results

# --- 5. Experiment Execution and Main Function ---
def run_experiment(args, run_id, mode):
    seed_everything(args.seed + run_id)
    src_train_loader = get_loader(args.source, args.batch_size, train=True)
    tgt_train_loader = get_loader(args.target, args.batch_size, train=True)
    tgt_eval_loader = get_loader(args.target, args.batch_size, train=False)

    model = ResNetWithSD(num_classes=7).to(args.device)
    adapter = MultiSDAdapter().to(args.device)
    
    optimizer = optim.AdamW(list(model.parameters()) + list(adapter.parameters()), lr=args.lr)
    scheduler = make_scheduler(optimizer, args.epochs)
    tracker = Tracker()

    for epoch in range(args.epochs):
        train(model, adapter, optimizer, src_train_loader, 
              tgt_train_loader if mode == 'adaptation' else None, 
              tracker, args, LabelSmoothingCrossEntropy(), scheduler, epoch)
    
    preds, labels, accs = evaluate(model, tgt_eval_loader, f'{mode}_run{run_id}', tracker, args)
# =====================================================
# 🔥 1. Save Per-Sample Classification Results
# =====================================================
    result_df = pd.DataFrame({
        "gt": labels,
        "pred": preds
    })

    # detail_name = f"classification_{mode}_run{run_id}.csv"
    # result_df.to_csv(detail_name, index=False)
    # print(f"[✔] Saved per-sample result -> {detail_name}")

# =====================================================
# 🔥 2. Compute Additional Evaluation Metrics
# =====================================================
    report_dict = classification_report(labels, preds, output_dict=True)

    report_df = pd.DataFrame(report_dict).transpose()

    # Round the metrics
    report_df = report_df.round(4)

    report_name = f"classification_report_{mode}_logits.csv"
    report_df.to_csv(report_name)

    print(f"[✔] Saved classification report -> {report_name}")

    accs["macro_f1"] = report_dict["macro avg"]["f1-score"]
    accs["weighted_f1"] = report_dict["weighted avg"]["f1-score"]
    accs["macro_precision"] = report_dict["macro avg"]["precision"]
    accs["macro_recall"] = report_dict["macro avg"]["recall"]
    accs["kappa"] = cohen_kappa_score(labels, preds)

    save_name = f"tsne_{mode}_run{run_id}_SDA_logits.png"
    plot_tsne(model, src_train_loader, tgt_train_loader, args.device, save_name)

    return accs, preds, labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lambda_coral', type=float, default=0.1, help='Deep CORAL weight')
    parser.add_argument('--lambda_sd_logits', type=float, default=0.4, help='Alpha for Logits SD')
    parser.add_argument('--lambda_sd_feat', type=float, default=0.1, help='Lambda for Feature SD')
    parser.add_argument('--temperature', type=float, default=1.0, help='Distillation Temperature')
    parser.add_argument('--source', type=str, default='isic2018')
    parser.add_argument('--target', type=str, default='isic2018_turbidity_medium_center')
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    modes = ['source_only', 'adaptation']
    final_csv_rows = []

    for mode in modes:
        print(f"\n{'='*20}\nMode: {mode.upper()}\n{'='*20}")
        for run in range(args.runs):
            accs, preds, labels = run_experiment(args, run, mode)
            row = {'mode': mode, 'run': run + 1}
            row.update(accs) 
            final_csv_rows.append(row)

    df = pd.DataFrame(final_csv_rows)
    df.to_csv('sda_logits.csv', index=False)
    print("\n[✔] Experiment completed. Results have been saved to 'sda_logits.csv'.")

if __name__ == '__main__':
    main()
