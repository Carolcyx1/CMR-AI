import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from monai.networks.nets import (
    DenseNet, SEResNet50, EfficientNetBN, 
    ViT, UNet, SwinUNETR, SegResNet
)
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from module import Attention, PreNorm, FeedForward
from pathlib import Path
import numpy as np
import re

import time
import math
from datetime import datetime, timedelta

# 新增导入
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# ==================== 数据加载器 ====================
class CardiacThreeClassDataset(Dataset):
    def __init__(self, root_dir, num_frames=25, min_views=2, image_size=256):
        self.root_dir = Path(root_dir)
        self.num_frames = num_frames
        self.min_views = min_views
        self.image_size = image_size
        
        self.classes = {'NC': 0, 'HCM': 1, 'DCM': 2}
        self.class_names = ['NC', 'HCM', 'DCM']
        self.view_types = ['Cine2CH', 'Cine3CH', 'Cine4CH', 'CineSAX']
        
        self.samples = self._collect_samples()
    
    def _parse_filename(self, filename):
        pattern = r'^(Cine(?:2CH|3CH|4CH|SAX))-(\d+)_(\d+)_'
        match = re.match(pattern, filename.stem)
        if match:
            return match.group(1), int(match.group(2)), int(match.group(3))
        return None, None, None
    
    def _collect_samples(self):
        samples = []
        print("开始加载数据...")
        
        for class_name, class_idx in self.classes.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                print(f"警告: 类别目录 {class_dir} 不存在")
                continue
                
            patient_dirs = [d for d in class_dir.iterdir() if d.is_dir()]
            valid_count = 0
            
            for patient_dir in patient_dirs:
                patient_data = self._organize_patient_files(patient_dir)
                is_valid, reason = self._validate_patient_data(patient_data)
                
                if is_valid:
                    samples.append({
                        'patient_id': patient_dir.name,
                        'label': class_idx,
                        'class_name': class_name,
                        'organized_data': patient_data,
                        'available_views': list(patient_data.keys())
                    })
                    valid_count += 1
                    print(f"✅ {patient_dir.name}: {reason}")
                else:
                    print(f"❌ {patient_dir.name}: {reason}")
            
            print(f"{class_name}: {valid_count}/{len(patient_dirs)} 有效")
        
        print(f"数据加载完成! 总共 {len(samples)} 个有效样本")
        return samples
    
    def _organize_patient_files(self, patient_dir):
        organized = {}
        dcm_files = list(patient_dir.glob("*.dcm"))
        
        for dcm_file in dcm_files:
            view_type, slice_idx, frame_idx = self._parse_filename(dcm_file)
            
            if view_type and slice_idx is not None and frame_idx is not None:
                if view_type not in organized:
                    organized[view_type] = {}
                if slice_idx not in organized[view_type]:
                    organized[view_type][slice_idx] = []
                
                organized[view_type][slice_idx].append({
                    'frame_idx': frame_idx,
                    'file_path': str(dcm_file)
                })
        
        # 排序
        for view_type in organized:
            for slice_idx in organized[view_type]:
                organized[view_type][slice_idx].sort(key=lambda x: x['frame_idx'])
        
        return organized
    
    def _validate_patient_data(self, patient_data):
        available_views = [view for view in self.view_types 
                          if view in patient_data and patient_data[view]]
        
        if len(available_views) < self.min_views:
            return False, f"只有{len(available_views)}个视图"
        
        for view_type in available_views:
            slices = patient_data[view_type]
            for slice_idx, frames in slices.items():
                if len(frames) < self.num_frames:
                    return False, f"{view_type}-层{slice_idx}只有{len(frames)}帧"
        
        return True, f"有{len(available_views)}个完整视图"
    
    def load_dicom_data(self, file_path):
        try:
            import pydicom
            ds = pydicom.dcmread(file_path)
            image_data = ds.pixel_array.astype(np.float32)
            return image_data
        except Exception as e:
            print(f"加载失败: {file_path} - {e}")
            return np.zeros((self.image_size, self.image_size), dtype=np.float32)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        organized_data = sample['organized_data']
        
        all_view_data = []
        used_view_types = []
        
        for view_type in sample['available_views']:
            view_slices_data = []
            slices_dict = organized_data[view_type]
            
            for slice_idx in sorted(slices_dict.keys()):
                frames_info = slices_dict[slice_idx][:self.num_frames]
                slice_frames = []
                
                for frame_info in frames_info:
                    image_data = self.load_dicom_data(frame_info['file_path'])
                    image_tensor = torch.tensor(image_data).unsqueeze(0)
                    
                    if image_tensor.shape[-2:] != (self.image_size, self.image_size):
                        image_tensor = torch.nn.functional.interpolate(
                            image_tensor.unsqueeze(0), 
                            size=(self.image_size, self.image_size), 
                            mode='bilinear'
                        ).squeeze(0)
                    
                    if image_tensor.max() > 0:
                        image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())
                    
                    slice_frames.append(image_tensor)
                
                slice_sequence = torch.stack(slice_frames)
                view_slices_data.append(slice_sequence)
            
            if view_slices_data:
                view_tensor = torch.stack(view_slices_data)
                print(view_tensor.shape)
                all_view_data.append(view_tensor)
                used_view_types.append(view_type)
        
        return {
            'image': all_view_data,
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'patient_id': sample['patient_id'],
            'class_name': sample['class_name'],
            'available_views': used_view_types
        }

# ==================== Collate函数 ====================
def cardiac_collate_fn(batch, max_views=4, image_size=320, num_frames=25):
    images_by_view = [[] for _ in range(max_views)]
    labels = torch.stack([item['label'] for item in batch])
    patient_ids = [item['patient_id'] for item in batch]
    available_views_list = [item['available_views'] for item in batch]
    
    for item in batch:
        for view_idx in range(max_views):
            if view_idx < len(item['image']):
                images_by_view[view_idx].append(item['image'][view_idx])
            else:
                images_by_view[view_idx].append(torch.zeros(0, 1,num_frames, image_size, image_size))
    
    return {
        'image': images_by_view,
        'label': labels,
        'patient_id': patient_ids,
        'available_views': available_views_list
    }

# ==================== 模型定义 ====================
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class ViViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, num_frames, dim=192, depth=4, heads=3, pool='cls', 
                 in_channels=3, dim_head=64, dropout=0., emb_dropout=0., scale_dim=4):
        super().__init__()
        
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim),
        )
        self.dim = dim

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, num_patches + 1, dim))
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        # print("patch shape:", x.shape)
        x = self.to_patch_embedding(x)
        b, t, n, _ = x.shape



        # 空间Transformer处理
        cls_space_tokens = repeat(self.space_token, '() n d -> b t n d', b=b, t=t)
        x = torch.cat((cls_space_tokens, x), dim=2)
        x += self.pos_embedding[:, :, :(n + 1)]
        x = self.dropout(x)

        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.space_transformer(x)
        x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)

        # 时间Transformer处理  
        cls_temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_temporal_tokens, x), dim=1)
        x = self.temporal_transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        return self.mlp_head(x)

class CardiacViViT(nn.Module):
    def __init__(self, num_classes=3, image_size=128, num_frames=25, 
                 dim=192, depth=4, heads=3, patch_size=16):
        super().__init__()
        
        self.vivit_encoder = ViViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=dim,
            num_frames=num_frames,
            dim=dim,
            depth=depth,
            heads=heads,
            pool='cls',
            in_channels=1,
            dropout=0.1
        )
        
        self.vivit_encoder.mlp_head = nn.Identity()
        
        self.view_fusion_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=dim*4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )
        
        self.view_cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(0.3),
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        print(f"真正ViViT配置: 图像{image_size}, 帧数{num_frames}, 维度{dim}")
        
    def encode_view(self, view_tensor, device):
        slice_features = []
        max_slices = 12
        num_slices = min(max_slices, view_tensor.shape[0])
        
        for slice_idx in range(num_slices):
            try:
                slice_data = view_tensor[slice_idx].unsqueeze(0).to(device)
                # print("slice_data:", slice_data.shape)
                slice_feature = self.vivit_encoder(slice_data)
                slice_features.append(slice_feature)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"ViViT处理slice {slice_idx} 时内存不足，跳过")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        if slice_features:
            view_feature = torch.stack(slice_features).mean(dim=0)
            return view_feature.squeeze(0)
        else:
            return torch.zeros(self.vivit_encoder.dim, device=device)
    
    def forward(self, batch_data):
        device = next(self.parameters()).device
        batch_features = []
        
        for patient_idx in range(len(batch_data['patient_id'])):
            view_features = []
            
            for view_idx, view_batch in enumerate(batch_data['image']):
                if patient_idx < len(view_batch) and view_batch[patient_idx].dim() > 0:
                    view_tensor = view_batch[patient_idx]
                    
                    if view_tensor.shape[0] == 0:
                        continue
                    
                    view_feature = self.encode_view(view_tensor, device)
                    view_features.append(view_feature)
            
            if view_features:
                view_sequence = torch.stack(view_features).unsqueeze(0)
                batch_size = view_sequence.shape[0]
                view_cls = self.view_cls_token.expand(batch_size, -1, -1)
                view_sequence = torch.cat((view_cls, view_sequence), dim=1)
                fused_output = self.view_fusion_transformer(view_sequence)
                patient_feature = fused_output[:, 0]
                batch_features.append(patient_feature.squeeze(0))
            else:
                batch_features.append(torch.zeros(self.vivit_encoder.dim, device=device))
        
        batch_features_tensor = torch.stack(batch_features)
        return self.classifier(batch_features_tensor)

# ==================== 评估和可视化功能 ====================
class ModelEvaluator:
    """模型评估和可视化类"""
    
    def __init__(self, class_names=['NC', 'HCM', 'DCM']):
        self.class_names = class_names
        self.num_classes = len(class_names)
    
    def evaluate_model(self, model, dataloader, device, criterion, phase="验证"):
        """评估模型并返回详细结果"""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        all_patient_ids = []
        
        eval_start_time = time.time()
        
        with torch.no_grad():
            for batch in dataloader:
                labels = batch['label'].to(device)
                patient_ids = batch['patient_id']
                
                with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                    outputs = model(batch)
                    loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                probabilities = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_patient_ids.extend(patient_ids)
        
        eval_time = time.time() - eval_start_time
        accuracy = 100 * correct / total if total > 0 else 0
        avg_loss = total_loss / len(dataloader)
        
        print(f'      {phase}完成 | 用时: {eval_time:6.1f}s | 准确率: {accuracy:6.2f}%')
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'predictions': np.array(all_predictions),
            'labels': np.array(all_labels),
            'probabilities': np.array(all_probabilities),
            'patient_ids': all_patient_ids,
            'eval_time': eval_time
        }
    
    def generate_classification_report(self, results, save_path=None):
        """生成分类报告"""
        print("\n" + "="*50)
        print("分类报告")
        print("="*50)
        
        report = classification_report(
            results['labels'], 
            results['predictions'], 
            target_names=self.class_names,
            digits=4
        )
        print(report)
        
        if save_path:
            with open(f"{save_path}_classification_report.txt", 'w') as f:
                f.write(report)
        
        return report
    
    def plot_confusion_matrix(self, results, save_path=None):
        """绘制混淆矩阵"""
        cm = confusion_matrix(results['labels'], results['predictions'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names)
        plt.title('混淆矩阵')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}_confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        return cm
    
    def plot_roc_curves(self, results, save_path=None):
        """绘制ROC曲线"""
        # 二值化标签
        y_true_bin = label_binarize(results['labels'], classes=range(self.num_classes))
        y_score = results['probabilities']
        
        # 计算每个类别的ROC曲线和AUC
        fpr = {}
        tpr = {}
        roc_auc = {}
        
        plt.figure(figsize=(10, 8))
        
        for i in range(self.num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            
            plt.plot(fpr[i], tpr[i], lw=2, 
                    label=f'{self.class_names[i]} (AUC = {roc_auc[i]:.3f})')
        
        # 计算微平均ROC曲线
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        plt.plot(fpr["micro"], tpr["micro"],
                label=f'微平均 (AUC = {roc_auc["micro"]:.3f})',
                color='deeppink', linestyle=':', linewidth=4)
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='随机分类器')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假正率')
        plt.ylabel('真正率')
        plt.title('多分类ROC曲线')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(f"{save_path}_roc_curves.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        return roc_auc
    
    def save_prediction_probabilities(self, results, save_path):
        """保存预测概率表格"""
        df = pd.DataFrame({
            'Patient_ID': results['patient_ids'],
            'True_Label': results['labels'],
            'True_Class': [self.class_names[i] for i in results['labels']],
            'Predicted_Label': results['predictions'],
            'Predicted_Class': [self.class_names[i] for i in results['predictions']]
        })
        
        # 添加每个类别的概率
        for i, class_name in enumerate(self.class_names):
            df[f'Probability_{class_name}'] = results['probabilities'][:, i]
        
        df['Correct'] = df['True_Label'] == df['Predicted_Label']
        
        # 保存为CSV
        df.to_csv(f"{save_path}_prediction_probabilities.csv", index=False, encoding='utf-8-sig')
        
        print(f"预测概率表格已保存: {save_path}_prediction_probabilities.csv")
        
        return df
    
    def generate_detailed_metrics(self, results, save_path=None):
        """生成详细指标"""
        cm = confusion_matrix(results['labels'], results['predictions'])
        
        # 计算每个类别的指标
        metrics = {}
        for i, class_name in enumerate(self.class_names):
            TP = cm[i, i]
            FP = cm[:, i].sum() - TP
            FN = cm[i, :].sum() - TP
            TN = cm.sum() - TP - FP - FN
            
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics[class_name] = {
                'Precision': precision,
                'Recall': recall,
                'Specificity': specificity,
                'F1-Score': f1_score,
                'TP': TP,
                'FP': FP,
                'FN': FN,
                'TN': TN
            }
        
        # 打印详细指标
        print("\n" + "="*60)
        print("详细分类指标")
        print("="*60)
        
        metrics_df = pd.DataFrame(metrics).T
        print(metrics_df.round(4))
        
        if save_path:
            metrics_df.to_csv(f"{save_path}_detailed_metrics.csv", encoding='utf-8-sig')
            print(f"详细指标已保存: {save_path}_detailed_metrics.csv")
        
        return metrics_df

# ==================== 学习率调度器 ====================
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine_decay)
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ==================== 训练函数 ====================
def train_true_vivit():
    DATA_PATH = "D:\\MRI_Data\\BatchData-1st"
    
    # 配置参数
    BATCH_SIZE = 4
    GRAD_ACCUM_STEPS = 4
    EFFECTIVE_BATCH_SIZE = BATCH_SIZE * GRAD_ACCUM_STEPS
    
    IMAGE_SIZE = 128
    NUM_FRAMES = 25
    DIM = 192
    DEPTH = 4
    HEADS = 3
    PATCH_SIZE = 16
    
    # 数据集划分比例
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    print("=== 真正ViViT模型训练 (带完整评估和可视化) ===")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 创建完整数据集
    full_dataset_start = time.time()
    full_dataset = CardiacThreeClassDataset(
        root_dir=DATA_PATH,
        num_frames=NUM_FRAMES,
        min_views=2,
        image_size=IMAGE_SIZE
    )
    dataset_load_time = time.time() - full_dataset_start
    
    if len(full_dataset) == 0:
        print("没有有效数据，无法训练")
        return
    
    print(f"数据集加载完成 | 用时: {dataset_load_time:.1f}s")
    print(f"完整数据集大小: {len(full_dataset)} 个样本")
    
    # 数据集划分
    split_start = time.time()
    train_size = int(TRAIN_RATIO * len(full_dataset))
    val_size = int(VAL_RATIO * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )
    
    print(f"训练集: {len(train_dataset)} 个样本")
    print(f"验证集: {len(val_dataset)} 个样本") 
    print(f"测试集: {len(test_dataset)} 个样本")
    
    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: cardiac_collate_fn(batch, max_views=4, image_size=IMAGE_SIZE,num_frames=NUM_FRAMES)
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: cardiac_collate_fn(batch, max_views=4, image_size=IMAGE_SIZE,num_frames=NUM_FRAMES)
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: cardiac_collate_fn(batch, max_views=4, image_size=IMAGE_SIZE,num_frames=NUM_FRAMES)
    )
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    model = CardiacViViT(
        num_classes=3,
        image_size=IMAGE_SIZE,
        num_frames=NUM_FRAMES,
        dim=DIM,
        depth=DEPTH,
        heads=HEADS,
        patch_size=PATCH_SIZE
    ).to(device)
    
    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数: {total_params:,} (可训练: {trainable_params:,})")
    
    # 优化器和损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )
    
    # 计算总训练步数
    num_epochs = 50  # 增加总轮数
    num_training_steps = num_epochs * len(train_dataloader) // GRAD_ACCUM_STEPS
    num_warmup_steps = num_training_steps // 5
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # 初始化评估器
    evaluator = ModelEvaluator(class_names=['NC', 'HCM', 'DCM'])
    
    print(f"训练配置完成")
    print(f"总训练轮数: {num_epochs}")
    print(f"早停耐心值: 8轮")
    print("-" * 80)
    
    # 训练统计
    train_start_time = time.time()
    best_val_accuracy = 0.0
    global_step = 0
    patience = 10  # 增加早停耐心值
    no_improve_count = 0
    
    # 记录训练历史
    train_history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    

    
    print(f"开始训练真正ViViT模型...")

    # 创建结果目录
    results_dir = f"vivit_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    import os
    os.makedirs(results_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f'\nEpoch {epoch+1:02d}/{num_epochs} 开始...')
        
        # 训练阶段
        model.train()
        train_total_loss = 0
        train_correct = 0
        train_total = 0
        accum_loss = 0
        
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(train_dataloader):
            try:
                labels = batch['label'].to(device)
                
                with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                    outputs = model(batch)
                    loss = criterion(outputs, labels)
                
                loss = loss / GRAD_ACCUM_STEPS
                loss.backward()
                
                accum_loss += loss.item() * GRAD_ACCUM_STEPS
                train_total_loss += loss.item() * GRAD_ACCUM_STEPS
                
                _, predicted = torch.max(outputs, 1)
                train_correct += (predicted == labels).sum().item()
                train_total += labels.size(0)
                
                # 显示批次进度（每5个批次或重要节点显示）
                if batch_idx % 5 == 0 or (batch_idx + 1) % GRAD_ACCUM_STEPS == 0 or batch_idx == len(train_dataloader) - 1:
                    current_loss = loss.item() * GRAD_ACCUM_STEPS
                    batch_accuracy = (predicted == labels).float().mean().item() * 100
                    print(f'  Batch {batch_idx:03d}/{len(train_dataloader)} | '
                          f'Loss: {current_loss:.4f} | Acc: {batch_accuracy:5.1f}%')
                
                # 梯度累积更新
                if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    global_step += 1
                    current_lr = optimizer.param_groups[0]['lr']
                    
                    # 每10步或重要节点显示梯度更新信息
                    if global_step % 10 == 0 or global_step <= 5:
                        avg_accum_loss = accum_loss / GRAD_ACCUM_STEPS
                        print(f'[Step {global_step:04d}] LR: {current_lr:.2e} | '
                              f'Accum Loss: {avg_accum_loss:.4f}')
                    
                    accum_loss = 0
                
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"批次 {batch_idx} 内存不足，跳过...")
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    accum_loss = 0
                    continue
                else:
                    raise e
        
        # 处理最后一个不完整的梯度累积
        if accum_loss > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            print(f'[Step {global_step:04d}] 最后累积步骤更新完成')
        
        train_time = time.time() - epoch_start_time
        
        # 验证阶段
        print(f'  开始验证...')
        val_results = evaluator.evaluate_model(model, val_dataloader, device, criterion, "验证")
        
        # 训练统计
        train_accuracy = 100 * train_correct / train_total if train_total > 0 else 0
        train_avg_loss = train_total_loss / len(train_dataloader)
        
        # 记录历史
        train_history['train_loss'].append(train_avg_loss)
        train_history['train_acc'].append(train_accuracy)
        train_history['val_loss'].append(val_results['loss'])
        train_history['val_acc'].append(val_results['accuracy'])
        
        epoch_time = time.time() - epoch_start_time
        
        # 内存使用
        if device.type == 'cuda':
            gpu_memory = torch.cuda.memory_allocated() / 1024**3
            gpu_memory_max = torch.cuda.max_memory_allocated() / 1024**3
            memory_info = f" | GPU内存: {gpu_memory:.2f}GB (峰值: {gpu_memory_max:.2f}GB)"
        else:
            memory_info = ""
        
        # 进度和预计完成时间
        elapsed_time = time.time() - train_start_time
        progress = (epoch + 1) / num_epochs * 100
        
        if epoch > 0:
            estimated_total_time = elapsed_time / progress * 100
            estimated_remaining = estimated_total_time - elapsed_time
            eta = datetime.now() + timedelta(seconds=estimated_remaining)
            time_info = f" | 进度: {progress:5.1f}% | 剩余: {estimated_remaining/60:6.1f}min"
        else:
            time_info = f" | 进度: {progress:5.1f}%"
        
        print("-" * 80)
        print(f'Epoch {epoch+1:02d} 完成 | 用时: {epoch_time:6.1f}s{memory_info}{time_info}')
        print(f'  训练损失: {train_avg_loss:.4f} | 训练准确率: {train_accuracy:6.2f}%')
        print(f'  验证损失: {val_results["loss"]:.4f} | 验证准确率: {val_results["accuracy"]:6.2f}%')
        
        if epoch > 0:
            print(f'  预计完成: {eta.strftime("%m-%d %H:%M")}')
        
        # 早停和最佳模型保存
        if val_results['accuracy'] > best_val_accuracy:
            best_val_accuracy = val_results['accuracy']
            no_improve_count = 0
            torch.save(model.state_dict(), f'{results_dir}/true_vivit_cardiac_best.pth')
            print(f'      🎯 新的最佳验证准确率! 模型已保存')
        else:
            no_improve_count += 1
            print(f'      ⏳ 验证准确率未提升 ({no_improve_count}/{patience})')
            if no_improve_count >= patience:
                print(f"      ⏹️  早停: {patience} 轮验证集无改善")
                break
        
        # 只在最后一轮保存检查点，避免占用内存
        if epoch == num_epochs - 1:
            checkpoint = {
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_history': train_history,
                'best_val_accuracy': best_val_accuracy
            }
            torch.save(checkpoint, f'{results_dir}/true_vivit_cardiac_final_checkpoint.pth')
            print(f'      💾 最终检查点已保存')

    
    # 最终测试和可视化
    print("\n" + "="*80)
    print("开始最终测试和可视化...")
    print("="*80)
    

    
    # 使用最佳模型进行最终测试
    print("加载最佳模型进行测试...")
    model.load_state_dict(torch.load(f'{results_dir}/true_vivit_cardiac_best.pth'))
    
    # 测试集评估
    test_results = evaluator.evaluate_model(model, test_dataloader, device, criterion, "最终测试")
    
    # 生成所有可视化结果
    print("\n生成分类报告...")
    evaluator.generate_classification_report(test_results, f"{results_dir}/test")
    
    print("\n生成混淆矩阵...")
    evaluator.plot_confusion_matrix(test_results, f"{results_dir}/test")
    
    print("\n生成ROC曲线...")
    roc_auc = evaluator.plot_roc_curves(test_results, f"{results_dir}/test")
    
    print("\n生成详细指标...")
    metrics_df = evaluator.generate_detailed_metrics(test_results, f"{results_dir}/test")
    
    print("\n保存预测概率表格...")
    prob_df = evaluator.save_prediction_probabilities(test_results, f"{results_dir}/test")
    
    # 训练完成统计
    total_training_time = time.time() - train_start_time
    
    print(f"\n" + "="*80)
    print(f"训练完成!")
    print("="*80)
    print(f"总训练时间: {total_training_time/60:.1f} 分钟")
    print(f"最佳验证准确率: {best_val_accuracy:.2f}%")
    print(f"最终测试准确率: {test_results['accuracy']:.2f}%")
    print(f"AUC分数: {roc_auc}")
    print(f"所有结果已保存到: {results_dir}")
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 保存最终模型
    torch.save(model.state_dict(), f'{results_dir}/true_vivit_cardiac_classifier_final.pth')
    print("最终模型已保存")

if __name__ == "__main__":
    train_true_vivit()
