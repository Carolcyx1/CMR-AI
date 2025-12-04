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

import matplotlib.pyplot as plt
# 设置matplotlib的全局字体为支持中文的字体
# plt.rcParams['font.sans-serif'] = ['SimHei'] # 'SimHei'为黑体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False # 解决负号'-'显示为方块的问题

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 使用GPU 0



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
                # print(view_tensor.shape)
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

import timm
import torch
import torch.nn as nn

import timm
import torch
import torch.nn as nn
import os

class FixedPathVideoSwinCardiac(nn.Module):
    """修复路径问题的Video-Swin分类器"""

    def __init__(self, num_classes=3, model_name='swin_tiny_patch4_window7_224'):
        super().__init__()

        # 可能的权重文件路径
        possible_paths = [
            'checkpoints/swin_tiny_patch4_window7_224.pth',  # 相对路径
            './checkpoints/swin_tiny_patch4_window7_224.pth', # 当前目录
            'swin_tiny_patch4_window7_224.pth',              # 直接放在代码目录
            '../checkpoints/swin_tiny_patch4_window7_224.pth', # 上级目录
        ]

        checkpoint_path = None
        for path in possible_paths:
            if os.path.exists(path):
                checkpoint_path = path
                print(f"找到权重文件: {path}")
                break

        # 创建模型
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
            in_chans=1,
        )

        # 加载预训练权重
        if checkpoint_path:
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                print(f"成功加载权重文件: {checkpoint_path}")

                # 处理权重文件格式
                if 'model' in checkpoint:
                    # 如果是包含'model'键的checkpoint
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    # 如果是包含'state_dict'键的checkpoint
                    state_dict = checkpoint['state_dict']
                else:
                    # 直接是state_dict
                    state_dict = checkpoint

                # 适配单通道输入
                if 'patch_embed.proj.weight' in state_dict:
                    original_weight = state_dict['patch_embed.proj.weight']
                    if original_weight.shape[1] == 3:  # 如果是3通道权重
                        state_dict['patch_embed.proj.weight'] = original_weight.mean(dim=1, keepdim=True)
                        print("已适配单通道输入")

                # 加载权重
                msg = self.backbone.load_state_dict(state_dict, strict=False)
                print(f"权重加载信息: {msg}")

            except Exception as e:
                print(f"加载权重失败: {e}")
                print("使用随机初始化")
        else:
            print("未找到权重文件，使用随机初始化")

        feature_dim = self.backbone.num_features
        print(f"特征维度: {feature_dim}")

        # 注意力机制
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        self.view_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def process_single_view(self, view_tensor):
        """处理单个视图 - 修复版本"""
        # print(f"🔍 输入张量形状: {view_tensor.shape}")

        # 处理形状 [2, 25, 1, 224, 224] -> [2, 25, 224, 224]
        if view_tensor.dim() == 5:
            # 移除通道维度 (第3维)
            view_tensor = view_tensor.squeeze(2)  # 从 [2, 25, 1, 224, 224] 变为 [2, 25, 224, 224]
            # print(f"✅ 移除通道维度后: {view_tensor.shape}")

        if view_tensor.dim() != 4:
            # print(f"❌ 不支持的维度: {view_tensor.dim()}，期望4维")
            return None

        try:
            num_slices, num_frames, H, W = view_tensor.shape
            # print(f"✅ 解析形状: slices={num_slices}, frames={num_frames}, H={H}, W={W}")
        except ValueError as e:
            # print(f"❌ 形状解析失败: {e}")
            return None
        device = next(self.parameters()).device

        slice_features = []

        for slice_idx in range(min(5, num_slices)):  # 最多2个切片
        # for slice_idx in range(max(1, num_slices)):  # 最多2个切片
            slice_data = view_tensor[slice_idx]  # [frames, H, W]

            frame_features = []
            # 选择关键帧
            key_frames = [0, num_frames // 2, num_frames - 1]  # 开始、中间、结束

            for frame_idx in key_frames:
                if frame_idx < num_frames:
                    frame = slice_data[frame_idx].unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                    frame = frame.to(device)

                    # 提取特征
                    frame_feat = self.backbone(frame)
                    frame_features.append(frame_feat)

            if frame_features:
                # 时间注意力
                frame_sequence = torch.stack(frame_features)  # [num_frames, 1, feature_dim]
                attended_frames, _ = self.temporal_attention(
                    frame_sequence, frame_sequence, frame_sequence
                )
                slice_feature = attended_frames.mean(dim=0).squeeze(0)
                slice_features.append(slice_feature)

        if slice_features:
            return torch.stack(slice_features).mean(dim=0)
        return None

    def forward(self, batch_data):
        device = next(self.parameters()).device
        batch_features = []

        for patient_idx in range(len(batch_data['patient_id'])):
            view_features = []

            for view_idx, view_batch in enumerate(batch_data['image']):
                if (patient_idx < len(view_batch) and
                    view_batch[patient_idx].dim() > 0 and
                    view_batch[patient_idx].shape[0] > 0):

                    view_tensor = view_batch[patient_idx]
                    view_feature = self.process_single_view(view_tensor)

                    if view_feature is not None:
                        view_features.append(view_feature)

            # 多视图融合
            if view_features:
                view_sequence = torch.stack(view_features).unsqueeze(0)
                fused_views, _ = self.view_attention(
                    view_sequence, view_sequence, view_sequence
                )
                patient_feature = fused_views.mean(dim=1).squeeze(0)
                batch_features.append(patient_feature)
            else:
                feature_dim = self.backbone.num_features
                batch_features.append(torch.zeros(feature_dim, device=device))

        features_tensor = torch.stack(batch_features)
        return self.classifier(features_tensor)

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
        # plt.title('混淆矩阵')
        # plt.xlabel('预测标签')
        # plt.ylabel('真实标签')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
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
                label=f'Micro Average (AUC = {roc_auc["micro"]:.3f})',
                color='deeppink', linestyle=':', linewidth=4)

        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        # plt.xlabel('假正率')
        # plt.ylabel('真正率')
        # plt.title('多分类ROC曲线')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multi-class ROC Curves')
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



def train_video_swin_cardiac():
    """使用Video-Swin的训练函数"""

    DATA_PATH = "/data/chengyuxi/datasets/CMR/BatchData-1st"

    # 配置参数
    BATCH_SIZE = 4
    GRAD_ACCUM_STEPS = 4
    IMAGE_SIZE = 224

    # 获取当前Python文件名
    import os
    current_file = os.path.splitext(os.path.basename(__file__))[0]
    # 创建结果目录
    results_dir = f"{current_file}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    # # 创建结果目录
    # results_dir = f"swin1_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"📁 结果将保存到: {results_dir}")

    print("=== 初始化数据集 ===")
    # 创建数据集
    full_dataset = CardiacThreeClassDataset(
        root_dir=DATA_PATH,
        num_frames=25,
        min_views=2,
        image_size=IMAGE_SIZE
    )

    if len(full_dataset) == 0:
        print("错误: 没有有效数据!")
        return

    # 数据集划分
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    # 数据加载器
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: cardiac_collate_fn(batch, max_views=4, image_size=IMAGE_SIZE, num_frames=25)
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: cardiac_collate_fn(batch, max_views=4, image_size=IMAGE_SIZE, num_frames=25)
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: cardiac_collate_fn(batch, max_views=4, image_size=IMAGE_SIZE, num_frames=25)
    )

    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建模型
    model = FixedPathVideoSwinCardiac(
        num_classes=3,
        model_name='swin_tiny_patch4_window7_224'
    ).to(device)

    # 优化器和损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )

    # 学习率调度
    num_epochs = 100
    num_training_steps = num_epochs * len(train_dataloader) // GRAD_ACCUM_STEPS
    num_warmup_steps = num_training_steps // 5

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # 评估器
    evaluator = ModelEvaluator(class_names=['NC', 'HCM', 'DCM'])

    print("=" * 80)
    print("开始训练 Video-Swin-Transformer 模型")
    print("=" * 80)
    print(f"训练集: {len(train_dataset)} 个样本")
    print(f"验证集: {len(val_dataset)} 个样本")
    print(f"测试集: {len(test_dataset)} 个样本")
    print(f"总训练轮数: {num_epochs}")
    print(f"批次大小: {BATCH_SIZE} (梯度累积: {GRAD_ACCUM_STEPS}步)")
    print(f"总训练步数: {num_training_steps}")
    print(f"结果目录: {results_dir}")

    # 训练统计
    train_start_time = time.time()
    best_val_accuracy = 0.0
    patience = 30
    no_improve_count = 0

    # 记录训练历史
    train_history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rate': []
    }

    # 保存训练配置
    config_info = {
        'batch_size': BATCH_SIZE,
        'grad_accum_steps': GRAD_ACCUM_STEPS,
        'image_size': IMAGE_SIZE,
        'num_epochs': num_epochs,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'test_samples': len(test_dataset),
        'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'device': str(device)
    }

    # 保存配置信息
    with open(f'{results_dir}/training_config.txt', 'w') as f:
        for key, value in config_info.items():
            f.write(f"{key}: {value}\n")

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f'\nEpoch {epoch+1:02d}/{num_epochs} 开始...')

        # 训练阶段
        model.train()
        train_total_loss = 0
        train_correct = 0
        train_total = 0
        accum_loss = 0

        # 批次进度监控
        batch_start_time = time.time()

        for batch_idx, batch in enumerate(train_dataloader):
            try:
                labels = batch['label'].to(device)

                # 前向传播
                with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                    outputs = model(batch)
                    loss = criterion(outputs, labels)

                # 梯度累积
                loss = loss / GRAD_ACCUM_STEPS
                loss.backward()

                accum_loss += loss.item() * GRAD_ACCUM_STEPS
                train_total_loss += loss.item() * GRAD_ACCUM_STEPS

                # 计算准确率
                _, predicted = torch.max(outputs, 1)
                train_correct += (predicted == labels).sum().item()
                train_total += labels.size(0)

                # 显示批次进度
                if (batch_idx + 1) % 2 == 0:  # 每2个批次显示一次
                    batch_time = time.time() - batch_start_time
                    current_loss = accum_loss / min(batch_idx + 1, GRAD_ACCUM_STEPS)
                    batch_accuracy = 100 * (predicted == labels).float().mean().item()

                    print(f'  Batch {batch_idx+1:03d}/{len(train_dataloader)} | '
                          f'Loss: {current_loss:.4f} | Acc: {batch_accuracy:5.1f}% | '
                          f'Time: {batch_time:.1f}s')
                    batch_start_time = time.time()

                # 梯度累积更新
                if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    current_lr = optimizer.param_groups[0]['lr']

                    # 显示梯度更新信息
                    print(f'  [Grad Step] LR: {current_lr:.2e} | '
                          f'Accum Loss: {accum_loss/GRAD_ACCUM_STEPS:.4f}')

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
            print(f'  [Final Grad Step] 最后累积步骤更新完成')

        # 计算训练统计
        train_accuracy = 100 * train_correct / train_total if train_total > 0 else 0
        train_avg_loss = train_total_loss / len(train_dataloader)
        current_lr = optimizer.param_groups[0]['lr']

        train_time = time.time() - epoch_start_time

        # 验证阶段
        print(f'  开始验证...')
        val_start_time = time.time()
        val_results = evaluator.evaluate_model(model, val_dataloader, device, criterion, "验证")
        val_time = time.time() - val_start_time

        # 记录历史
        train_history['epoch'].append(epoch + 1)
        train_history['train_loss'].append(train_avg_loss)
        train_history['train_acc'].append(train_accuracy)
        train_history['val_loss'].append(val_results['loss'])
        train_history['val_acc'].append(val_results['accuracy'])
        train_history['learning_rate'].append(current_lr)

        epoch_time = time.time() - epoch_start_time

        # 内存使用统计
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
            time_info = f" | 进度: {progress:5.1f}% | 剩余: {estimated_remaining/60:6.1f}min | ETA: {eta.strftime('%m-%d %H:%M')}"
        else:
            time_info = f" | 进度: {progress:5.1f}%"

        print("-" * 80)
        print(f'Epoch {epoch+1:02d} 完成 | 用时: {epoch_time:6.1f}s{memory_info}{time_info}')
        print(f'  训练损失: {train_avg_loss:.4f} | 训练准确率: {train_accuracy:6.2f}%')
        print(f'  验证损失: {val_results["loss"]:.4f} | 验证准确率: {val_results["accuracy"]:6.2f}%')
        print(f'  学习率: {current_lr:.2e}')
        print(f'  训练时间: {train_time:.1f}s | 验证时间: {val_time:.1f}s')

        # 早停和最佳模型保存
        if val_results['accuracy'] > best_val_accuracy:
            best_val_accuracy = val_results['accuracy']
            no_improve_count = 0
            best_model_path = f'{results_dir}/video_swin_cardiac_best.pth'
            torch.save(model.state_dict(), best_model_path)
            print(f'      🎯 新的最佳验证准确率! 模型已保存到: {best_model_path}')
        else:
            no_improve_count += 1
            print(f'      ⏳ 验证准确率未提升 ({no_improve_count}/{patience})')
            if no_improve_count >= patience:
                print(f"      ⏹️  早停: {patience} 轮验证集无改善")
                break

        # 每10轮保存一次检查点
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_history': train_history,
                'best_val_accuracy': best_val_accuracy
            }
            checkpoint_path = f'{results_dir}/video_swin_cardiac_checkpoint_epoch_{epoch+1}.pth'
            torch.save(checkpoint, checkpoint_path)
            print(f'      💾 检查点已保存: {checkpoint_path}')

    # 训练完成统计
    total_training_time = time.time() - train_start_time
    end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print("\n" + "="*80)
    print("训练完成!")
    print("="*80)
    print(f"总训练时间: {total_training_time/60:.1f} 分钟")
    print(f"总训练轮数: {epoch + 1}")
    print(f"最佳验证准确率: {best_val_accuracy:.2f}%")
    print(f"结果目录: {results_dir}")

    # 更新配置信息
    config_info['end_time'] = end_time
    config_info['total_training_time_minutes'] = total_training_time/60
    config_info['best_val_accuracy'] = best_val_accuracy
    config_info['total_epochs_trained'] = epoch + 1

    with open(f'{results_dir}/training_config.txt', 'w') as f:
        for key, value in config_info.items():
            f.write(f"{key}: {value}\n")

    # 最终测试
    print("\n开始最终测试...")
    try:
        best_model_path = f'{results_dir}/video_swin_cardiac_best.pth'
        model.load_state_dict(torch.load(best_model_path))
        test_results = evaluator.evaluate_model(model, test_dataloader, device, criterion, "最终测试")

        # 更新配置信息
        config_info['test_accuracy'] = test_results['accuracy']

        # 生成可视化结果
        print("\n生成评估报告...")
        evaluator.generate_classification_report(test_results, f"{results_dir}/final_results")
        evaluator.plot_confusion_matrix(test_results, f"{results_dir}/final_results")
        evaluator.plot_roc_curves(test_results, f"{results_dir}/final_results")
        evaluator.generate_detailed_metrics(test_results, f"{results_dir}/final_results")
        evaluator.save_prediction_probabilities(test_results, f"{results_dir}/final_results")

        print(f"\n最终测试准确率: {test_results['accuracy']:.2f}%")

    except Exception as e:
        print(f"最终测试失败: {e}")
        config_info['test_accuracy'] = 'Failed'

    # 保存训练历史
    try:
        history_df = pd.DataFrame(train_history)
        history_path = f'{results_dir}/training_history.csv'
        history_df.to_csv(history_path, index=False)
        print(f"训练历史已保存: {history_path}")

        # 绘制训练曲线
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(train_history['epoch'], train_history['train_loss'], label='train_loss')
        plt.plot(train_history['epoch'], train_history['val_loss'], label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        # plt.title('训练和验证损失')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(train_history['epoch'], train_history['train_acc'], label='train_acc')
        plt.plot(train_history['epoch'], train_history['val_acc'], label='val_acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{results_dir}/training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()

        print(f"训练曲线已保存: {results_dir}/training_curves.png")

    except Exception as e:
        print(f"保存训练历史失败: {e}")

    # 最终更新配置信息
    with open(f'{results_dir}/training_config.txt', 'w') as f:
        for key, value in config_info.items():
            f.write(f"{key}: {value}\n")

    print(f"\n🎉 所有结果已保存到: {results_dir}")
    print(f"完成时间: {end_time}")

    # 显示目录结构
    print(f"\n📂 结果目录内容:")
    for item in os.listdir(results_dir):
        print(f"  - {item}")
if __name__ == "__main__":
    train_video_swin_cardiac()



