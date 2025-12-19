import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import timm
import torchvision.models as models
from pathlib import Path
import numpy as np
import re
import time
import math
from datetime import datetime, timedelta

# 可视化相关导入
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# Grad-CAM相关导入
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from torchinfo import summary

# 设置matplotlib
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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
                else:
                    pass

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
                images_by_view[view_idx].append(torch.zeros(0, 1, num_frames, image_size, image_size))

    return {
        'image': images_by_view,
        'label': labels,
        'patient_id': patient_ids,
        'available_views': available_views_list
    }

# ==================== 模型定义 ====================
class FixedPathVideoSwinCardiac(nn.Module):
    """Swin Transformer 模型"""
    
    def __init__(self, num_classes=3, model_name='swin_tiny_patch4_window7_224'):
        super().__init__()
        
        possible_paths = [
            'checkpoints/swin_tiny_patch4_window7_224.pth',
            './checkpoints/swin_tiny_patch4_window7_224.pth',
            'swin_tiny_patch4_window7_224.pth',
            '../checkpoints/swin_tiny_patch4_window7_224.pth',
        ]
        
        checkpoint_path = None
        for path in possible_paths:
            if os.path.exists(path):
                checkpoint_path = path
                print(f"找到权重文件: {path}")
                break
        
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
            in_chans=1,
        )
        
        if checkpoint_path:
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                print(f"成功加载权重文件: {checkpoint_path}")
                
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                msg = self.backbone.load_state_dict(state_dict, strict=False)
                print(f"权重加载信息: {msg}")
            except Exception as e:
                print(f"加载权重失败: {e}")
                print("使用随机初始化")
        else:
            print("未找到权重文件，使用随机初始化")
        
        feature_dim = self.backbone.num_features
        print(f"Swin Backbone 特征维度: {feature_dim}")
        
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
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def process_single_view(self, view_tensor):
        if view_tensor.dim() == 5:
            view_tensor = view_tensor.squeeze(2)
        
        if view_tensor.dim() != 4:
            return None
        
        num_slices, num_frames, H, W = view_tensor.shape
        device = next(self.parameters()).device
        slice_features = []
        
        for slice_idx in range(min(5, num_slices)):
            slice_data = view_tensor[slice_idx]
            frame_features = []
            key_frames = [0, num_frames // 2, num_frames - 1]
            
            for frame_idx in key_frames:
                if frame_idx < num_frames:
                    frame = slice_data[frame_idx].unsqueeze(0).unsqueeze(0)
                    frame = frame.to(device)
                    frame_feat = self.backbone(frame)
                    frame_features.append(frame_feat)
            
            if frame_features:
                frame_sequence = torch.stack(frame_features)
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

class CardiacCNN(nn.Module):
    """CNN对比模型"""
    
    def __init__(self, num_classes=3, backbone_name='resnet50', pretrained=False, checkpoint_dir='./checkpoints'):
        super().__init__()
        
        self.backbone_name = backbone_name
        self.checkpoint_dir = checkpoint_dir
        
        if backbone_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=False)
            self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.backbone.fc = nn.Identity()
            feature_dim = 512
        
        elif backbone_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=False)
            self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.backbone.fc = nn.Identity()
            feature_dim = 2048
        
        elif backbone_name == 'densenet121':
            self.backbone = models.densenet121(pretrained=False)
            original_conv = self.backbone.features.conv0
            self.backbone.features.conv0 = nn.Conv2d(
                1, original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None
            )
            self.backbone.classifier = nn.Identity()
            feature_dim = 1024
        
        elif backbone_name == 'efficientnet_b0':
            self.backbone = timm.create_model('efficientnet_b0',
                                              pretrained=False,
                                              num_classes=0,
                                              in_chans=1)
            feature_dim = self.backbone.num_features
        
        elif backbone_name == 'convnext_tiny':
            self.backbone = timm.create_model('convnext_tiny',
                                              pretrained=False,
                                              num_classes=0,
                                              in_chans=1)
            feature_dim = self.backbone.num_features
        
        else:
            raise ValueError(f"不支持的骨干网络: {backbone_name}")
        
        print(f"{backbone_name} 特征维度: {feature_dim}")
        
        if pretrained:
            self._load_pretrained_weights()
        
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
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        self.feature_dim = feature_dim
    
    def _load_pretrained_weights(self):
        weight_file = None
        possible_files = [
            f"{self.backbone_name}.pth",
            os.path.join(self.checkpoint_dir, f"{self.backbone_name}.pth"),
            f"checkpoints/{self.backbone_name}.pth",
        ]
        
        for file_path in possible_files:
            if os.path.exists(file_path):
                weight_file = file_path
                break
        
        if not weight_file:
            print(f"⚠️  未找到 {self.backbone_name} 的预训练权重文件")
            return
        
        print(f"✅ 找到预训练权重: {weight_file}")
        
        try:
            checkpoint = torch.load(weight_file, map_location='cpu')
            
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('module.'):
                    new_key = key[7:]
                elif key.startswith('backbone.'):
                    new_key = key[9:]
                else:
                    new_key = key
                
                if 'fc' in new_key or 'classifier' in new_key or 'head' in new_key:
                    continue
                
                new_state_dict[new_key] = value
            
            self.backbone.load_state_dict(new_state_dict, strict=False)
            print(f"✅ 预训练权重加载完成")
        
        except Exception as e:
            print(f"❌ 权重加载失败: {e}")
    
    def process_single_view(self, view_tensor):
        if view_tensor.dim() == 5:
            view_tensor = view_tensor.squeeze(2)
        
        if view_tensor.dim() != 4:
            return None
        
        try:
            num_slices, num_frames, H, W = view_tensor.shape
        except ValueError:
            return None
        
        device = next(self.parameters()).device
        slice_features = []
        
        for slice_idx in range(min(5, num_slices)):
            slice_data = view_tensor[slice_idx]
            frame_features = []
            key_frames = [0, num_frames // 2, num_frames - 1]
            
            for frame_idx in key_frames:
                if frame_idx < num_frames:
                    frame = slice_data[frame_idx].unsqueeze(0).unsqueeze(0)
                    frame = frame.to(device)
                    frame_feat = self.backbone(frame)
                    frame_features.append(frame_feat)
            
            if frame_features:
                frame_sequence = torch.stack(frame_features)
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
            
            if view_features:
                view_sequence = torch.stack(view_features).unsqueeze(0)
                fused_views, _ = self.view_attention(
                    view_sequence, view_sequence, view_sequence
                )
                patient_feature = fused_views.mean(dim=1).squeeze(0)
                batch_features.append(patient_feature)
            else:
                batch_features.append(torch.zeros(self.feature_dim, device=device))
        
        features_tensor = torch.stack(batch_features)
        return self.classifier(features_tensor)

# ==================== 注意力记录器 (用于可视化) ====================
class AttentionRecorder:
    def __init__(self, model):
        self.model = model
        self.temporal_weights = []
        self.view_weights = None
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        def temporal_hook(module, input, output):
            weights = output[1].detach().cpu()
            self.temporal_weights.append(weights)
        
        def view_hook(module, input, output):
            weights = output[1].detach().cpu()
            self.view_weights = weights
        
        self.hooks.append(self.model.temporal_attention.register_forward_hook(temporal_hook))
        self.hooks.append(self.model.view_attention.register_forward_hook(view_hook))
    
    def clear(self):
        self.temporal_weights = []
        self.view_weights = None
    
    def remove(self):
        for h in self.hooks:
            h.remove()

def reshape_transform(tensor):
    """适配 Swin Transformer 输出的万能转换函数"""
    if isinstance(tensor, (tuple, list)):
        tensor = tensor[0]
    
    if tensor.dim() == 3:
        B = tensor.shape[0]
        L = tensor.shape[1]
        C = tensor.shape[2]
        H = int(np.sqrt(L))
        W = int(np.sqrt(L))
        result = tensor.transpose(1, 2).reshape(B, C, H, W)
        return result
    
    if tensor.dim() == 4:
        B = tensor.shape[0]
        D1 = tensor.shape[1]
        D2 = tensor.shape[2]
        D3 = tensor.shape[3]
        
        if D3 >= D1 and D3 >= D2:
            result = tensor.permute(0, 3, 1, 2)
            return result
        
        return tensor
    
    return tensor

# ==================== 模型工厂 ====================
class ModelFactory:
    @staticmethod
    def create_model(model_type, **kwargs):
        model_type = model_type.lower()
        
        if model_type == 'swin':
            return FixedPathVideoSwinCardiac(num_classes=3)
        elif model_type == 'resnet18':
            return CardiacCNN(num_classes=3, backbone_name='resnet18')
        elif model_type == 'cnn' or model_type == 'resnet50':
            return CardiacCNN(num_classes=3, backbone_name='resnet50')
        elif model_type == 'densenet121':
            return CardiacCNN(num_classes=3, backbone_name='densenet121')
        elif model_type == 'efficientnet':
            return CardiacCNN(num_classes=3, backbone_name='efficientnet_b0')
        elif model_type == 'convnext':
            return CardiacCNN(num_classes=3, backbone_name='convnext_tiny')
        else:
            raise ValueError(f"未知的模型类型: {model_type}")
    
    @staticmethod
    def get_supported_models():
        return {
            'swin': 'Swin Transformer (Vision Transformer)',
            'cnn': 'CNN (默认使用 ResNet50)',
            'resnet': 'ResNet 系列',
            'efficientnet': 'EfficientNet 系列',
            'densenet': 'DenseNet 系列',
            'convnext': 'ConvNeXt 系列'
        }

# ==================== 模型评估器 (包含可视化) ====================
class ModelEvaluator:
    def __init__(self, class_names=['NC', 'HCM', 'DCM']):
        self.class_names = class_names
        self.num_classes = len(class_names)
    
    def evaluate_model(self, model, dataloader, device, criterion, phase="验证"):
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
        print("\n" + "=" * 50)
        print("分类报告")
        print("=" * 50)
        
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
        cm = confusion_matrix(results['labels'], results['predictions'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Class')
        plt.ylabel('Actual Class')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}_confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        return cm
    
    def plot_roc_curves(self, results, save_path=None):
        y_true_bin = label_binarize(results['labels'], classes=range(self.num_classes))
        y_score = results['probabilities']
        
        fpr = {}
        tpr = {}
        roc_auc = {}
        
        plt.figure(figsize=(10, 8))
        
        for i in range(self.num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            
            plt.plot(fpr[i], tpr[i], lw=2,
                     label=f'{self.class_names[i]} (AUC = {roc_auc[i]:.3f})')
        
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        plt.plot(fpr["micro"], tpr["micro"],
                 label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})',
                 color='deeppink', linestyle=':', linewidth=4)
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FP)')
        plt.ylabel('True Positive Rate (TP)')
        plt.title('Multiclass ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(f"{save_path}_roc_curves.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        return roc_auc
    
    def save_prediction_probabilities(self, results, save_path):
        df = pd.DataFrame({
            'Patient_ID': results['patient_ids'],
            'True_Label': results['labels'],
            'True_Class': [self.class_names[i] for i in results['labels']],
            'Predicted_Label': results['predictions'],
            'Predicted_Class': [self.class_names[i] for i in results['predictions']]
        })
        
        for i, class_name in enumerate(self.class_names):
            df[f'Probability_{class_name}'] = results['probabilities'][:, i]
        
        df['Correct'] = df['True_Label'] == df['Predicted_Label']
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

    # ==================== 新增：可视化函数 ====================
    def visualize_inference0(self, model, sample_batch, device, save_dir=None):
        save_dir = os.path.join(save_dir,"visualize_inference")     
        os.makedirs(save_dir, exist_ok=True)
        """可视化推理过程"""
        model.eval()
        recorder = AttentionRecorder(model)
        recorder.clear()
        
        # 获取 Batch 大小
        batch_size = len(sample_batch['patient_id'])    
        print(f"正在分析患者: {sample_batch['patient_id'][0]} ...")
        
        inputs = sample_batch['image']
        batch_input = {
            'image': inputs,
            'patient_id': sample_batch['patient_id'],
            'available_views': sample_batch['available_views']
        }
        
        try:
            with torch.no_grad():
                logits = model(batch_input)
                probs = torch.nn.functional.softmax(logits, dim=1)
                pred_idx = torch.argmax(probs[0]).item()
                conf = probs[0][pred_idx].item()
        except Exception as e:
            print(f"推理阶段出错: {e}")
            return
        
        pred_label = self.class_names[pred_idx]
        true_label_idx = sample_batch['label'][0].item()
        true_label = self.class_names[true_label_idx]
        
        # 绘图初始化
        fig = plt.figure(figsize=(20, 10))
        plt.suptitle(f"Patient Analysis Report: {sample_batch['patient_id'][0]}",
                     fontsize=18, fontweight='bold')
        
        # A. 视图注意力
        ax1 = plt.subplot(2, 3, 1)
        best_view_idx = 0
        best_view_name = "Unknown"
        
        if recorder.view_weights is not None:
            vw = recorder.view_weights[0].mean(dim=0).numpy()
            views = sample_batch['available_views'][0]
            vw = vw[:len(views)]
            vw = vw / (vw.sum() + 1e-6)
            
            sns.barplot(x=views, y=vw, ax=ax1, palette='Blues_r')
            ax1.set_title("View Importance")
            ax1.set_ylim(0, 1.1)
            best_view_idx = np.argmax(vw)
            best_view_name = views[best_view_idx]
        else:
            ax1.text(0.5, 0.5, "No View Attention Captured")
        
        # B. 心脏运动/时间分析
        ax2 = plt.subplot(2, 3, 2)
        try:
            view_batch_list = inputs[best_view_idx]
            if isinstance(view_batch_list, list):
                patient_view_tensor = view_batch_list[0]
            else:
                patient_view_tensor = view_batch_list
            
            slice_data = patient_view_tensor[0].cpu().numpy()
            num_frames = slice_data.shape[0]
            
            use_attention = False
            current_idx_ptr = 0
            target_temporal_weights = None
            
            for v_i, v_name in enumerate(sample_batch['available_views'][0]):
                n_s = inputs[v_i][0].shape[0]
                if v_i == best_view_idx:
                    if current_idx_ptr < len(recorder.temporal_weights):
                        target_temporal_weights = recorder.temporal_weights[current_idx_ptr]
                    break
                current_idx_ptr += min(5, n_s)
            
            if target_temporal_weights is not None:
                t_w = target_temporal_weights
                if t_w.dim() == 3: t_w = t_w[0, 0, :]
                elif t_w.dim() == 2: t_w = t_w[0, :]
                t_w = t_w.numpy()
                
                if np.var(t_w) > 1e-5:
                    use_attention = True
                    t_w_norm = (t_w - t_w.min()) / (t_w.max() - t_w.min() + 1e-6)
                    if len(t_w) == 3:
                        sns.lineplot(x=['Start', 'Mid', 'End'], y=t_w_norm, marker='o', ax=ax2, color='orange', label='Attention')
                    else:
                        sns.lineplot(data=t_w_norm, ax=ax2, color='orange', label='Attention')
                    ax2.set_title(f"Temporal Attention ({best_view_name})")
            
            if not use_attention:
                motion_diff = []
                for t in range(1, num_frames):
                    diff = np.abs(slice_data[t] - slice_data[t-1]).mean()
                    motion_diff.append(diff)
                
                if len(motion_diff) > 0:
                    motion_diff = np.array(motion_diff)
                    motion_norm = (motion_diff - motion_diff.min()) / (motion_diff.max() - motion_diff.min() + 1e-6)
                    x_axis = range(1, num_frames)
                    sns.lineplot(x=x_axis, y=motion_norm, ax=ax2, color='green', linewidth=2, label='Pixel Motion')
                    ax2.fill_between(x_axis, motion_norm, alpha=0.2, color='green')
                    ax2.set_title(f"Heart Motion Cycle ({best_view_name})")
                    ax2.set_xlabel("Frame Index")
                    ax2.set_ylabel("Motion Intensity")
                else:
                    ax2.text(0.5, 0.5, "Not enough frames", ha='center')
        
        except Exception as e:
            print(f"运动曲线绘制失败: {e}")
            ax2.text(0.5, 0.5, "Error plotting motion")
        
        # C. Grad-CAM
        ax3 = plt.subplot(2, 3, 3)
        ax4 = plt.subplot(2, 3, 4)
        ax5 = plt.subplot(2, 3, 5)
        
        try:
            view_batch_list = inputs[best_view_idx]
            patient_view_tensor = view_batch_list[0]
            slice_idx = 0
            frame_idx = patient_view_tensor.shape[1] // 2
            
            raw_frame = patient_view_tensor[slice_idx, frame_idx]
            if raw_frame.dim() == 2:
                target_img_tensor = raw_frame.unsqueeze(0).unsqueeze(0).to(device)
            elif raw_frame.dim() == 3:
                target_img_tensor = raw_frame.unsqueeze(0).to(device)
            
            if target_img_tensor.dim() == 5: target_img_tensor = target_img_tensor.squeeze(2)
            
            target_layers = [model.backbone.layers[-1].blocks[-1].norm1]
            cam = GradCAM(model=model.backbone, target_layers=target_layers, reshape_transform=reshape_transform)
            grayscale_cam = cam(input_tensor=target_img_tensor, targets=None)
            grayscale_cam = grayscale_cam[0, :]
            
            img_np = target_img_tensor.squeeze().cpu().numpy()
            img_min, img_max = img_np.min(), img_np.max()
            if img_max - img_min > 1e-6:
                img_np_norm = (img_np - img_min) / (img_max - img_min)
            else:
                img_np_norm = img_np
            img_rgb = np.stack([img_np_norm]*3, axis=2)
            
            ax3.imshow(img_np_norm, cmap='gray')
            ax3.set_title(f"{best_view_name} (Mid Frame)")
            ax3.axis('off')
            
            visualization = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True)
            ax4.imshow(visualization)
            ax4.set_title(f"Grad-CAM Overlay")
            ax4.axis('off')
            
            norm = mcolors.Normalize(vmin=0, vmax=1)
            sm = cm.ScalarMappable(cmap='jet', norm=norm)
            sm.set_array([])
            cbar4 = fig.colorbar(sm, ax=ax4, fraction=0.046, pad=0.04)
            cbar4.ax.tick_params(labelsize=8)
            
            im = ax5.imshow(grayscale_cam, cmap='jet', vmin=0, vmax=1)
            ax5.set_title("Attention Mask")
            ax5.axis('off')
            
            cbar5 = fig.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)
            cbar5.ax.tick_params(labelsize=8)
        
        except Exception as e:
            ax4.text(0.5, 0.5, "Grad-CAM Failed", ha='center')
            ax4.axis('off')
            ax5.axis('off')
            print(f"Grad-CAM Error: {e}")
        
        # D. 信息摘要
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        info_text = f"Model: Video-Swin\n"
        info_text += f"----------------------\n"
        info_text += f"Patient ID:\n{sample_batch['patient_id'][0]}\n\n"
        
        if pred_label == true_label:
            status = "CORRECT"
            color_code = 'green'
        else:
            status = "WRONG"
            color_code = 'red'
        
        info_text += f"Ground Truth: {true_label}\n"
        info_text += f"Prediction:   {pred_label}\n"
        info_text += f"Confidence:   {conf:.2%}\n"
        info_text += f"Status:       {status}\n\n"
        info_text += f"Best View:    {best_view_name}\n"
        
        ax6.text(0.05, 0.5, info_text, fontsize=12, va='center', family='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))
        
        plt.tight_layout()
        if save_dir:
            save_path = os.path.join(save_dir, f"{sample_batch['patient_id'][0]}_viz.png")
            plt.savefig(save_path)
            print(f"保存至: {save_path}")
        plt.show()
        
        recorder.remove()
    

    # ==================== 可视化函数 (支持 Batch) ====================
    def visualize_inference(self, model, sample_batch, device, save_dir=None):
        """可视化推理过程 (支持任意 Batch Size)"""
        # save_dir = os.path.join(save_dir, "visualize_inference")     
        # os.makedirs(save_dir, exist_ok=True)
        
        model.eval()
        recorder = AttentionRecorder(model)
        recorder.clear()
        
        # 获取 Batch 大小
        batch_size = len(sample_batch['patient_id'])    
        print(f"正在分析 Batch (Size={batch_size})...")
        
        inputs = sample_batch['image']
        batch_input = {
            'image': inputs,
            'patient_id': sample_batch['patient_id'],
            'available_views': sample_batch['available_views']
        }
        
        # 1. 批量推理
        try:
            with torch.no_grad():
                logits = model(batch_input)
                probs_batch = torch.nn.functional.softmax(logits, dim=1)
        except Exception as e:
            print(f"推理阶段出错: {e}")
            recorder.remove()
            return
        
        # 2. 全局钩子指针 (用于追踪 Attention 列表)
        global_hook_ptr = 0

        # 3. 循环处理 Batch 中的每个病人
        for i in range(batch_size):
            patient_id = sample_batch['patient_id'][i]
            print(f"  -> 生成可视化: {patient_id} ({i+1}/{batch_size})")

            # 获取当前病人的结果
            probs = probs_batch[i]
            pred_idx = torch.argmax(probs).item()
            conf = probs[pred_idx].item()
            pred_label = self.class_names[pred_idx]
            
            true_label_idx = sample_batch['label'][i].item()
            true_label = self.class_names[true_label_idx]
            views = sample_batch['available_views'][i]
            
            # --- 绘图初始化 ---
            fig = plt.figure(figsize=(20, 10))
            plt.suptitle(f"Patient Analysis Report: {patient_id}",
                         fontsize=18, fontweight='bold')
            
            # --- A. 视图注意力 ---
            ax1 = plt.subplot(2, 3, 1)
            best_view_idx = 0
            best_view_name = "Unknown"
            
            if recorder.view_weights is not None:
                # 取第 i 个病人的权重
                vw = recorder.view_weights[i].mean(dim=0).cpu().numpy()
                vw = vw[:len(views)] # 截断
                vw = vw / (vw.sum() + 1e-6)
                
                sns.barplot(x=views, y=vw, ax=ax1, palette='Blues_r')
                ax1.set_title("View Importance")
                ax1.set_ylim(0, 1.1)
                best_view_idx = np.argmax(vw)
                best_view_name = views[best_view_idx]
            else:
                ax1.text(0.5, 0.5, "No View Attention Captured")
            
            # --- B. 心脏运动/时间分析 ---
            ax2 = plt.subplot(2, 3, 2)
            try:
                # 获取该病人、最佳视图的数据
                # inputs[view_idx] 是 list，取第 i 个元素
                patient_view_tensor = inputs[best_view_idx][i] 
                
                slice_data = patient_view_tensor[0].cpu().numpy()
                num_frames = slice_data.shape[0]
                
                # 寻找 Temporal Attention
                use_attention = False
                target_temporal_weights = None
                
                # 计算当前病人的局部指针偏移
                temp_ptr = global_hook_ptr
                for v_idx in range(len(views)):
                    v_tensor = inputs[v_idx][i]
                    processed_slices = min(5, v_tensor.shape[0])
                    
                    if v_idx == best_view_idx:
                        if temp_ptr < len(recorder.temporal_weights):
                            target_temporal_weights = recorder.temporal_weights[temp_ptr]
                    temp_ptr += processed_slices
                
                # 绘制
                if target_temporal_weights is not None:
                    t_w = target_temporal_weights
                    if t_w.dim() == 3: t_w = t_w[0, 0, :]
                    elif t_w.dim() == 2: t_w = t_w[0, :]
                    t_w = t_w.cpu().numpy()
                    
                    if np.var(t_w) > 1e-5:
                        use_attention = True
                        t_w_norm = (t_w - t_w.min()) / (t_w.max() - t_w.min() + 1e-6)
                        if len(t_w) == 3:
                            sns.lineplot(x=['Start', 'Mid', 'End'], y=t_w_norm, marker='o', ax=ax2, color='orange', label='Attention')
                        else:
                            sns.lineplot(data=t_w_norm, ax=ax2, color='orange', label='Attention')
                        ax2.set_title(f"Temporal Attention ({best_view_name})")
                
                if not use_attention:
                    motion_diff = []
                    for t in range(1, num_frames):
                        diff = np.abs(slice_data[t] - slice_data[t-1]).mean()
                        motion_diff.append(diff)
                    
                    if len(motion_diff) > 0:
                        motion_diff = np.array(motion_diff)
                        motion_norm = (motion_diff - motion_diff.min()) / (motion_diff.max() - motion_diff.min() + 1e-6)
                        x_axis = range(1, num_frames)
                        sns.lineplot(x=x_axis, y=motion_norm, ax=ax2, color='green', linewidth=2, label='Pixel Motion')
                        ax2.fill_between(x_axis, motion_norm, alpha=0.2, color='green')
                        ax2.set_title(f"Heart Motion Cycle ({best_view_name})")
                        ax2.set_xlabel("Frame Index")
                        ax2.set_ylabel("Motion Intensity")
                    else:
                        ax2.text(0.5, 0.5, "Not enough frames", ha='center')
            
            except Exception as e:
                print(f"  - 运动曲线错误: {e}")
                ax2.text(0.5, 0.5, "Error plotting motion")
            
            # --- C. Grad-CAM ---
            ax3 = plt.subplot(2, 3, 3)
            ax4 = plt.subplot(2, 3, 4)
            ax5 = plt.subplot(2, 3, 5)
            
            try:
                patient_view_tensor = inputs[best_view_idx][i]
                slice_idx = 0
                frame_idx = patient_view_tensor.shape[1] // 2
                
                raw_frame = patient_view_tensor[slice_idx, frame_idx]
                if raw_frame.dim() == 2:
                    target_img_tensor = raw_frame.unsqueeze(0).unsqueeze(0).to(device)
                elif raw_frame.dim() == 3:
                    target_img_tensor = raw_frame.unsqueeze(0).to(device)
                
                if target_img_tensor.dim() == 5: target_img_tensor = target_img_tensor.squeeze(2)
                
                target_layers = [model.backbone.layers[-1].blocks[-1].norm1]
                cam = GradCAM(model=model.backbone, target_layers=target_layers, reshape_transform=reshape_transform)
                grayscale_cam = cam(input_tensor=target_img_tensor, targets=None)
                grayscale_cam = grayscale_cam[0, :]
                
                img_np = target_img_tensor.squeeze().cpu().numpy()
                img_min, img_max = img_np.min(), img_np.max()
                if img_max - img_min > 1e-6:
                    img_np_norm = (img_np - img_min) / (img_max - img_min)
                else:
                    img_np_norm = img_np
                img_rgb = np.stack([img_np_norm]*3, axis=2)
                
                ax3.imshow(img_np_norm, cmap='gray')
                ax3.set_title(f"{best_view_name} (Mid Frame)")
                ax3.axis('off')
                
                visualization = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True)
                ax4.imshow(visualization)
                ax4.set_title(f"Grad-CAM Overlay")
                ax4.axis('off')
                
                norm = mcolors.Normalize(vmin=0, vmax=1)
                sm = cm.ScalarMappable(cmap='jet', norm=norm)
                sm.set_array([])
                cbar4 = fig.colorbar(sm, ax=ax4, fraction=0.046, pad=0.04)
                cbar4.ax.tick_params(labelsize=8)
                
                im = ax5.imshow(grayscale_cam, cmap='jet', vmin=0, vmax=1)
                ax5.set_title("Attention Mask")
                ax5.axis('off')
                
                cbar5 = fig.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)
                cbar5.ax.tick_params(labelsize=8)
            
            except Exception as e:
                ax4.text(0.5, 0.5, "Grad-CAM Failed", ha='center')
                ax4.axis('off')
                ax5.axis('off')
                print(f"  - Grad-CAM Error: {e}")
            
            # --- D. 信息摘要 ---
            ax6 = plt.subplot(2, 3, 6)
            ax6.axis('off')
            
            info_text = f"Model: Video-Swin\n"
            info_text += f"----------------------\n"
            info_text += f"Patient ID:\n{patient_id}\n\n"
            
            if pred_label == true_label:
                status = "CORRECT"
                color_code = 'green'
            else:
                status = "WRONG"
                color_code = 'red'
            
            info_text += f"Ground Truth: {true_label}\n"
            info_text += f"Prediction:   {pred_label}\n"
            info_text += f"Confidence:   {conf:.2%}\n"
            info_text += f"Status:       {status}\n\n"
            info_text += f"Best View:    {best_view_name}\n"
            
            ax6.text(0.05, 0.5, info_text, fontsize=12, va='center', family='monospace',
                     bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))
            
            plt.tight_layout()
            if save_dir:
                save_path = os.path.join(save_dir, f"{patient_id}_viz.png")
                plt.savefig(save_path)
                # print(f"保存至: {save_path}")
            plt.close(fig) # 关闭画布释放内存

            # --- 更新全局钩子指针 ---
            # 计算当前病人消耗了多少个钩子记录 (每个视图 min(5, slices) 个)
            slices_consumed = 0
            for v_idx in range(len(views)):
                v_tensor = inputs[v_idx][i]
                slices_consumed += min(5, v_tensor.shape[0])
            global_hook_ptr += slices_consumed
        
        recorder.remove()

    def visualize_test_samples(self, model, test_dataloader, device, results_dir, num_samples=5):
        """可视化测试集中的样本"""
        print("\n=== 开始可视化测试样本 ===")
        
        model.eval()
        test_iter = iter(test_dataloader)
        visual_dir = os.path.join(results_dir, 'visualize_inference')
        os.makedirs(visual_dir, exist_ok=True)
        # for i in range(min(num_samples, len(test_dataloader))):
        for i in range(num_samples):
            try:
                batch = next(test_iter)
                self.visualize_inference(model, batch, device, save_dir=visual_dir)
                print(f"已完成 {i+1}/{num_samples} 个batch样本的可视化")
            except StopIteration:
                break
            except Exception as e:
                print(f"可视化样本 {i+1} 时出错: {e}")
                continue
        
        print(f"\n🎉 可视化完成! 结果保存到: {visual_dir}")

# ==================== 训练函数 ====================
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine_decay)
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_model(parent_results_dir=None,data_path="/datasets",model_type='swin',batch_size=4, image_size=224,num_epochs=100,
                train_dataloader=None,val_dataloader=None,test_dataloader=None,train_dataset=None, val_dataset=None, test_dataset=None,grad_accum_steps=4,patience=15, learning_rate=1e-4,weight_decay= 1e-4,pretrained=False,visualize_test=True, num_visualize=5, **kwargs):
    """训练模型的通用函数"""
    # config = {
    #     'data_path': "/data/chengyuxi/datasets/CMR/BatchData-1st",
    #     'batch_size': 4,
    #     'grad_accum_steps': 4,
    #     'image_size': 224,
    #     'num_epochs': 100,
    #     'learning_rate': 1e-4,
    #     'weight_decay': 1e-4,
    #     'model_name': None,
    #     'pretrained': False,
    #     'patience': 15,
    # }
    
    # config.update(kwargs)
    
    current_file = os.path.splitext(os.path.basename(__file__))[0]
    if parent_results_dir is None:
        parent_results_dir = f"{current_file}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    os.makedirs(parent_results_dir, exist_ok=True)
    
    results_dir = os.path.join(parent_results_dir, f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(results_dir, exist_ok=True)

    print(f"📂 父目录: {parent_results_dir}")
    print(f"📁 结果将保存到: {results_dir}")

    
    print(f"=== 训练 {model_type.upper()} 模型 ===")
    
    # print("=== 初始化数据集 ===")
    # full_dataset = CardiacThreeClassDataset(
    #     root_dir=data_path,
    #     num_frames=25,
    #     min_views=2,
    #     image_size=image_size
    # )

    # from torch.utils.data import Subset
    # # 只取前100个样本
    # original_dataset = full_dataset  # 保存原始数据集引用
    # full_dataset = Subset(original_dataset, indices=list(range(100)))
    
    # if len(full_dataset) == 0:
    #     print("错误: 没有有效数据!")
    #     return None
    
    # train_size = int(0.7 * len(full_dataset))
    # val_size = int(0.15 * len(full_dataset))
    # test_size = len(full_dataset) - train_size - val_size
    
    # generator = torch.Generator().manual_seed(42)
    # train_dataset, val_dataset, test_dataset = random_split(
    #     full_dataset, [train_size, val_size, test_size], generator=generator
    # )
    
    # collate_fn = lambda batch: cardiac_collate_fn(
    #     batch,
    #     max_views=4,
    #     image_size=image_size,
    #     num_frames=25
    # )
    
    # train_dataloader = DataLoader(
    #     train_dataset,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     num_workers=0,
    #     collate_fn=collate_fn
    # )
    
    # val_dataloader = DataLoader(
    #     val_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=0,
    #     collate_fn=collate_fn
    # )
    
    # test_dataloader = DataLoader(
    #     test_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=0,
    #     collate_fn=collate_fn
    # )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")


    
    print(f"\n创建 {model_type} 模型...")
    model = ModelFactory.create_model(
        model_type=model_type,
        num_classes=3,
        # model_name=model_name,
        pretrained=pretrained
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    print("\n测试模型输出形状...")
    with torch.no_grad():
        test_batch = next(iter(train_dataloader))
        test_output = model(test_batch)
        print(f"测试批次标签形状: {test_batch['label'].shape}")
        print(f"模型输出形状: {test_output.shape}")
        
        assert test_output.shape[0] == test_batch['label'].shape[0]
        assert test_output.shape[1] == 3
        print("✅ 模型输出形状测试通过")
    
    # ==================== 新增：保存模型结构摘要 ====================
    # 3. 输出模型结构 (TorchNetViz / TorchInfo)
    print("\n=== 模型结构摘要 ===")
    # 构建一个假的输入 [1, 1, 224, 224] 给 summary 看 backbone 结构
    try:
        model_summary=summary(model.backbone, input_data=torch.randn(1, 1, image_size, image_size).to(device), 
                col_names=["input_size", "output_size", "num_params"], depth=2)
        # 保存到文件
        summary_file = os.path.join(results_dir, "model_architecture.txt")
        with open(summary_file, "a") as f:
            f.write(str(model_summary))
    except:
        print("Torchinfo summary 失败，可能需要调整 input size")


    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    num_training_steps = num_epochs * len(train_dataloader) // grad_accum_steps
    num_warmup_steps = num_training_steps // 5
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    evaluator = ModelEvaluator(class_names=['NC', 'HCM', 'DCM'])
    
    print("=" * 80)
    print(f"开始训练 {model_type.upper()} 模型")
    print("=" * 80)
    print(f"训练集: {len(train_dataset)} 个样本")
    print(f"验证集: {len(val_dataset)} 个样本")
    print(f"测试集: {len(test_dataset)} 个样本")
    print(f"总训练轮数: {num_epochs}")
    print(f"结果目录: {results_dir}")
    
    train_start_time = time.time()
    best_val_accuracy = 0.0
    no_improve_count = 0
    
    train_history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rate': []
    }
    
    config_info = {
        'model_type': model_type,
        # 'model_name': model_name,
        'pretrained': pretrained,
        'batch_size': batch_size,
        'grad_accum_steps': grad_accum_steps,
        'image_size': image_size,
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'patience': patience,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'test_samples': len(test_dataset),
        'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'device': str(device),
        'results_dir': results_dir,
        'parent_results_dir': parent_results_dir
    }
    
    with open(f'{results_dir}/training_config.txt', 'w') as f:
        for key, value in config_info.items():
            f.write(f"{key}: {value}\n")
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f'\nEpoch {epoch + 1:02d}/{num_epochs} 开始...')
        
        model.train()
        train_total_loss = 0
        train_correct = 0
        train_total = 0
        accum_loss = 0

        batch_start_time = time.time()
        
        for batch_idx, batch in enumerate(train_dataloader):
            try:
                labels = batch['label'].to(device)
                
                with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                    outputs = model(batch)
                    loss = criterion(outputs, labels)
                
                loss = loss / grad_accum_steps
                loss.backward()
                
                accum_loss += loss.item() * grad_accum_steps
                train_total_loss += loss.item() * grad_accum_steps
                
                _, predicted = torch.max(outputs, 1)
                train_correct += (predicted == labels).sum().item()
                train_total += labels.size(0)

                if (batch_idx + 1) % 2 == 0:
                    batch_time = time.time() - batch_start_time
                    current_batch_loss_display = loss.item() * grad_accum_steps
                    batch_accuracy = 100 * (predicted == labels).float().mean().item()

                    print(f'  Batch {batch_idx + 1:03d}/{len(train_dataloader)} | '
                          f'Loss: {current_batch_loss_display:.4f} | Acc: {batch_accuracy:5.1f}% | '
                          f'Time: {batch_time:.1f}s')
                    batch_start_time = time.time()


                if (batch_idx + 1) % grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    current_lr = optimizer.param_groups[0]['lr']

                    print(f'  [Grad Step] LR: {current_lr:.2e} | '
                          f'Accum Loss: {accum_loss / grad_accum_steps:.4f}')
                   
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
        
        if accum_loss > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            print(f'  [Final Grad Step] 最后累积步骤更新完成')

        
        train_accuracy = 100 * train_correct / train_total if train_total > 0 else 0
        train_avg_loss = train_total_loss / (batch_idx + 1)
        current_lr = optimizer.param_groups[0]['lr']

        train_time = time.time() - epoch_start_time

        print(f'  开始验证...')
        val_start_time = time.time()
        val_results = evaluator.evaluate_model(model, val_dataloader, device, criterion, "验证")
        val_time = time.time() - val_start_time

        
        train_history['epoch'].append(epoch + 1)
        train_history['train_loss'].append(train_avg_loss)
        train_history['train_acc'].append(train_accuracy)
        train_history['val_loss'].append(val_results['loss'])
        train_history['val_acc'].append(val_results['accuracy'])
        train_history['learning_rate'].append(current_lr)
        
        epoch_time = time.time() - epoch_start_time

        if device.type == 'cuda':
            gpu_memory = torch.cuda.memory_allocated() / 1024 ** 3
            gpu_memory_max = torch.cuda.max_memory_allocated() / 1024 ** 3
            memory_info = f" | GPU内存: {gpu_memory:.2f}GB (峰值: {gpu_memory_max:.2f}GB)"
        else:
            memory_info = ""

        elapsed_time = time.time() - train_start_time
        progress = (epoch + 1) / num_epochs * 100

        if epoch > 0:
            estimated_total_time = elapsed_time / progress * 100
            estimated_remaining = estimated_total_time - elapsed_time
            eta = datetime.now() + timedelta(seconds=estimated_remaining)
            time_info = f" | 进度: {progress:5.1f}% | 剩余: {estimated_remaining / 60:6.1f}min | ETA: {eta.strftime('%m-%d %H:%M')}"
        else:
            time_info = f" | 进度: {progress:5.1f}%"


        print("-" * 80)
        print(f'Epoch {epoch + 1:02d} 完成 | 用时: {epoch_time:6.1f}s')
        print(f'  训练损失: {train_avg_loss:.4f} | 训练准确率: {train_accuracy:6.2f}%')
        print(f'  验证损失: {val_results["loss"]:.4f} | 验证准确率: {val_results["accuracy"]:6.2f}%')
        print(f'  学习率: {current_lr:.2e}')
        
        if val_results['accuracy'] > best_val_accuracy:
            best_val_accuracy = val_results['accuracy']
            no_improve_count = 0
            best_model_path = f'{results_dir}/{model_type}_best.pth'
            torch.save(model.state_dict(), best_model_path)
            print(f'      🎯 新的最佳验证准确率! 模型已保存到: {best_model_path}')
        else:
            no_improve_count += 1
            print(f'      ⏳ 验证准确率未提升 ({no_improve_count}/{patience})')
            if no_improve_count >= patience:
                print(f"      ⏹️  早停: {patience} 轮验证集无改善")
                break

        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_history': train_history,
                'best_val_accuracy': best_val_accuracy
            }
            checkpoint_path = f'{results_dir}/{model_type}_checkpoint_epoch_{epoch + 1}.pth'
            torch.save(checkpoint, checkpoint_path)
            print(f'      💾 检查点已保存: {checkpoint_path}')


    total_training_time = time.time() - train_start_time
    end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    print("\n" + "=" * 80)
    print("训练完成!")
    print("=" * 80)
    print(f"总训练时间: {total_training_time / 60:.1f} 分钟")
    print(f"总训练轮数: {epoch + 1}")
    print(f"最佳验证准确率: {best_val_accuracy:.2f}%")
    
    config_info['end_time'] = end_time
    config_info['total_training_time_minutes'] = total_training_time / 60
    config_info['best_val_accuracy'] = best_val_accuracy
    config_info['total_epochs_trained'] = epoch + 1
    
    with open(f'{results_dir}/training_config.txt', 'w') as f:
        for key, value in config_info.items():
            f.write(f"{key}: {value}\n")

    print("\n开始最终测试...")
    try:
        best_model_path = f'{results_dir}/{model_type}_best.pth'
        model.load_state_dict(torch.load(best_model_path))
        test_results = evaluator.evaluate_model(model, test_dataloader, device, criterion, "最终测试")
        
        config_info['test_accuracy'] = test_results['accuracy']
        
        print("\n生成评估报告...")
        evaluator.generate_classification_report(test_results, f"{results_dir}/final_results")
        evaluator.plot_confusion_matrix(test_results, f"{results_dir}/final_results")
        evaluator.plot_roc_curves(test_results, f"{results_dir}/final_results")
        evaluator.generate_detailed_metrics(test_results, f"{results_dir}/final_results")
        evaluator.save_prediction_probabilities(test_results, f"{results_dir}/final_results")
        
        print(f"\n最终测试准确率: {test_results['accuracy']:.2f}%")
        
        # 可视化测试样本（如果开启）
        if visualize_test:
            print("测试可视化")
            evaluator.visualize_test_samples(model, test_dataloader, device, results_dir, len(test_dataloader))
    
    except Exception as e:
        print(f"最终测试失败: {e}")
        config_info['test_accuracy'] = 'Failed'
    
    try:
        history_df = pd.DataFrame(train_history)
        history_path = f'{results_dir}/training_history.csv'
        history_df.to_csv(history_path, index=False)
        print(f"训练历史已保存: {history_path}")

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(train_history['epoch'], train_history['train_loss'], label='train_loss')
        plt.plot(train_history['epoch'], train_history['val_loss'], label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
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
        
        print(f"训练曲线已保存: {results_dir}/training_curves.png")
    
    
    except Exception as e:
        print(f"保存训练历史失败: {e}")


    
    print(f"\n🎉 所有结果已保存到: {results_dir}")
    print(f"完成时间: {end_time}")

    print(f"\n📂 结果目录内容:")
    for item in os.listdir(results_dir):
        print(f"  - {item}")
    
    return config_info

# ==================== 对比实验函数 ====================
def run_experiments(data_path="/datasets",batch_size=4, image_size=224,num_epochs=100,train_dataloader=None,val_dataloader=None,test_dataloader=None,train_dataset=None, val_dataset=None, test_dataset=None,grad_accum_steps=4,
               patience=15, learning_rate=1e-4,weight_decay= 1e-4,pretrained=False,visualize_test=True, 
               num_visualize=5, **kwargs):
    """运行多个实验进行对比"""
    experiments = [
        {'model_type': 'swin', 'model_name': 'swin_tiny_patch4_window7_224', 'visualize_test': True},
        {'model_type': 'cnn', 'model_name': 'resnet50', 'visualize_test': True},
        {'model_type': 'efficientnet', 'model_name': 'efficientnet_b0', 'visualize_test': True},
        {'model_type': 'densenet121', 'model_name': 'densenet121', 'visualize_test': True},
    ]
    
    current_file = os.path.splitext(os.path.basename(__file__))[0]
    parent_results_dir = f"{current_file}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"📂 实验父目录: {parent_results_dir}")
    
    results = []
    
    for i, exp in enumerate(experiments):
        print(f"\n{'=' * 60}")
        print(f"开始实验 {i + 1}/{len(experiments)}: {exp['model_type']}")
        print(f"{'=' * 60}")
        
        try:
            config = train_model(
                parent_results_dir=parent_results_dir,
                data_path=data_path,
                model_type=exp['model_type'],
                batch_size=batch_size,
                image_size=image_size,
                num_epochs=num_epochs,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                test_dataloader=test_dataloader,
                train_dataset=train_dataset, 
                val_dataset=val_dataset, 
                test_dataset=test_dataset,
                # model_name=args.model_name,
                grad_accum_steps=grad_accum_steps,
                patience=patience,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                pretrained=pretrained,
                visualize_test=exp["visualize_test"],            
                num_visualize=num_visualize
            )
            
            results.append({
                'model_type': exp['model_type'],
                'model_name': exp.get('model_name', 'default'),
                'results_dir': config['results_dir'],
                'best_val_accuracy': config['best_val_accuracy'],
                'test_accuracy': config.get('test_accuracy', 'N/A'),
                'total_params': config['total_params'],
                'training_time': config['total_training_time_minutes'],
            })
        except Exception as e:
            print(f"实验失败: {e}")
            results.append({
                'model_type': exp['model_type'],
                'model_name': exp.get('model_name', 'default'),
                'error': str(e)
            })
        
        time.sleep(1)
    
    results_df = pd.DataFrame(results)
    comparison_path = os.path.join(parent_results_dir, "experiment_comparison.csv")
    results_df.to_csv(comparison_path, index=False)
    
    print(f"\n{'=' * 60}")
    print("实验结果对比")
    print(f"{'=' * 60}")
    
    print(results_df.to_string(index=False))
    print(f"\n实验结果对比已保存到: {comparison_path}")
    
    if len(results) > 1 and all('error' not in r for r in results):
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        models = [r['model_type'] for r in results]
        val_acc = [r['best_val_accuracy'] for r in results]
        
        bars = plt.bar(models, val_acc, color=['skyblue', 'lightgreen', 'salmon', 'gold'])
        plt.title('Best Validation Accuracy Comparison')
        plt.ylabel('Accuracy (%)')
        plt.xticks(rotation=45)
        
        for bar, acc in zip(bars, val_acc):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f'{acc:.1f}%', ha='center', va='bottom')
        
        plt.subplot(1, 2, 2)
        train_time = [r['training_time'] for r in results]
        bars = plt.bar(models, train_time, color=['skyblue', 'lightgreen', 'salmon', 'gold'])
        plt.title('Training Time Comparison')
        plt.ylabel('Time (minutes)')
        plt.xticks(rotation=45)
        
        for bar, t in zip(bars, train_time):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f'{t:.1f}m', ha='center', va='bottom')
        
        plt.tight_layout()
        comparison_plot_path = os.path.join(parent_results_dir, "experiment_comparison.png")
        plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
        print(f"对比图表已保存到: {comparison_plot_path}")
    
    print(f"\n🎉 所有实验完成! 结果保存在: {parent_results_dir}")
    return results


# ==================== 主函数 ====================
def main():
    """主函数，支持多种运行模式"""
    import argparse
    
    parser = argparse.ArgumentParser(description='心脏影像分类系统')
    parser.add_argument('--mode', type=str, default='experiment', 
                       choices=['train', 'experiment'],
                       help='运行模式: train(训练单个模型), experiment(对比实验)')
    parser.add_argument('--model', type=str, default='swin',
                       choices=['swin', 'cnn', 'resnet', 'efficientnet',
                                'densenet', 'vgg', 'convnext'],
                       help='选择模型类型')
    parser.add_argument('--model_name', type=str, default=None,
                       help='具体模型名称')
    # parser.add_argument('--visualize_test', action='store_true',
    #                    help='训练完成后可视化测试样本')
    parser.add_argument('--visualize_test',type=bool,default=True,
                    help='训练完成后可视化测试样本')
    parser.add_argument('--num_visualize', type=int, default=5,
                       help='可视化样本数量')
    parser.add_argument('--pretrained', action='store_true',default=False,
                       help='使用预训练权重,action如果指定了这个参数，它的值就是 True,否则false')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='批次大小')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--image_size', type=int, default=224,
                       help='图像大小')
    parser.add_argument('--grad_accum_steps', type=int, default=4,
                       help='梯度累积步数')
    parser.add_argument('--patience', type=int, default=15,
                       help='早停耐心值')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='权重衰减系数')
    parser.add_argument('--data_path', type=str,
                       default="/data/chengyuxi/datasets/CMR/BatchData-1st",
                       help='数据路径')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU ID')
    
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print(f"使用GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")



    print("=== 初始化数据集 ===")
    full_dataset = CardiacThreeClassDataset(
        root_dir=args.data_path,
        num_frames=25,
        min_views=2,
        image_size=args.image_size
    )

    # from torch.utils.data import Subset
    # # 只取前100个样本
    # original_dataset = full_dataset  # 保存原始数据集引用
    # full_dataset = Subset(original_dataset, indices=list(range(20)))
    
    if len(full_dataset) == 0:
        print("错误: 没有有效数据!")
        return None
    
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )
    
    collate_fn = lambda batch: cardiac_collate_fn(
        batch,
        max_views=4,
        image_size=args.image_size,
        num_frames=25
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    if args.mode == 'train':
        print(f"=== 训练模式: {args.model.upper()} 模型 ===")


        config = train_model(
            data_path=args.data_path,
            model_type=args.model,
            batch_size=args.batch_size,
            image_size=args.image_size,
            num_epochs=args.epochs,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader,
            train_dataset=train_dataset, 
            val_dataset=val_dataset, 
            test_dataset=test_dataset,
            # model_name=args.model_name,
            grad_accum_steps=args.grad_accum_steps,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            pretrained=args.pretrained,
            visualize_test=args.visualize_test,            
            num_visualize=args.num_visualize,
        )
        
        print(f"\n{'=' * 60}")
        print(f"训练完成总结 - {args.model.upper()} 模型")
        print(f"{'=' * 60}")
        if config:
            print(f"最佳验证准确率: {config['best_val_accuracy']:.2f}%")
            print(f"测试准确率: {config.get('test_accuracy', 'N/A')}")
            print(f"训练轮数: {config['total_epochs_trained']}")
            print(f"训练时间: {config['total_training_time_minutes']:.1f} 分钟")
            print(f"模型参数量: {config['total_params']:,}")
    
    elif args.mode == 'experiment':
        print(f"=== 对比实验模式 ===")
        run_experiments(
            data_path=args.data_path,
            batch_size=args.batch_size,
            image_size=args.image_size,
            num_epochs=args.epochs,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader,
            train_dataset=train_dataset, 
            val_dataset=val_dataset, 
            test_dataset=test_dataset,
            # model_name=args.model_name,
            grad_accum_steps=args.grad_accum_steps,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            pretrained=args.pretrained,
            visualize_test=args.visualize_test,            
            num_visualize=args.num_visualize,
            )


if __name__ == "__main__":
    # 使用示例:
    # 1. 训练单个模型并可视化测试样本: python integrated_system.py --mode train --model swin --visualize_test
    # 2. 运行对比实验: python integrated_system.py --mode experiment
    # 3. 训练CNN模型: python integrated_system.py --mode train --model cnn
    
    main()
