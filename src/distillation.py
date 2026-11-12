"""Scale-aware region-weighted spatial feature-response distillation.

Channel alignment is parameter-free attention transfer, identical in A1--A4.
Inputs are post-augmentation normalized XYWH boxes from Ultralytics batches.
"""
from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn.functional as F


@dataclass
class KDConfig:
    variant: str = 'A4'
    coefficient: float = .5
    background: float = .1
    foreground: float = 1.
    reference_scale: float = 32.
    scale_power: float = 1.
    min_weight: float = .5
    max_weight: float = 4.
    layer_weights: Tuple[float, float, float] = (.5, .3, .2)
    eps: float = 1.e-6

    def __post_init__(self):
        if self.variant not in ('A1','A2','A3','A4'): raise ValueError(self.variant)
        if self.coefficient < 0: raise ValueError('coefficient must be nonnegative')
        if not (0 < self.background <= self.foreground): raise ValueError('invalid region weights')
        if not (0 < self.min_weight <= self.max_weight): raise ValueError('invalid scale clip')
        if len(self.layer_weights) != 3 or min(self.layer_weights) < 0 or sum(self.layer_weights) <= 0:
            raise ValueError('invalid layer weights')


def scale_weights(boxes, image_hw, cfg):
    """Object equivalent side in pixels of the actual augmented training tensor."""
    h,w = image_hw
    scale = (boxes[:,2].float().clamp_min(0)*w * boxes[:,3].float().clamp_min(0)*h).sqrt()
    return (cfg.reference_scale / scale.clamp_min(cfg.eps)).pow(cfg.scale_power).clamp(cfg.min_weight,cfg.max_weight)


def region_mask(boxes, batch_idx, batch_size, feature_hw, image_hw, cfg):
    h,w=feature_hw
    device=boxes.device
    if cfg.variant == 'A1': return torch.ones((batch_size,h,w),device=device,dtype=torch.float32)
    mask=torch.full((batch_size,h,w),cfg.background,device=device,dtype=torch.float32)
    weights=scale_weights(boxes,image_hw,cfg) if cfg.variant in ('A3','A4') else torch.ones(len(boxes),device=device)
    # Floor/ceil rasterization guarantees even sub-cell boxes cover at least one cell.
    xy1=(boxes[:,:2]-boxes[:,2:]/2).clamp(0,1)
    xy2=(boxes[:,:2]+boxes[:,2:]/2).clamp(0,1)
    factors=boxes.new_tensor([w,h])
    low=(xy1*factors).floor().long(); high=(xy2*factors).ceil().long()
    for n,(bi,lo,hi) in enumerate(zip(batch_idx.long().tolist(),low.tolist(),high.tolist())):
        x1,y1=lo; x2,y2=hi
        if not 0 <= bi < batch_size: raise ValueError('invalid batch index')
        if x2<=x1 or y2<=y1: continue
        target=mask[bi,y1:y2,x1:x2]
        mask[bi,y1:y2,x1:x2]=torch.maximum(target,cfg.foreground*weights[n])
    return mask


def attention_map(feature, eps=1.e-6):
    """FP32 energy map normalized to unit spatial RMS; channel-count invariant."""
    a=feature.float().square().mean(dim=1)
    return a / (a.square().mean(dim=(-2,-1),keepdim=True)+eps).sqrt()


def distillation_loss(student_features, teacher_features, batch, cfg):
    if len(student_features)!=3 or len(teacher_features)!=3: raise ValueError('Need P3/P4/P5')
    alpha = cfg.layer_weights if cfg.variant=='A4' else (1/3,1/3,1/3)
    alpha = [v/sum(alpha) for v in alpha]
    losses=[]
    for sf,tf in zip(student_features,teacher_features):
        if sf.shape[0]!=tf.shape[0]: raise ValueError('batch size mismatch')
        sa=attention_map(sf,cfg.eps); ta=attention_map(tf.detach(),cfg.eps)
        if ta.shape[-2:]!=sa.shape[-2:]:
            ta=F.interpolate(ta[:,None],size=sa.shape[-2:],mode='bilinear',align_corners=False)[:,0]
        mask=region_mask(batch['bboxes'],batch['batch_idx'],sf.shape[0],sf.shape[-2:],batch['img'].shape[-2:],cfg)
        per_image=(mask*(sa-ta).square()).sum(dim=(-2,-1))/mask.sum(dim=(-2,-1)).clamp_min(cfg.eps)
        losses.append(per_image.mean())
    total=sum(a*l for a,l in zip(alpha,losses))
    if not torch.isfinite(total): raise FloatingPointError('Nonfinite distillation loss')
    return total,torch.stack(losses).detach()
