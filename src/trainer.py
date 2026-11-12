"""Ultralytics 8.3.203 extension; all training-only state stays out of saved EMA."""
import json
from dataclasses import asdict
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from .distillation import KDConfig, distillation_loss


class FeatureCapture:
    def __init__(self): self.features=None
    def __call__(self,module,args):
        # Detect.forward replaces elements of its input list: copy the list now.
        self.features=list(args[0])


class DistilledCriterion:
    def __init__(self,base,teacher,capture,cfg):
        self.base,self.teacher,self.capture,self.cfg=base,teacher,capture,cfg
        self.stats=[]

    def __call__(self,preds,batch):
        det,items=self.base(preds,batch)
        sf=self.capture.features
        self.capture.features=None
        if self.cfg.coefficient==0:
            return det,items
        if sf is None: raise RuntimeError('Student features were not captured')
        tc=FeatureCapture()
        handle=self.teacher.model[-1].register_forward_pre_hook(tc)
        try:
            with torch.no_grad():
                self.teacher.eval()
                self.teacher(batch['img'])
        finally:
            handle.remove()
        kd,levels=distillation_loss(sf,tc.features,batch,self.cfg)
        b=batch['img'].shape[0]
        # Base loss is a vector in 8.3.203; trainer sums it. Add KD exactly once.
        total=det.sum()+b*self.cfg.coefficient*kd
        if not torch.isfinite(total): raise FloatingPointError('Nonfinite joint loss')
        self.stats.append([float(kd.detach()),float(items.sum()),*levels.cpu().tolist()])
        return total,items


class SARKDTrainer(DetectionTrainer):
    # Supplied by scripts/train.py; absent from Ultralytics' strict config namespace.
    kd_options=None
    teacher_path=None

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.add_callback('on_fit_epoch_end',self._log_kd)
        self.add_callback('on_train_batch_end',self._check_gradients)

    def _setup_train(self):
        super()._setup_train()
        if self.world_size>1: raise RuntimeError('This implementation is verified for single GPU only')
        cfg=KDConfig(**self.kd_options)
        self.teacher=YOLO(str(self.teacher_path)).model.float().to(self.device).eval()
        self.teacher.requires_grad_(False)
        capture=FeatureCapture()
        # EMA is already constructed above. No teacher, hooks, adapters or custom
        # criteria exist on EMA; checkpoints are ordinary DetectionModel objects.
        self._capture_handle=self.model.model[-1].register_forward_pre_hook(capture)
        self.kd_criterion=DistilledCriterion(self.model.criterion,self.teacher,capture,cfg)
        self.model.criterion=self.kd_criterion
        (self.save_dir/'kd_config.json').write_text(json.dumps(dict(teacher=str(self.teacher_path),**asdict(cfg)),indent=2),encoding='utf-8')
        assert not any(p.requires_grad for p in self.teacher.parameters())
        assert len(self.ema.ema.model[-1]._forward_pre_hooks)==0

    def _check_gradients(self,trainer):
        if any(p.grad is not None for p in self.teacher.parameters()): raise AssertionError('Teacher gradient detected')
        # AMP overflow is handled by GradScaler; reject nonfinite loss, log skips separately.
        if not torch.isfinite(self.loss): raise FloatingPointError('Nonfinite training loss')

    def _log_kd(self,trainer):
        values=self.kd_criterion.stats
        if values:
            means=torch.tensor(values).mean(0).tolist()
            row=dict(epoch=self.epoch+1,kd_loss=means[0],det_loss=means[1],levels=means[2:],
                     coefficient=self.kd_criterion.cfg.coefficient,amp_scale=float(self.scaler.get_scale()))
            with (self.save_dir/'kd_losses.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(row)+'\n')
            values.clear()
