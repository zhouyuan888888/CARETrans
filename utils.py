import torch
try:
    from apex import amp
    has_apex = True
except ImportError:
    amp = None
    has_apex = False

from timm.utils.clip_grad import dispatch_clip_grad

class ApexScalerAccum:
    state_dict_key = "amp"

    def __call__(self, loss, optimizer, clip_grad=None, clip_mode='norm', parameters=None, create_graph=False,
        update_grad=True):
        with amp.scale_loss(loss, optimizer) as scaled_loss: scaled_loss.backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None: dispatch_clip_grad(amp.master_params(optimizer), clip_grad, mode=clip_mode)
            optimizer.step()

    def state_dict(self):
        if 'state_dict' in amp.__dict__: return amp.state_dict()

    def load_state_dict(self, state_dict):
        if 'load_state_dict' in amp.__dict__:
            amp.load_state_dict(state_dict)

class NativeScalerAccum:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, clip_mode='norm', parameters=None, create_graph=False,
        update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)
                dispatch_clip_grad(parameters, clip_grad, mode=clip_mode)
            self._scaler.step(optimizer)
            self._scaler.update()
    def state_dict(self): return self._scaler.state_dict()
    def load_state_dict(self, state_dict): self._scaler.load_state_dict(state_dict)

class AverageMeter:
    def __init__(self, fmt="{val:.2f} ({avg:.2f})"):
        self.fmt = fmt
        self.reset()
    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.count = 0
    def update(self, value, n=1):
        self.val = float(value)
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self): return round(self.sum / self.count,2)+1e-12 if self.count else 0.0
    
    def __str__(self): return self.fmt.format(val=self.val, avg=self.avg)