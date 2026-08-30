import os
import glob
import tqdm
import random
import tensorboardX
import librosa
import librosa.filters
from scipy import signal
from os.path import basename
import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import trimesh
import mcubes
from rich.console import Console
from torch import dtype
from torch_ema import ExponentialMovingAverage
import pandas as pd
from packaging import version as pver
import imageio
import lpips
from scipy.spatial.transform import Rotation
from .syncer_loss import ContrastiveVideoLossV2, BSLoss, Filter


def custom_meshgrid(*args):
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


global n
n = 0


def blend_with_mask_cuda(src, dst, mask):
    global n
    src = src.permute(2, 0, 1)
    dst = dst.permute(2, 0, 1)
    mask = mask.unsqueeze(0)
    n += 1
    blended = src * mask + dst * (1 - mask)

    return blended.permute(1, 2, 0).detach().cpu().numpy()


def get_audio_features(features, att_mode, index):
    if att_mode == 0:
        return features[[index]]
    elif att_mode == 1:
        left = index - 8
        pad_left = 0
        if left < 0:
            pad_left = -left
            left = 0
        auds = features[left:index]
        if pad_left > 0:
            # pad audio when index < 8
            auds = torch.cat([torch.zeros(pad_left, *auds.shape[1:], device=auds.device, dtype=auds.dtype), auds],
                             dim=0)
        return auds
    elif att_mode == 2:
        left = index - 4
        right = index + 4
        pad_left = 0
        pad_right = 0
        if left < 0:
            pad_left = -left
            left = 0
        if right > features.shape[0]:
            pad_right = right - features.shape[0]
            right = features.shape[0]
        auds = features[left:right]
        if pad_left > 0:
            auds = torch.cat([torch.zeros_like(auds[:pad_left]), auds], dim=0)
        if pad_right > 0:
            auds = torch.cat([auds, torch.zeros_like(auds[:pad_right])], dim=0)  # [8, 16]
        return auds
    else:
        raise NotImplementedError(f'wrong att_mode: {att_mode}')


@torch.jit.script
def linear_to_srgb(x):
    return torch.where(x < 0.0031308, 12.92 * x, 1.055 * x ** 0.41666 - 0.055)


@torch.jit.script
def srgb_to_linear(x):
    return torch.where(x < 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


# from pytorch3d
def _angle_from_tan(
        axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
) -> torch.Tensor:
    """Extract euler angle from rotation matrix entries (pytorch3d)."""

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


def _index_from_letter(letter: str) -> int:
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    raise ValueError("letter must be either X, Y or Z.")


def matrix_to_euler_angles(matrix: torch.Tensor, convention: str = 'XYZ') -> torch.Tensor:
    """Convert rotation matrices to euler angles (pytorch3d)."""
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)


@torch.cuda.amp.autocast(enabled=False)
def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """Build single-axis rotation matrix (pytorch3d)."""

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


@torch.cuda.amp.autocast(enabled=False)
def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str = 'XYZ') -> torch.Tensor:
    """Convert euler angles to rotation matrices (pytorch3d)."""


    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]

    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])


@torch.cuda.amp.autocast(enabled=False)
def convert_poses(poses):
    out = torch.empty(poses.shape[0], 6, dtype=torch.float32, device=poses.device)
    out[:, :3] = matrix_to_euler_angles(poses[:, :3, :3])
    out[:, 3:] = poses[:, :3, 3]
    return out


@torch.cuda.amp.autocast(enabled=False)
def get_bg_coords(H, W, device):
    X = torch.arange(H, device=device) / (H - 1) * 2 - 1  # in [-1, 1]
    Y = torch.arange(W, device=device) / (W - 1) * 2 - 1  # in [-1, 1]
    xs, ys = custom_meshgrid(X, Y)
    bg_coords = torch.cat([xs.reshape(-1, 1), ys.reshape(-1, 1)], dim=-1).unsqueeze(0)  # [1, H*W, 2], in [-1, 1]
    return bg_coords


@torch.cuda.amp.autocast(enabled=False)
def get_rays(poses, intrinsics, H, W, N=-1, patch_size=1, rect=None):
    '''Return rays_o, rays_d, inds for pose/intrinsics grid.'''

    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics

    if rect is not None:
        xmin, xmax, ymin, ymax = rect
        N = (xmax - xmin) * (ymax - ymin)

    i, j = custom_meshgrid(torch.linspace(0, W - 1, W, device=device),
                           torch.linspace(0, H - 1, H, device=device))
    i = i.t().reshape([1, H * W]).expand([B, H * W]) + 0.5
    j = j.t().reshape([1, H * W]).expand([B, H * W]) + 0.5  # 512 ×512 -> 1×262144(512×512)

    results = {}

    if N > 0:
        N = min(N, H * W)
        if patch_size > 1:

            # patch sampling skips corners
            num_patch = N // (patch_size ** 2)
            inds_x = torch.randint(0, H - patch_size, size=[num_patch], device=device)
            inds_y = torch.randint(0, W - patch_size, size=[num_patch], device=device)
            inds = torch.stack([inds_x, inds_y], dim=-1)  # [np, 2]
            # inds = torch.stack([inds_x, inds_y], dim=-1)  # [np, 2]

            pi, pj = custom_meshgrid(torch.arange(patch_size, device=device), torch.arange(patch_size, device=device))
            offsets = torch.stack([pi.reshape(-1), pj.reshape(-1)], dim=-1)  # [p^2, 2]

            inds = inds.unsqueeze(1) + offsets.unsqueeze(0)  # [np, p^2, 2]
            inds = inds.view(-1, 2)  # [N, 2]
            inds = inds[:, 0] * W + inds[:, 1]  # [N], flatten

            inds = inds.expand([B, N])

        # lips rect ray sampling
        elif rect is not None:
            mask = torch.zeros(H, W, dtype=torch.bool, device=device)
            xmin, xmax, ymin, ymax = rect
            mask[xmin:xmax, ymin:ymax] = 1
            inds = torch.where(mask.view(-1))[0]
            inds = inds.unsqueeze(0)  # [1, N]

        else:
            inds = torch.randint(0, H * W, size=[N], device=device)  # may duplicate
            inds = inds.expand([B, N])

            # inds = inds.expand([B, N])
        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)


    else:
        inds = torch.arange(H * W, device=device).expand([B, H * W])

    results['i'] = i
    results['j'] = j
    results['inds'] = inds

    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)

    rays_d = directions @ poses[:, :3, :3].transpose(-1,
                                                     -2)

    rays_o = poses[..., :3, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d)  # [B, N, 3]

    results['rays_o'] = rays_o
    results['rays_d'] = rays_d

    return results


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def torch_vis_2d(x, renormalize=False):
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    if isinstance(x, torch.Tensor):
        if len(x.shape) == 3:
            x = x.permute(1, 2, 0).squeeze()
        x = x.detach().cpu().numpy()

    print(f'[torch_vis_2d] {x.shape}, {x.dtype}, {x.min()} ~ {x.max()}')

    x = x.astype(np.float32)

    if renormalize:
        x = (x - x.min(axis=0, keepdims=True)) / (x.max(axis=0, keepdims=True) - x.min(axis=0, keepdims=True) + 1e-8)

    plt.imshow(x)
    plt.show()


def extract_fields(bound_min, bound_max, resolution, query_func, S=128):
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(S)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(S)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(S)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)  # [S, 3]
                    val = query_func(pts).reshape(len(xs), len(ys),
                                                  len(zs)).detach().cpu().numpy()  # [S, 1] --> [x, y, z]
                    u[xi * S: xi * S + len(xs), yi * S: yi * S + len(ys), zi * S: zi * S + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    u = extract_fields(bound_min, bound_max, resolution, query_func)


    vertices, triangles = mcubes.marching_cubes(u, threshold)

    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


def ssim_1d_loss(pred, true, C1=1e-4, C2=9e-4):
    """1D SSIM between two signals."""
    if pred.size() != true.size():
        raise ValueError(f'Expected input size ({pred.size()}) to match target size ({true.size()}).')

    mu1 = pred.mean(dim=1, keepdim=True)
    mu2 = true.mean(dim=1, keepdim=True)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (pred * pred).mean(dim=1, keepdim=True) - mu1_sq
    sigma2_sq = (true * true).mean(dim=1, keepdim=True) - mu2_sq
    sigma12 = (pred * true).mean(dim=1, keepdim=True) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    ssim_val = ssim_map.mean()

    return ssim_val


def transform_semantic_target(coeff_3dmm, frame_index, semantic_radius):
    num_frames = coeff_3dmm.shape[0]
    seq = list(range(frame_index - semantic_radius, frame_index + semantic_radius + 1))
    index = [min(max(item, 0), num_frames - 1) for item in seq]
    coeff_3dmm_g = coeff_3dmm[index, :]
    return coeff_3dmm_g.transpose(1, 0)


class LSEMeter:
    def __init__(self, metric='contrast', net='alex', device=None):
        """LSE contrast/difference/lpips meter."""
        self.V = 0  # Accumulated value
        self.N = 0  # Number of updates
        self.metric = metric  # Metric type

        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if metric == 'lpips':
            self.lpips_fn = lpips.LPIPS(net=net).eval().to(self.device)

    def clear(self):
        """Reset accumulated stats."""
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        """Convert inputs to BCHW on device."""
        outputs = []
        for inp in inputs:
            if inp.dim() == 4:  # Assuming input shape [B, H, W, C]
                inp = inp.permute(0, 3, 1, 2).contiguous()  # Convert to [B, C, H, W]
            inp = inp.to(self.device)
            outputs.append(inp)
        return outputs

    def update(self, preds, truths):
        """Accumulate LSE from preds/truths."""
        preds, truths = self.prepare_inputs(preds, truths)

        if self.metric == 'contrast':
            contrast = F.cosine_similarity(preds, truths, dim=1)  # [B, H, W]
            logsumexp_value = torch.logsumexp(contrast.view(-1), dim=0)  # LSE over all pixels
        elif self.metric == 'difference':
            difference = F.mse_loss(preds, truths, reduction='none').mean(dim=(1, 2, 3))  # Per-sample MSE
            logsumexp_value = torch.logsumexp(difference, dim=0)  # LSE over batch
        elif self.metric == 'lpips':
            lpips_value = self.lpips_fn(truths, preds, normalize=True).mean()  # Average over batch
            logsumexp_value = torch.log(lpips_value + 1e-8)  # Prevent log(0)
        else:
            raise ValueError("Unsupported metric: use 'contrast', 'difference', or 'lpips'")

        self.V += logsumexp_value.item()
        self.N += 1

    def measure(self):
        """Average LSE value."""
        return self.V / self.N if self.N > 0 else 0.0

    def write(self, writer, global_step, prefix=""):
        """Log metric to tensorboard."""
        writer.add_scalar(os.path.join(prefix, f"LSE ({self.metric})"), self.measure(), global_step)

    def report(self):
        """Format metric string."""
        return f'LSE ({self.metric}) = {self.measure():.6f}'


class PSNRMeter:
    def __init__(self):
        self.V = 0
        self.N = 0

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths)  # [B, N, 3] or [B, H, W, 3], range in [0, 1]

        psnr = -10 * np.log10(np.mean((preds - truths) ** 2))

        self.V += psnr
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "PSNR"), self.measure(), global_step)

    def report(self):
        return f'PSNR = {self.measure():.6f}'


class LPIPSMeter:
    def __init__(self, net='alex', device=None):
        self.V = 0
        self.N = 0
        self.net = net

        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fn = lpips.LPIPS(net=net).eval().to(self.device)

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            inp = inp.permute(0, 3, 1, 2).contiguous()  # [B, 3, H, W]
            inp = inp.to(self.device)
            outputs.append(inp)
        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths)  # [B, H, W, 3] --> [B, 3, H, W], range in [0, 1]
        v = self.fn(truths, preds, normalize=True).item()  # normalize=True: [0, 1] to [-1, 1]
        self.V += v
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, f"LPIPS ({self.net})"), self.measure(), global_step)

    def report(self):
        return f'LPIPS ({self.net}) = {self.measure():.6f}'


class LMDMeter:
    def __init__(self, backend='dlib', region='mouth'):
        self.backend = backend
        self.region = region  # mouth or face

        if self.backend == 'dlib':
            import dlib

            self.predictor_path = './shape_predictor_68_face_landmarks.dat'
            if not os.path.exists(self.predictor_path):
                raise FileNotFoundError(
                    'Please download dlib checkpoint from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2')

            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(self.predictor_path)

        else:

            import face_alignment
            try:
                self.predictor = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
            except:
                self.predictor = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)

        self.V = 0
        self.N = 0

    def get_landmarks(self, img):

        if self.backend == 'dlib':
            dets = self.detector(img, 1)
            for det in dets:
                shape = self.predictor(img, det)
                lms = np.zeros((68, 2), dtype=np.int32)
                for i in range(0, 68):
                    lms[i, 0] = shape.part(i).x
                    lms[i, 1] = shape.part(i).y
                break

        else:
            lms = self.predictor.get_landmarks(img)[-1]

        lms = lms.astype(np.float32)

        return lms

    def vis_landmarks(self, img, lms):
        plt.imshow(img)
        plt.plot(lms[48:68, 0], lms[48:68, 1], marker='o', markersize=1, linestyle='-', lw=2)
        plt.show()

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            inp = inp.detach().cpu().numpy()
            inp = (inp * 255).astype(np.uint8)
            outputs.append(inp)
        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds[0], truths[0])  # [H, W, 3] numpy array

        lms_pred = self.get_landmarks(preds)
        lms_truth = self.get_landmarks(truths)

        if self.region == 'mouth':
            lms_pred = lms_pred[48:68]
            lms_truth = lms_truth[48:68]

        lms_pred = lms_pred - lms_pred.mean(0)
        lms_truth = lms_truth - lms_truth.mean(0)

        dist = np.sqrt(((lms_pred - lms_truth) ** 2).sum(1)).mean(0)

        self.V += dist
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, f"LMD ({self.backend})"), self.measure(), global_step)

    def report(self):
        return f'LMD ({self.backend}) = {self.measure():.6f}'


class Trainer(object):
    def __init__(self,
                 name,  # name of this experiment
                 opt,  # extra conf
                 model,  # network
                 criterion=None,  # loss function, if None, assume inline implementation in train_step
                 criterion_audio2attr=None,
                 optimizer=None,  # optimizer
                 ema_decay=None,  # if use EMA, set the decay
                 ema_update_interval=1000,  # update ema per $ training steps.
                 lr_scheduler=None,  # scheduler
                 metrics=[],
                 # metrics for evaluation, if None, use val_loss to measure performance, el
                 local_rank=0,  # which GPU am I
                 world_size=1,  # total num of GPUs
                 device=None,  # device to use, usually setting to None is OK. (auto choose device)
                 mute=False,  # whether to mute all print
                 fp16=False,  # amp optimize level
                 eval_interval=1,  # eval once every $ epoch
                 max_keep_ckpt=50,  # max num of saved ckpts in disk
                 workspace='workspace',  # workspace to save logs & ckpts
                 best_mode='min',  # the smaller/larger result, the better
                 use_loss_as_metric=True,  # use loss as the first metric
                 report_metric_at_train=False,  # also report metrics at training
                 use_checkpoint="latest",  # which ckpt to use at init time
                 use_tensorboardX=True,  # whether to use tensorboard for logging
                 scheduler_update_every_step=False,  # whether to call scheduler.step() after every train step
                 torso_dic=None
                 ):

        self.name = name
        self.opt = opt
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.ema_update_interval = ema_update_interval
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.flip_finetune_lips = self.opt.finetune_lips
        self.flip_init_lips = self.opt.init_lips
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(
            f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()
        self.torso_dic = torso_dic

        torch.cuda.set_device("cuda:0")
        model.cuda()
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        self.model = model

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion
        self.criterionL1 = nn.L1Loss().to(self.device)
        if criterion_audio2attr is not None:
            self.criterion_audio2attr = criterion_audio2attr.to(self.device)
        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4)  # naive adam
        else:
            self.optimizer = optimizer(self.model)

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1)  # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        if self.opt.patch_size > 1 or self.opt.finetune_lips or True:
            import lpips
            self.criterion_lpips_alex = lpips.LPIPS(net='alex').to(self.device)

        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [],  # metrics[0], or valid_loss
            "checkpoints": [],  # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
        }

        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            os.makedirs(self.ckpt_path, exist_ok=True)

        self.log(
            f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "latest_model":
                self.log("[INFO] Loading latest checkpoint (model only)...")
                self.load_checkpoint(model_only=True)
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else:  # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)

    def __del__(self):
        if self.log_ptr:
            self.log_ptr.close()

    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute:
                self.console.print(*args, **kwargs)
            if self.log_ptr:
                print(*args, file=self.log_ptr)
                self.log_ptr.flush()  # write immediately to file


    def train_step(self, data, preds=None):

        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]
        bg_coords = data['bg_coords']  # [1, N, 2]
        poses_label = data['poses']  # [B, 4, 4]
        face_mask = data['face_mask']  # [B, N]
        upface_mask = data['upface_mask']  # [B, N]
        lowface_mask = data['lowface_mask']  # [B, N]
        eye_mask = data['eye_mask']  # [B, N]
        lhalf_mask = data['lhalf_mask']
        eye_label = data['eye']  # [B, 1/7]
        auds = data['auds']  # [B, 29, 16]
        index = data['index']  # [B]
        Xhub_feature = data['Xhub_feature']  # [B, 256]
        bg_torso_img = data['bg_torso_img']
        bg_color_ori = data['bg_color_ori']
        images_ori = data["images_ori"]
        loss_perception = 0

        if not self.opt.torso:
            rgb = data['images']  # [B, N, 3]
        else:
            rgb = data['bg_torso_color']

        B, N, C = rgb.shape

        if self.opt.color_space == 'linear':
            rgb[..., :3] = srgb_to_linear(rgb[..., :3])

        bg_color = data['bg_color']

        if preds is not None:
            matrix_pose_nerf = preds[0]
            if matrix_pose_nerf is not None:
                matrix_pose = self.model.nerf_matrix_to_ngp_tensor(matrix_pose_nerf, scale=self.opt.scale,
                                                                   offset=self.opt.offset)
            else:
                matrix_pose = None  # unused in stage 2
            bs = preds[1]
            attr_loss = preds[2]
        else:
            raise Exception("preds is None")

        if self.opt.finetune_lips is True or self.opt.bs_start is True:
            eye = bs
        else:
            eye = eye_label

        if self.opt.torso or self.opt.finetune_lips:
            # use predicted pose
            poses = matrix_pose
            intrinsics = self.opt.intrinsics
            H = 512
            W = 512
            num_rays = self.opt.num_rays
            if self.opt.finetune_lips is not True:
                rays = get_rays(poses, intrinsics, H, W, num_rays, self.opt.patch_size)
            else:
                rect = data['rect']
                rays = get_rays(poses, intrinsics, H, W, -1, rect=rect)
            rays_o = rays['rays_o']  # [B, N, 3]
            rays_d = rays['rays_d']  # [B, N, 3]
            bg_coords = torch.gather(self.opt.bg_coords_ori, 1, torch.stack(2 * [rays['inds']], -1))  # [1, N, 2]
            if self.opt.torso is True:
                rgb = torch.gather(bg_torso_img, 1, torch.stack(3 * [rays['inds']], -1))  # [B, N, 3]
            else:
                C = images_ori.shape[-1]
                rgb = torch.gather(images_ori.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1))  # [B, N, 3/4]
            B, N, C = rgb.shape
            if self.opt.color_space == 'linear':
                rgb[..., :3] = srgb_to_linear(rgb[..., :3])
            bg_color = torch.gather(bg_color_ori, 1, torch.stack(3 * [rays['inds']], -1))  # [B, N, 3]
        else:
            poses = poses_label

        if not self.opt.torso:
            outputs = self.model.render(Xhub_feature,
                                        rays_o,
                                        rays_d,
                                        auds,
                                        bg_coords,
                                        poses=poses,
                                        eye=eye,
                                        index=index,
                                        staged=False,
                                        bg_color=bg_color,
                                        perturb=True,
                                        force_all_rays=False if (
                                                    self.opt.patch_size <= 1 and not self.opt.train_camera) else True,
                                        **vars(self.opt))
        else:
            outputs = self.model.render_torso(Xhub_feature, rays_o, rays_d, auds, bg_coords, poses=poses, eye=eye,
                                              index=index, staged=False, bg_color=bg_color, perturb=True,
                                              force_all_rays=False if (
                                                          self.opt.patch_size <= 1 and not self.opt.train_camera) else True,
                                              **vars(self.opt))

        if not self.opt.torso:
            pred_rgb = outputs['image']
        else:
            pred_rgb = outputs['torso_color']

        if not self.opt.torso:

            bs_loss = attr_loss

        if self.opt.torso:

            poses_loss = attr_loss

        # loss factor
        step_factor = min(self.global_step / self.opt.iters, 1.0)

        # MSE loss
        loss = self.criterion(pred_rgb, rgb).mean(-1)  # [B, N, 3] --> [B, N]

        if self.opt.torso:
            loss = loss.mean()
            loss += ((1 - self.model.anchor_points[:, 3]) ** 2).mean()
            loss += poses_loss
            return pred_rgb, rgb, loss, poses_loss

        if self.opt.unc_loss and not self.flip_finetune_lips:
            alpha = 0.2
            uncertainty = outputs['uncertainty']  # [N], abs sum
            beta = uncertainty + 1

            unc_weight = F.softmax(uncertainty, dim=-1) * N
            loss *= alpha + (1 - alpha) * ((1 - step_factor) + step_factor * unc_weight.detach()).clamp(0, 10)

            beta = uncertainty + 1
            norm_rgb = torch.norm((pred_rgb - rgb), dim=-1).detach()
            loss_u = norm_rgb / (2 * beta ** 2) + (torch.log(beta) ** 2) / 2
            loss_u *= face_mask.view(-1)

            loss += 0.01 * step_factor * loss_u

            loss_static_uncertainty = (uncertainty * (~face_mask.view(-1)))
            loss += 0.01 * step_factor * loss_static_uncertainty

        # patch LPIPS loss
        if self.opt.patch_size > 1 and not self.opt.finetune_lips:
            rgb = rgb.view(-1, self.opt.patch_size, self.opt.patch_size, 3).permute(0, 3, 1, 2).contiguous() * 2 - 1
            pred_rgb = pred_rgb.view(-1, self.opt.patch_size, self.opt.patch_size, 3).permute(0, 3, 1,
                                                                                              2).contiguous() * 2 - 1

            loss_lpips = self.criterion_lpips_alex(pred_rgb, rgb)

            loss = loss + 0.1 * loss_lpips

        # lips LPIPS loss
        if self.opt.finetune_lips:
            xmin, xmax, ymin, ymax = data['rect']
            rgb = rgb.view(-1, xmax - xmin, ymax - ymin, 3).permute(0, 3, 1, 2).contiguous() * 2 - 1
            pred_rgb = pred_rgb.view(-1, xmax - xmin, ymax - ymin, 3).permute(0, 3, 1, 2).contiguous() * 2 - 1

            padding_h = max(0, (32 - rgb.shape[-2] + 1) // 2)
            padding_w = max(0, (32 - rgb.shape[-1] + 1) // 2)

            if padding_w or padding_h:
                rgb = torch.nn.functional.pad(rgb, (padding_w, padding_w, padding_h, padding_h))
                pred_rgb = torch.nn.functional.pad(pred_rgb, (padding_w, padding_w, padding_h, padding_h))

            loss = loss + 0.01 * self.criterion_lpips_alex(pred_rgb, rgb)

        mask_lowface = False
        # alternate lips finetune steps
        if self.flip_finetune_lips:
            mask_lowface = True
            self.opt.finetune_lips = not self.opt.finetune_lips

        loss = loss.mean()

        if self.opt.patch_size > 1 and not self.opt.finetune_lips:
            if self.opt.pyramid_loss:
                loss = loss + 0.1 * loss_perception

        # entropy on weights_sum
        # entropy on weights_sum
        if self.opt.torso:
            alphas = outputs['torso_alpha'].clamp(1e-5, 1 - 1e-5)
            loss_ws = - alphas * torch.log2(alphas) - (1 - alphas) * torch.log2(1 - alphas)
            loss = loss + 1e-4 * loss_ws.mean()

        else:
            alphas = outputs['weights_sum'].clamp(1e-5, 1 - 1e-5)
            loss_ws = - alphas * torch.log2(alphas) - (1 - alphas) * torch.log2(1 - alphas)
            loss = loss + 1e-4 * loss_ws.mean()

        # ambient audio loss (static outside face)
        if self.opt.amb_aud_loss and not self.opt.torso:
            ambient_aud = outputs['ambient_aud']
            if mask_lowface:  # stage 3
                loss_amb_aud = (ambient_aud * (~lowface_mask.view(-1))).mean()
            else:  # stage 2
                loss_amb_aud = (ambient_aud * (~face_mask.view(-1))).mean()
            # ramp ambient loss weight
            lambda_amb = step_factor * self.opt.lambda_amb
            loss += lambda_amb * loss_amb_aud

        # ambient eye loss
        if self.opt.amb_eye_loss and not self.opt.torso:
            ambient_eye = outputs['ambient_eye']
            loss_cross = ((ambient_eye) * (face_mask.view(-1))).mean()
            lambda_amb = step_factor * self.opt.lambda_amb
            loss += lambda_amb * loss_cross

        # xyz jitter regularization
        if self.global_step % 16 == 0 and not self.flip_finetune_lips:
            xyzs, dirs, enc_a, ind_code, eye = outputs['rays']
            xyz_delta = (torch.rand(size=xyzs.shape, dtype=xyzs.dtype, device=xyzs.device) * 2 - 1) * 1e-3
            with torch.no_grad():
                sigmas_raw, rgbs_raw, ambient_aud_raw, ambient_eye_raw, unc_raw = self.model(xyzs, dirs, enc_a.detach(),
                                                                                             ind_code.detach(), eye)
            sigmas_reg, rgbs_reg, ambient_aud_reg, ambient_eye_reg, unc_reg = self.model(xyzs + xyz_delta, dirs,
                                                                                         enc_a.detach(),
                                                                                         ind_code.detach(), eye)

            lambda_reg = step_factor * 1e-5
            reg_loss = 0
            if self.opt.unc_loss:
                reg_loss += self.criterion(unc_raw, unc_reg).mean()
            if self.opt.amb_aud_loss:
                reg_loss += self.criterion(ambient_aud_raw, ambient_aud_reg).mean()
            if self.opt.amb_eye_loss:
                reg_loss += self.criterion(ambient_eye_raw, ambient_eye_reg).mean()

            loss += reg_loss * lambda_reg

        if self.opt.bs_loss is True:
            loss += (bs_loss).squeeze(0)

        return pred_rgb, rgb, loss, bs_loss

    def eval_step(self, data, Xhub_input_bs, avg=None, t=None, bs=None, pose_diff_true=None):

        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]
        bg_coords = data['bg_coords']  # [1, N, 2]
        poses = data['poses']  # [B, 7]

        images = data['images']  # [B, H, W, 3/4]
        if self.opt.portrait:
            images = data['bg_gt_images']
        auds = data['auds']
        index = data['index']  # [B]
        eye = data['eye']  # [B, 1]

        Xhub_feature = data['Xhub_feature']  # [B, 256]
        bg_torso_img = data['bg_torso_img']
        bg_color_ori = data['bg_color_ori']
        images_ori = data["images_ori"]

        B, H, W, C = images.shape

        if self.opt.color_space == 'linear':
            images[..., :3] = srgb_to_linear(images[..., :3])

        # eval bg color
        bg_color = data['bg_color']

        euler_angles_avg, translation_avg, avg_bs, avg_pose_bs = avg[0], avg[1], avg[2], avg[3]

        all_bs = bs.to(self.device)

        bs_data = [avg_bs, eye]
        pose_bs_data = [None, pose_diff_true]

        standar_gaussian_pose = torch.randn((1, self.opt.noise_dim_pose), dtype=torch.float32, device=self.device)
        standar_gaussian_bs = torch.randn((1, 256), dtype=torch.float32, device=self.device)
        standar_gaussian = [standar_gaussian_pose, standar_gaussian_bs]
        if self.opt.torso:
            output, matrix_pose_nerf, _, _ = self.model.forward_audio(poses_bs=pose_bs_data, bs=bs_data,
                                                                          audio_feat_pose=Xhub_feature,
                                                                          audio_feat_bs=Xhub_input_bs,
                                                                          external_code=standar_gaussian,
                                                                          transform=(
                                                                          euler_angles_avg, translation_avg, avg_bs), t=t,
                                                                          clip=True)
            matrix_pose = self.model.nerf_matrix_to_ngp_tensor(matrix_pose_nerf, scale=self.opt.scale,
                                                               offset=self.opt.offset)
        elif self.opt.finetune_lips:
            output, matrix_pose_nerf, bs, _, _ = self.model.forward_audio(poses_bs=pose_bs_data, bs=bs_data,
                                                                          audio_feat_pose=Xhub_feature,
                                                                          audio_feat_bs=Xhub_input_bs,
                                                                          external_code=standar_gaussian,
                                                                          transform=(
                                                                          euler_angles_avg, translation_avg, avg_bs), t=t,
                                                                          clip=True)
            matrix_pose = self.model.nerf_matrix_to_ngp_tensor(matrix_pose_nerf, scale=self.opt.scale,
                                                               offset=self.opt.offset)
        else:
            output, a, mu, logvar = self.model.forward_audio(poses_bs=pose_bs_data, bs=bs_data,
                                                                          audio_feat_pose=Xhub_feature,
                                                                          audio_feat_bs=Xhub_input_bs,
                                                                          external_code=standar_gaussian,
                                                                          transform=(
                                                                          euler_angles_avg, translation_avg, avg_bs), t=t,
                                                                          clip=True)
            bs = a


        if self.opt.special or self.opt.finetune_lips:
            eye = bs
        else:
            eye = eye
        if self.opt.torso or self.opt.finetune_lips:
            poses = matrix_pose
            intrinsics = self.opt.intrinsics
            H = 512
            W = 512
            num_rays = -1
            rays = get_rays(poses, intrinsics, H, W, num_rays, self.opt.patch_size)
            rays_o = rays['rays_o'].to(self.device)  # [B, N, 3]
            rays_d = rays['rays_d'].to(self.device)  # [B, N, 3]
            bg_coords = self.opt.bg_coords_ori  # [1, N, 2]
            bg_color = bg_color_ori


        outputs = self.model.render(Xhub_feature,
                                    rays_o,
                                    rays_d,
                                    auds,
                                    bg_coords,
                                    poses,
                                    eye=eye,
                                    index=index,
                                    staged=True,
                                    bg_color=bg_color,
                                    perturb=False, **vars(self.opt))

        pred_rgb = outputs['image'].reshape(B, H, W, 3)

        # images1 = Image.fromarray(image_array)
        # images1.save("output_image2.png")

        pred_depth = outputs['depth'].reshape(B, H, W)
        pred_ambient_aud = outputs['ambient_aud'].reshape(B, H, W)
        pred_ambient_eye = outputs['ambient_eye'].reshape(B, H, W)
        pred_uncertainty = outputs['uncertainty'].reshape(B, H, W)

        loss_raw = self.criterion(pred_rgb, images)
        loss = loss_raw.mean()

        return pred_rgb, pred_depth, pred_ambient_aud, pred_ambient_eye, pred_uncertainty, images, loss, loss_raw

    def test_step(self, data, avg=None, bg_color=None, perturb=False,
                  infer_feats=None, standar_gaussian=None, t=None, Xhub_input_bs=None, bs_norm=None, poses_norm=None,
                  cal=False):

        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]

        bg_coords = data['bg_coords']  # [1, N, 2]
        poses = data['poses']  # [B, 7]

        auds = data['auds']  # [B, 29, 16]
        index = data['index']
        H, W = data['H'], data['W']

        if self.opt.infer is not None:
            Xhub_feature = data['Xhub_features_infer']
            all_Xhub_feature = infer_feats
        else:
            Xhub_feature = data['Xhub_feature']  # [B, 256]

        bg_torso_img = data['bg_torso_img']
        bg_color_ori = data['bg_color_ori']
        images_ori = data["images_ori"]

        # fixed eye at test
        if self.opt.exp_eye and self.opt.fix_eye >= 0:
            eye = torch.FloatTensor([self.opt.fix_eye]).view(1, 1).to(self.device)
        else:
            eye = data['eye']  # [B, 1]

        if bg_color is not None:
            bg_color = bg_color.to(self.device)
        else:
            bg_color = data['bg_color']

        if avg is not None:
            euler_angles_avg, translation_avg, avg_bs, avg_pose_bs = avg[0], avg[1], avg[2], avg[3]
        else:
            raise Exception("avg is None")

        if infer_feats is not None:
            global all_bs, all_matrix_pose_narray
            if cal:
                standar_gaussian = standar_gaussian
                # torch
                bs_data = [avg_bs.repeat(standar_gaussian[0].shape[0], 1), bs_norm]


                poses_norm = poses_norm[0, :].repeat(poses_norm.shape[0], 1)
                pose_bs_data = [None, poses_norm]

                output, all_matrix_pose_nerf, all_bs, _, _ = self.model.forward_audio(poses_bs=pose_bs_data,
                                                                                      bs=bs_data,
                                                                                      audio_feat_pose=all_Xhub_feature,
                                                                                      audio_feat_bs=Xhub_input_bs,
                                                                                      external_code=standar_gaussian,
                                                                                      transform=(
                                                                                      euler_angles_avg, translation_avg,
                                                                                      avg_bs), t=t, clip=True)
                with open(os.path.join(self.workspace, 'blink_max'), 'rb') as f:
                    dic_ = pickle.load(f)
                blink_5_max = dic_["max_5"]
                blink_6_max = dic_["max_6"]
                filter = Filter(max_5=blink_5_max, max_6=blink_6_max)
                filtered_all_bs = filter(all_bs)
                all_bs = filtered_all_bs
                # all_bs[:, 5] = torch
                # all_bs[:, 6] = torch
                # all_bs[:, 6] = filtered_all_bs[:, 5]

                all_bs_array = all_bs.cpu().detach().numpy()
                # all_bs_array[:, 5] = np.zeros((all_bs_array.shape[0]))
                # all_bs_array[:, 6] = np.zeros((all_bs_array.shape[0]))
                plt.figure(figsize=(10, 5))

                plt.plot(all_bs_array[:, 5], color='red', label='left eye')
                plt.plot(all_bs_array[:, 6], color='blue', label='right eye')
                plt.title('Line Plot of Columns 6 and 7')
                plt.xlabel('Row Index')
                plt.ylabel('Values')
                plt.legend()
                plt.savefig(
                    os.path.join(self.workspace, f"line graph (FROM PRE INfer_{self.opt.infer.split('/')[-2]})"))
                plt.close()

                eye_bs_data = {
                    "Time": np.arange(0, all_bs_array.shape[0]),
                    "Left Eye": all_bs_array[:, 5],
                    "Right Eye": all_bs_array[:, 6]
                }
                df = pd.DataFrame(eye_bs_data)

                output_csv = os.path.join(self.workspace, f"all_pred_bs_eye_with_{self.opt.infer.split('/')[-2]}.csv")
                df.to_csv(output_csv, index=False)
                print(f'CSV file saved as {output_csv}')


                bs = all_bs[data["index"][0], :].unsqueeze(0)
                all_matrix_pose = self.model.nerf_matrix_to_ngp_tensor(all_matrix_pose_nerf,
                                                                       scale=self.opt.scale,
                                                                       offset=self.opt.offset)

                all_matrix_pose_narray = all_matrix_pose.cpu().detach().numpy()

                # smooth inferred poses
                N = all_matrix_pose_narray.shape[0]
                K = 32 // 2

                trans = all_matrix_pose_narray[:, :3, 3].copy()  # [N, 3]
                rots = all_matrix_pose_narray[:, :3, :3].copy()  # [N, 3, 3]

                for i in range(N):
                    start = max(0, i - K)
                    end = min(N, i + 1)
                    all_matrix_pose_narray[i, :3, 3] = trans[start:end].mean(0)
                    all_matrix_pose_narray[i, :3, :3] = Rotation.from_matrix(rots[start:end]).mean().as_matrix()

                matrix_pose = torch.tensor(all_matrix_pose_narray, dtype=torch.float32).to(self.device)[
                              data["index"][0], :, :].unsqueeze(0)
            else:
                bs = all_bs[data["index"][0], :].unsqueeze(0)
                matrix_pose = torch.tensor(all_matrix_pose_narray, dtype=torch.float32).to(self.device)[
                              data["index"][0], :, :].unsqueeze(0)

        else:
            # standar_gaussian = torch
            standar_gaussian = standar_gaussian
            # torch
            bs_data = [avg_bs, bs_norm]

            pose_bs_data = [None, poses_norm]

            pose_bs, matrix_pose_nerf, bs, _, _ = self.model.forward_audio(poses_bs=pose_bs_data,
                                                                           bs=bs_data,
                                                                           audio_feat_pose=Xhub_feature,
                                                                           audio_feat_bs=Xhub_input_bs,
                                                                           external_code=standar_gaussian,
                                                                           transform=(
                                                                               euler_angles_avg, translation_avg,
                                                                               avg_bs), t=t)

            matrix_pose = self.model.nerf_matrix_to_ngp_tensor(matrix_pose_nerf, scale=self.opt.scale,
                                                               offset=self.opt.offset)
        if self.opt.special:
            eye = bs
        else:
            eye = eye
        if self.opt.torso or self.opt.finetune_lips or self.opt.special:
            intrinsics = self.opt.intrinsics
            num_rays = -1
            bg_coords = self.opt.bg_coords_ori
            bg_color = bg_color_ori
            if self.opt.special and False:
                poses = matrix_pose
                rays_o = []
                rays_d = []
                for pose in poses:
                    rays = get_rays(pose, intrinsics, H, W, num_rays, self.opt.patch_size)
                    rays_o.append(rays['rays_o'])  # [B, N, 3]
                    rays_d.append(rays['rays_d'])  # [B, N, 3]

            else:
                poses = matrix_pose
                rays = get_rays(poses, intrinsics, H, W, num_rays, self.opt.patch_size)
                rays_o = rays['rays_o']  # [B, N, 3]
                rays_d = rays['rays_d']  # [B, N, 3]

        self.model.testing = True
        if self.opt.special:
            outputs = self.model.render(Xhub_feature, rays_o, rays_d, auds, bg_coords, poses, eye=eye, index=index,
                                        staged=True, bg_color=bg_color, perturb=perturb, **vars(self.opt))
        else:
            outputs = self.model.render(Xhub_feature, rays_o, rays_d, auds, bg_coords, poses, eye=eye, index=index,
                                        staged=True, bg_color=bg_color, perturb=perturb, **vars(self.opt))
        self.model.testing = False

        pred_rgb = outputs['image'].reshape(-1, H, W, 3)
        pred_depth = outputs['depth'].reshape(-1, H, W)

        return pred_rgb, pred_depth

    def save_mesh(self, save_path=None, resolution=256, threshold=10):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'meshes', f'{self.name}_{self.epoch}.ply')

        self.log(f"==> Saving mesh to {save_path}")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        def query_func(pts):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    sigma = self.model.density(pts.to(self.device))['sigma']
            return sigma

        vertices, triangles = extract_geometry(self.model.aabb_infer[:3], self.model.aabb_infer[3:],
                                               resolution=resolution, threshold=threshold, query_func=query_func)

        mesh = trimesh.Trimesh(vertices, triangles, process=False)  # important, process=True leads to seg fault...
        mesh.export(save_path)

        self.log(f"==> Finished saving mesh.")


    def train(self, train_loader, valid_loader, max_epochs):
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))

        pose = train_loader._data.poses_nerf
        euler_angles, translation = quaternion_to_euler_pose(pose)

        # mean training pose
        avg_poses = np.zeros((1, 4, 4), dtype=np.float32)
        poses_clone = pose.clone().cpu().numpy()
        N = poses_clone.shape[0]
        trans = poses_clone[:, :3, 3].copy()  # [N, 3]
        rots = poses_clone[:, :3, :3].copy()  # [N, 3, 3]
        avg_poses[:, :3, 3] = trans[0:N].mean(0)
        avg_poses[:, :3, :3] = Rotation.from_matrix(rots[0:N]).mean().as_matrix()
        avg_poses[:, 3, 3] = 1
        avg_poses = torch.tensor(avg_poses, device=self.device)
        euler_angles_avg, translation_avg = quaternion_to_euler_pose(avg_poses)

        self.model.avg_pose = {"quaternion": avg_poses,
                               "euler": euler_angles_avg,
                               "translation": translation_avg}

        # pose clip bounds
        max_pose = torch.max(torch.cat((euler_angles, translation), dim=1), dim=0)[0]  # (6,)
        min_pose = torch.min(torch.cat((euler_angles, translation), dim=1), dim=0)[0]  # (6,)
        self.model.max_true = max_pose.to(self.device)
        self.model.min_true = min_pose.to(self.device)

        euler_angles_array = np.array(euler_angles.cpu().detach().numpy())
        translation_array = np.array(translation.cpu().detach().numpy())
        # euler_angles = euler_angles_array

        bs = train_loader._data.eye_area

        avg_bs = torch.mean(bs, dim=0).unsqueeze(0).to(self.device)
        self.model.avg_bs = avg_bs
        avg_pose_bs = torch.cat([euler_angles_avg, translation_avg, avg_bs], dim=1)

        with open(os.path.join(self.opt.workspace, 'max_min.pkl'), 'wb') as f:
            pickle.dump({'max': max_pose,
                         'min': min_pose,
                         'avg_pose': {"quaternion": avg_poses,
                                      "euler": euler_angles_avg,
                                      "translation": translation_avg},
                         'avg_bs': avg_bs,
                         'avg_pose_bs': avg_pose_bs}, f)
        print("Tensors saved to max_min.pkl")

        Xhub_input = torch.tensor(train_loader._data.Xhub_features, dtype=torch.float32).to(self.device)

        audio_drive_label = torch.cat(
            ((euler_angles - euler_angles_avg.cpu()), (translation - translation_avg.cpu()), (bs - avg_bs.cpu())),
            dim=1).to(self.device)

        audio_drive_label_array = audio_drive_label.cpu().detach().numpy()
        audio_drive_label_mean = audio_drive_label.mean(dim=0, keepdim=True)
        audio_drive_label_std = audio_drive_label.std(dim=0, keepdim=True)

        self.model.mean_true = audio_drive_label_mean.to(self.device)
        self.model.std_true = audio_drive_label_std.to(self.device)

        with open(os.path.join(self.opt.workspace, 'label_mean_std.pkl'), 'wb') as f:
            pickle.dump({'mean': audio_drive_label_mean, 'std': audio_drive_label_std}, f)
        print("Tensors saved to max_min.pkl")

        # normalize drive labels
        audio_drive_label_centered = (audio_drive_label - audio_drive_label_mean) / (audio_drive_label_std + 1e-8)

        shifted_audio_drive_label = audio_drive_label_centered.clone()

        shifted_audio_drive_label[1:-1] = audio_drive_label_centered[:-2]

        shifted_audio_drive_label_array = shifted_audio_drive_label.cpu().detach().numpy()
        audio_drive_label_centered_array = audio_drive_label_centered.cpu().detach().numpy()

        mean = 0
        std = 0.01
        noise = torch.normal(mean, std, size=(shifted_audio_drive_label.shape[0], shifted_audio_drive_label.shape[1]),
                             dtype=torch.float32, device=audio_drive_label_centered.device)

        if train_loader._data.opt.torso is False and train_loader._data.opt.finetune_lips is True and self.torso_dic is not None:

            # # freeze these keys
            #         v.requires_grad = False

            for k, v in self.model.named_parameters():
                if k.split('.')[0] == 'au2BsNet' or k.split('.')[0] == 'au2PoseNet':
                    print(f'[INFO] freeze {k}, {v.shape}')
                    v.requires_grad = False

            target_semantics_list = []
            semantic_radius = 13
            for frame_idx in range(audio_drive_label_centered.shape[0]):
                target_semantics = transform_semantic_target(Xhub_input, frame_idx, semantic_radius)
                target_semantics_list.append(target_semantics)
            Xhub_input_ = torch.stack(target_semantics_list, dim=0)

            bs_data = [avg_bs.repeat(N, 1), audio_drive_label_centered[:, 6:]]
            pose_bs_data = [audio_drive_label_centered[:, :6], shifted_audio_drive_label[:, :6] + noise[:, :6]]

            with torch.no_grad():
                t = torch.arange(0, N).unsqueeze(-1).to(self.device)
                out, _, _, mu_pred, logvar_pred = self.model.forward_audio(poses_bs=pose_bs_data,
                                                                           bs=bs_data,
                                                                           audio_feat_pose=Xhub_input,
                                                                           audio_feat_bs=Xhub_input_,
                                                                           external_code=None,
                                                                           transform=(euler_angles_avg,
                                                                                      translation_avg,
                                                                                      avg_bs),
                                                                           t=t)
                label = audio_drive_label_centered
                Lmse_bs = self.criterionL1(out[:, 6:], label[:, 6:13])
                Lmse_pose = self.criterionL1(out[:, :6], label[:, :6])

                kl_pose_loss = 0.5 * (-logvar_pred[0] + mu_pred[0] ** 2 + torch.exp(logvar_pred[0]) - 1).mean() * 0.1
                kl_bs_loss = 0.5 * (-logvar_pred[1] + mu_pred[1] ** 2 + torch.exp(logvar_pred[1]) - 1).mean() * 0.1

                print(
                    f"pred_bs loss is {Lmse_bs}, pred_pose loss{Lmse_pose}, pred_POSE KL loss{kl_pose_loss}, pred_BS KL loss{kl_bs_loss}")

        elif train_loader._data.opt.torso is False and train_loader._data.opt.finetune_lips is False:
            # freeze au2BsNet and au2PoseNet
            for k, v in self.model.named_parameters():
                if k.split('.')[0] == 'au2BsNet':
                    print(f'[INFO] open {k}, {v.shape}')
                    v.requires_grad = True

            target_semantics_list = []
            semantic_radius = 13
            for frame_idx in range(audio_drive_label_centered.shape[0]):
                target_semantics = transform_semantic_target(Xhub_input, frame_idx, semantic_radius)
                target_semantics_list.append(target_semantics)
            Xhub_input_ = torch.stack(target_semantics_list, dim=0)

            t = torch.arange(0, N).unsqueeze(-1).to(self.device)

            bs_data = [avg_bs.repeat(N, 1), audio_drive_label_centered[:, 6:] ]
            pose_bs_data = [audio_drive_label_centered[:, :6], shifted_audio_drive_label[:, :6] + noise[:, :6]]

            out, a, mu_pred, logvar_pred = self.model.forward_audio(poses_bs=pose_bs_data,
                                                                 bs=bs_data, audio_feat_pose=Xhub_input,
                                                                 audio_feat_bs=Xhub_input_,
                                                                 external_code=None, transform=(euler_angles_avg,
                                                                                                translation_avg,
                                                                                                avg_bs),
                                                                 t=t)
            label = audio_drive_label_centered
            Lmse_bs = self.criterionL1(out[:, :7], label[:, 6:13])
            Lmse_pose = 0

            kl_pose_loss = 0.5 * (-logvar_pred[0] + mu_pred[0] ** 2 + torch.exp(logvar_pred[0]) - 1).mean() * 0.1
            kl_bs_loss = 0.5 * (-logvar_pred[1] + mu_pred[1] ** 2 + torch.exp(logvar_pred[1]) - 1).mean() * 0.1

            print(
                f"pred_bs loss is {Lmse_bs}, pred_pose loss{Lmse_pose}, pred_POSE KL loss{kl_pose_loss}, pred_BS KL loss{kl_bs_loss}")

        elif train_loader._data.opt.torso is True:

            # freeze au2BsNet and au2PoseNet
            for k, v in self.model.named_parameters():
                if k.split('.')[0] == 'au2BsNet':
                    print(f'[INFO] freeze {k}, {v.shape}')
                    v.requires_grad = False

        # mark unseen density voxels
        if self.model.cuda_ray and train_loader._data.opt.torso:
            self.model.mark_untrained_grid(train_loader._data.poses, train_loader._data.intrinsics)
        elif self.model.cuda_ray and train_loader._data.opt.torso is False:
            self.model.mark_untrained_grid(train_loader._data.poses, train_loader._data.intrinsics)

        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch

            self.train_one_epoch(train_loader, pose_label=audio_drive_label_centered,
                                 avg=(euler_angles_avg, translation_avg, avg_bs, avg_pose_bs))

            if self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(full=True, best=False)

            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader, pose_label=audio_drive_label_centered,
                                        avg=(euler_angles_avg, translation_avg, avg_bs, avg_pose_bs))
                self.save_checkpoint(full=False, best=True)

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, loader, name=None):
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader, name)
        self.use_tensorboardX = use_tensorboardX


    def test(self, loader, train_loader, save_path=None, name=None, write_image=False):

        if self.opt.infer is None:
            self.log("[INFO] skip test: --infer is required for inference video export")
            return

        if save_path is None:
            save_path = os.path.join(self.workspace, 'results')

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        os.makedirs(save_path, exist_ok=True)

        bs = train_loader._data.eye_area.to(self.device)

        poses = train_loader._data.poses_nerf.to(self.device)
        euler_angles, translation = quaternion_to_euler_pose(poses)
        pose_angles = torch.cat([euler_angles, translation], dim=1)

        # load infer stats
        if self.opt.infer is not None:
            workspace_name = self.opt.workspace  # .split('_')[0] + '_trial' + '_torso' + '_audio'# + self.opt.workspace.split('_')[-1] #+ 'testV20'
            with open(os.path.join(workspace_name, 'max_min.pkl'), 'rb') as f:
                dic_ = pickle.load(f)

            self.model.max_true = dic_['max'].to(self.device)
            self.model.min_true = dic_['min'].to(self.device)
            avg_pose = {k: v.to(self.device) for k, v in dic_['avg_pose'].items()}
            self.model.avg_pose = avg_pose
            avg_bs = dic_["avg_bs"]
            avg_pose_bs = dic_["avg_pose_bs"]

            with open(os.path.join(workspace_name, 'label_mean_std.pkl'), 'rb') as f:
                dic_mean = pickle.load(f)
            self.model.mean_true = dic_mean["mean"].to(self.device)
            self.model.std_true = dic_mean["std"].to(self.device)

            Xhub_input = torch.tensor(loader._data.Xhub_features, dtype=torch.float32).to(self.device)

            standar_gaussian_pose = torch.randn((Xhub_input.shape[0], self.opt.noise_dim_pose), dtype=torch.float32,
                                                device=self.device)
            standar_gaussian_bs = torch.randn((Xhub_input.shape[0], 256), dtype=torch.float32, device=self.device)
            standar_gaussian = [standar_gaussian_pose, standar_gaussian_bs]

            t = torch.arange(0, Xhub_input.shape[0]).unsqueeze(-1).to(self.device)

            bs_norm = ((bs - avg_bs[:, :7]) - self.model.mean_true[:, 6:13]) / (self.model.std_true[:, 6:13] + 1e-8)

            # bs_norm_ = bs_norm[:Xhub_input.shape[0], :]

            # repeat first-frame bs/pose norm
            bs_norm_ = bs_norm[0, :].unsqueeze(0).repeat(Xhub_input.shape[0], 1)

            bs_data = [avg_bs.repeat(Xhub_input.shape[0], 1), bs_norm_]

            poses_norm = ((pose_angles - avg_pose_bs[:, :6]) - self.model.mean_true[:, :6]) / (
                    self.model.std_true[:, :6] + 1e-8)

            # poses_norm_ = poses_norm[:Xhub_input.shape[0], :]

            # repeat first-frame bs/pose norm
            poses_norm_ = poses_norm[0, :].unsqueeze(0).repeat(Xhub_input.shape[0], 1)


            pose_bs_data = [None, poses_norm_[:, :6]]

            target_semantics_list = []
            semantic_radius = 13
            for frame_idx in range(Xhub_input.shape[0]):
                target_semantics = transform_semantic_target(Xhub_input, frame_idx, semantic_radius)
                target_semantics_list.append(target_semantics)
            Xhub_input_ = torch.stack(target_semantics_list, dim=0)


            pose_bs, matrix_pose_nerf, arkit, _, _ = self.model.forward_audio(poses_bs=pose_bs_data, bs=bs_data,
                                                                              audio_feat_pose=Xhub_input,
                                                                              audio_feat_bs=Xhub_input_,
                                                                              external_code=standar_gaussian,
                                                                              transform=(avg_pose["euler"],
                                                                                         avg_pose["translation"],
                                                                                         avg_bs), t=t)
            arkit_5 = torch.max(arkit[:, 5], dim=0)[0]
            arkit_6 = torch.max(arkit[:, 6], dim=0)[0]
            with open(os.path.join(self.workspace, 'blink_max'), 'wb') as f:
                pickle.dump({'max_5': arkit_5, 'max_6': arkit_6}, f)
            all_bs_from_image = arkit.cpu().detach().numpy()

            all_bs_from_image_array = all_bs_from_image
            # plt.figure(figsize=(10, 5))
            # plt.title('Line Plot of Columns 6 and 7')
            # plt.xlabel('Row Index')
            # plt.ylabel('Values')
            # plt.legend()
            # plt.savefig(os.path.join(self.workspace, "line graph (FROM PRE test)"))
            # plt.close()

            matrix_pose = self.model.nerf_matrix_to_ngp_tensor(matrix_pose_nerf, scale=self.opt.scale,
                                                               offset=self.opt.offset)

            # # visualize_and_export_poses_video(matrix_pose
            # #                                  save_path="poses_video(V20_test)
            # matrix_pose = matrix_pose.cpu().detach().numpy()
            # N = matrix_pose.shape[0]
            # K = 10 // 2
            # trans = matrix_pose[:, :3, 3].copy()  # [N, 3]
            # rots = matrix_pose[:, :3, :3].copy()  # [N, 3, 3]
            #     start = max(0, i - K)
            #     end = min(N, i + K + 1)

            #                                  save_path="poses_video(V34_smooth_testV2).mp4")

        else:
            with open(os.path.join(self.opt.workspace, 'max_min.pkl'), 'rb') as f:
                dic_ = pickle.load(f)

            self.model.max_true = dic_['max'].to(self.device)
            self.model.min_true = dic_['min'].to(self.device)
            avg_pose = {k: v.to(self.device) for k, v in dic_['avg_pose'].items()}
            self.model.avg_pose = avg_pose
            avg_bs = dic_["avg_bs"]
            avg_pose_bs = dic_["avg_pose_bs"]

            with open(os.path.join(self.opt.workspace, 'label_mean_std.pkl'), 'rb') as f:
                dic_mean = pickle.load(f)
            self.model.mean_true = dic_mean["mean"].to(self.device)
            self.model.std_true = dic_mean["std"].to(self.device)

            Xhub_input = torch.tensor(loader._data.Xhub_features, dtype=torch.float32).to(self.device)

            standar_gaussian_pose = torch.randn((Xhub_input.shape[0], self.opt.noise_dim_pose), dtype=torch.float32,
                                                device=self.device)
            standar_gaussian_bs = torch.randn((Xhub_input.shape[0], 256), dtype=torch.float32, device=self.device)
            standar_gaussian = [standar_gaussian_pose, standar_gaussian_bs]

            t = torch.arange(0, Xhub_input.shape[0]).unsqueeze(-1).to(self.device)


            bs_norm = ((bs - avg_bs[:, :7]) - self.model.mean_true[:, 6:13]) / (self.model.std_true[:, 6:13] + 1e-8)
            all_bs_norm = bs_norm[:Xhub_input.shape[0], :]
            # torch
            #                      dim=1)]
            bs_data = [avg_bs.repeat(Xhub_input.shape[0], 1), all_bs_norm]

            poses_norm = ((pose_angles - avg_pose_bs[:, :6]) - self.model.mean_true[:, :6]) / (
                        self.model.std_true[:, :6] + 1e-8)
            all_poses_norm = poses_norm[:Xhub_input.shape[0], :]
            pose_bs_data = [None, all_poses_norm[:, :6]]


            target_semantics_list = []
            semantic_radius = 13
            for frame_idx in range(Xhub_input.shape[0]):
                target_semantics = transform_semantic_target(Xhub_input, frame_idx, semantic_radius)
                target_semantics_list.append(target_semantics)
            Xhub_input_ = torch.stack(target_semantics_list, dim=0)

            pose_bs, matrix_pose_nerf, arkit, _, _ = self.model.forward_audio(poses_bs=pose_bs_data,
                                                                              bs=bs_data, audio_feat_pose=Xhub_input,
                                                                              audio_feat_bs=Xhub_input_,
                                                                              external_code=standar_gaussian,
                                                                              transform=(avg_pose["euler"],
                                                                                         avg_pose["translation"],
                                                                                         avg_bs), t=t)

            matrix_pose = self.model.nerf_matrix_to_ngp_tensor(matrix_pose_nerf, scale=self.opt.scale,
                                                               offset=self.opt.offset)

            #                                  save_path=f"{self.workspace}_poses_video(V16).mp4")

            matrix_pose = matrix_pose.cpu().detach().numpy()
            N = matrix_pose.shape[0]
            K = 10 // 2

            trans = matrix_pose[:, :3, 3].copy()  # [N, 3]
            rots = matrix_pose[:, :3, :3].copy()  # [N, 3, 3]

            for i in range(N):
                start = max(0, i - K)
                end = min(N, i + K + 1)
                matrix_pose[i, :3, 3] = trans[start:end].mean(0)
                matrix_pose[i, :3, :3] = Rotation.from_matrix(rots[start:end]).mean().as_matrix()

            #                                  save_path=f"{self.workspace}_poses_video(smooth).mp4")

        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                         bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()

        all_preds = []
        all_preds_depth = []
        if self.opt.infer is not None:
            infer_feats = torch.tensor(loader._data.Xhub_features_infer, dtype=torch.float32).to(self.device)

            standar_gaussian_pose = torch.randn((infer_feats.shape[0], self.opt.noise_dim_pose), dtype=torch.float32,
                                                device=self.device)
            standar_gaussian_bs = torch.randn((infer_feats.shape[0], 256), dtype=torch.float32, device=self.device)
            standar_gaussian = [standar_gaussian_pose, standar_gaussian_bs]

            t = torch.arange(0, infer_feats.shape[0]).unsqueeze(-1).to(self.device)

            target_semantics_list = []
            semantic_radius = 13
            for frame_idx in range(infer_feats.shape[0]):
                target_semantics = transform_semantic_target(Xhub_input, frame_idx, semantic_radius)
                target_semantics_list.append(target_semantics)
            Xhub_input_infer = torch.stack(target_semantics_list, dim=0)

            bs_norm_ = bs_norm[:infer_feats.shape[0], :]
            poses_norm_ = poses_norm[:infer_feats.shape[0], :]

        else:
            standar_gaussian_pose = torch.randn((1, self.opt.noise_dim_pose), dtype=torch.float32, device=self.device)
            standar_gaussian_bs = torch.randn((1, 256), dtype=torch.float32, device=self.device)
            standar_gaussian = [standar_gaussian_pose, standar_gaussian_bs]

        with torch.no_grad():

            for i, data in enumerate(loader):
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    if self.opt.infer is not None:
                        if i == 0:
                            preds, preds_depth = self.test_step(data, avg=(
                            avg_pose["euler"], avg_pose["translation"], avg_bs, avg_pose_bs),
                                                                infer_feats=infer_feats,
                                                                standar_gaussian=standar_gaussian, t=t,
                                                                Xhub_input_bs=Xhub_input_infer, bs_norm=bs_norm_,
                                                                poses_norm=poses_norm_, cal=True)
                        else:
                            preds, preds_depth = self.test_step(data, avg=(
                            avg_pose["euler"], avg_pose["translation"], avg_bs, avg_pose_bs),
                                                                infer_feats=infer_feats,
                                                                standar_gaussian=standar_gaussian, t=t,
                                                                Xhub_input_bs=Xhub_input_infer, bs_norm=bs_norm_,
                                                                poses_norm=poses_norm_, cal=False)
                    else:
                        num_idx = data['index'][0]
                        t = torch.tensor([num_idx], dtype=torch.float32, device=self.device).unsqueeze(0)
                        Xhub_input_bs = Xhub_input_[num_idx, :, :].unsqueeze(0)
                        bs_norm = all_bs_norm[num_idx, :].unsqueeze(0)
                        poses_norm = all_poses_norm[num_idx, :].unsqueeze(0)
                        preds, preds_depth = self.test_step(data, avg=(
                        avg_pose["euler"], avg_pose["translation"], avg_bs, avg_pose_bs), infer_feats=None,
                                                            standar_gaussian=standar_gaussian, t=t,
                                                            Xhub_input_bs=Xhub_input_bs, bs_norm=bs_norm,
                                                            poses_norm=poses_norm)

                path = os.path.join(save_path, f'{name}_{i:04d}_rgb.png')
                path_depth = os.path.join(save_path, f'{name}_{i:04d}_depth.png')

                # self.log(f"[INFO] saving test image to {path}")

                if self.opt.color_space == 'linear':
                    preds = linear_to_srgb(preds)
                if self.opt.portrait:
                    pred = blend_with_mask_cuda(preds[0], data["bg_gt_images"].squeeze(0),
                                                data["bg_face_mask"].squeeze(0))
                    pred = (pred * 255).astype(np.uint8)
                else:
                    pred = preds[0].detach().cpu().numpy()
                    pred = (pred * 255).astype(np.uint8)

                pred_depth = preds_depth[0].detach().cpu().numpy()
                pred_depth = (pred_depth * 255).astype(np.uint8)

                if write_image:
                    imageio.imwrite(path, pred)
                    imageio.imwrite(path_depth, pred_depth)

                all_preds.append(pred)
                all_preds_depth.append(pred_depth)

                pbar.update(loader.batch_size)

        # export infer video
        all_preds = np.stack(all_preds, axis=0)
        all_preds_depth = np.stack(all_preds_depth, axis=0)
        imageio.mimwrite(os.path.join(save_path, f'{name}_infer.mp4'), all_preds, fps=25, quality=8, macro_block_size=1)
        imageio.mimwrite(os.path.join(save_path, f'{name}_depth_infer.mp4'), all_preds_depth, fps=25, quality=8,
                         macro_block_size=1)

        if self.opt.infer is not None:
            hum = self.opt.path.split('/')[-1]
            if len(self.opt.infer.split('/')) > 3:
                aud_path = os.path.join('data/inference/exp', self.opt.infer.split('/')[-2], "aud.wav")
                save = os.path.join(save_path, f"{hum}_with_{self.opt.infer.split('/')[-2]}.mp4")
                # test video
                os.system(
                    f'ffmpeg -i {os.path.join(save_path, f"{name}_infer.mp4")} -i {aud_path} -strict -2 -c:v copy {save} -y')

        if self.opt.aud != '' and self.opt.asr_model == 'ave':
            os.system(
                f'ffmpeg -i {os.path.join(save_path, f"{name}.mp4")} -i {self.opt.aud} -strict -2 -c:v copy {os.path.join(save_path, f"{name}_audio.mp4")} -y')

        self.log(f"==> Finished Test.")

    def train_gui(self, train_loader, step=16):

        self.model.train()

        total_loss = torch.tensor([0], dtype=torch.float32, device=self.device)

        loader = iter(train_loader)

        # mark untrained grid
        if self.global_step == 0:
            self.model.mark_untrained_grid(train_loader._data.poses, train_loader._data.intrinsics)

        for _ in range(step):

            # wrap dataloader iterator
            try:
                data = next(loader)
            except StopIteration:
                loader = iter(train_loader)
                data = next(loader)

            # refresh density grid periodically
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()

            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, truths, loss = self.train_step(data)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            total_loss += loss.detach()

            if self.ema is not None and self.global_step % self.ema_update_interval == 0:
                self.ema.update()

        average_loss = total_loss.item() / step

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        outputs = {
            'loss': average_loss,
            'lr': self.optimizer.param_groups[0]['lr'],
        }

        return outputs

    def test_gui(self, pose, intrinsics, W, H, auds, eye=None, index=0, bg_color=None, spp=1, downscale=1):

        rH = int(H * downscale)
        rW = int(W * downscale)
        intrinsics = intrinsics * downscale

        if auds is not None:
            auds = auds.to(self.device)

        pose = torch.from_numpy(pose).unsqueeze(0).to(self.device)
        rays = get_rays(pose, intrinsics, rH, rW, -1)

        bg_coords = get_bg_coords(rH, rW, self.device)

        if eye is not None:
            eye = torch.FloatTensor([eye]).view(1, 1).to(self.device)

        data = {
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'H': rH,
            'W': rW,
            'auds': auds,
            'index': [index],  # support choosing index for individual codes
            'eye': eye,
            'poses': pose,
            'bg_coords': bg_coords,
        }

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.fp16):
                # spp acts as perturb seed
                # no perturb on first GUI spp pass
                preds, preds_depth = self.test_step(data, bg_color=bg_color, perturb=False if spp == 1 else spp)

        if self.ema is not None:
            self.ema.restore()

        if downscale != 1:
            preds = F.interpolate(preds.permute(0, 3, 1, 2), size=(H, W), mode='bilinear').permute(0, 2, 3,
                                                                                                   1).contiguous()
            preds_depth = F.interpolate(preds_depth.unsqueeze(1), size=(H, W), mode='nearest').squeeze(1)

        if self.opt.color_space == 'linear':
            preds = linear_to_srgb(preds)

        pred = preds[0].detach().cpu().numpy()
        pred_depth = preds_depth[0].detach().cpu().numpy()

        outputs = {
            'image': pred,
            'depth': pred_depth,
        }

        return outputs

    def test_gui_with_data(self, data, W, H):

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.fp16):
                # spp acts as perturb seed
                # no perturb on first GUI spp pass
                preds, preds_depth = self.test_step(data, perturb=False)

        if self.ema is not None:
            self.ema.restore()

        if self.opt.color_space == 'linear':
            preds = linear_to_srgb(preds)

        preds = F.interpolate(preds.permute(0, 3, 1, 2), size=(H, W), mode='bilinear').permute(0, 2, 3, 1).contiguous()
        preds_depth = F.interpolate(preds_depth.unsqueeze(1), size=(H, W), mode='nearest').squeeze(1)

        pred = preds[0].detach().cpu().numpy()
        pred_depth = preds_depth[0].detach().cpu().numpy()

        outputs = {
            'image': pred,
            'depth': pred_depth,
        }

        return outputs

    def train_one_epoch(self, loader, pose_label=None, avg=None):
        self.log(f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        # DistributedSampler.set_epoch
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                             mininterval=1,
                             bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        pose = loader._data.poses_nerf
        euler_angles, translation = quaternion_to_euler_pose(pose)

        euler_angles_array = np.array(euler_angles.cpu().detach().numpy())
        translation_array = np.array(translation.cpu().detach().numpy())

        Xhub_input = torch.tensor(loader._data.Xhub_features, dtype=torch.float32).to(self.device)

        bs = loader._data.eye_area

        audio_drive_label = pose_label
        label = audio_drive_label

        euler_angles_avg, translation_avg, avg_bs, avg_pose_bs = avg[0], avg[1], avg[2], avg[3]

        target_semantics_list = []
        semantic_radius = 13
        for frame_idx in range(label.shape[0]):
            target_semantics = transform_semantic_target(Xhub_input, frame_idx, semantic_radius)
            target_semantics_list.append(target_semantics)
        Xhub_input_ = torch.stack(target_semantics_list, dim=0)


        shifted_label = label.clone()
        shifted_label[1:-1] = label[:-2]

        N = Xhub_input_.shape[0]
        mean = 0
        std = 0.01

        pose_bs_data = [label[:, :6], shifted_label[:, :6]]
        bs_data = [avg_bs.repeat(label.shape[0], 1), label[:, 6:]]
        t = torch.arange(0, audio_drive_label.shape[0]).unsqueeze(-1).to(self.device)

        loss_fn = ContrastiveVideoLossV2(n=3)
        bs_loss = BSLoss()


        printed_once = False
        for idx, data in enumerate(loader):
            if self.opt.torso:
                t = torch.arange(0, audio_drive_label.shape[0]).unsqueeze(-1).to(self.device)

                noise = torch.normal(mean, std,
                                     size=(t.shape[0], label.shape[1]),
                                     dtype=torch.float32, device=label.device)
                pose_bs_data_ = [label[:, :6], shifted_label[:, :6] + noise[:, :6]]

                output, matrix_poses, mu_pred, logvar_pred = self.model.forward_audio(poses_bs=pose_bs_data_,
                                                                                      bs=bs_data,
                                                                                      audio_feat_pose=Xhub_input,
                                                                                      audio_feat_bs=Xhub_input_,
                                                                                      t=t,
                                                                                      transform=(euler_angles_avg,
                                                                                                 translation_avg,
                                                                                                 avg_bs),
                                                                                      clip=True)
                attr_loss = loss_fn(predictions=output[:, :6],
                                    mu_var=[mu_pred[0], logvar_pred[0]],
                                    target_idx=data["index"][0],
                                    target=label[:, :6], idx=self.epoch - 1)

                if attr_loss < 0.001:
                    for k, v in self.model.named_parameters():
                        if k.split('.')[0] == 'au2PoseNet':
                            if not printed_once:
                                print(f'[INFO] freeze {k}, {v.shape}')
                                printed_once = True
                            v.requires_grad = False

                matrix_pose_nerf = matrix_poses[data["index"][0], :, :].unsqueeze(0)

                bs_ = bs[data["index"][0]].unsqueeze(0)
            elif self.opt.finetune_lips is not True:
                t = torch.arange(0, audio_drive_label.shape[0]).unsqueeze(-1).to(self.device)

                noise = torch.normal(mean, std,
                                     size=(label.shape[0], label.shape[1]),
                                     dtype=torch.float32, device=label.device)

                pose_bs_data_ = [label[:, :6], shifted_label[:, :6] + noise[:, :6]]
                bs_data = [avg_bs.repeat(label.shape[0], 1), label[:, 6:]]
                output, a, mu_pred, logvar_pred = self.model.forward_audio(poses_bs=pose_bs_data_,
                                                                        bs=bs_data,
                                                                        audio_feat_pose=Xhub_input,
                                                                        audio_feat_bs=Xhub_input_,
                                                                        t=t,transform=(euler_angles_avg,
                                                                                       translation_avg,
                                                                                        avg_bs))

                matrix_pose_nerf = None

                bs_ = a[data["index"][0]].unsqueeze(0)
                attr_loss = bs_loss(output, audio_drive_label[:, 6:13], mu_var=[mu_pred[1], logvar_pred[1]])
            else:
                with torch.no_grad():
                    self.model.au2PoseNet.eval()
                    self.model.au2BsNet.eval()
                    t = torch.arange(0, audio_drive_label.shape[0]).unsqueeze(-1).to(self.device)
                    noise = torch.normal(mean, std,
                                         size=(label.shape[0], label.shape[1]),
                                         dtype=torch.float32, device=label.device)
                    pose_bs_data_ = [label[:, :6], shifted_label[:, :6] + noise[:, :6]]

                    bs_data = [avg_bs.repeat(label.shape[0], 1), label[:, 6:]]
                    output, matrix_poses, bs_pre, mu_pred, logvar_pred = self.model.forward_audio(
                        poses_bs=pose_bs_data_,
                        bs=bs_data,
                        audio_feat_pose=Xhub_input,
                        audio_feat_bs=Xhub_input_,
                        t=t, transform=(euler_angles_avg,
                                        translation_avg,
                                        avg_bs))
                matrix_pose_nerf = matrix_poses[data["index"][0], :, :].unsqueeze(0)

                # matrix_pose_nerf = torch

                bs_ = bs_pre[data["index"][0]].unsqueeze(0)
                attr_loss = bs_loss(output, audio_drive_label[:, :13], mu_var=[mu_pred[1], logvar_pred[1]])

            # refresh density grid periodically
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()

            self.local_step += 1
            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, truths, loss, attr_loss = self.train_step(data, preds=[matrix_pose_nerf, bs_, attr_loss])


            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            #     else:

            loss_val = loss.item()
            if torch.isnan(loss):
                print()
            total_loss += loss_val

            if self.ema is not None and self.global_step % self.ema_update_interval == 0:
                self.ema.update()

            if self.local_rank == 0:
                if self.report_metric_at_train:
                    for metric in self.metrics:
                        metric.update(preds, truths)

                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)

                if loader._data.opt.torso is not True:
                    if self.scheduler_update_every_step:
                        pbar.set_description(f"loss={loss_val:.4f} ({total_loss / self.local_step:.4f}), "
                                             f"lr={self.optimizer.param_groups[0]['lr']:.6f},"
                                             f"bs_loss={float(attr_loss.detach().cpu().numpy()):.4f}")
                    else:
                        pbar.set_description(f"loss={loss_val:.4f} "
                                             f"({total_loss / self.local_step:.4f}) "
                                             f"(bs_loss={float(attr_loss.detach().cpu().numpy()):.4f})")
                else:
                    if self.scheduler_update_every_step:
                        pbar.set_description(f"loss={loss_val:.4f} ({total_loss / self.local_step:.4f}), "
                                             f"lr={self.optimizer.param_groups[0]['lr']:.6f},"
                                             f"pose_loss={float(attr_loss.detach().cpu().numpy()):.4f}")
                    else:
                        pbar.set_description(f"loss={loss_val:.4f} "
                                             f"({total_loss / self.local_step:.4f}) "
                                             f"(pose_loss={float(attr_loss.detach().cpu().numpy()):.4f})")

                pbar.update(loader.batch_size)

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()
        self.log(f"loss={average_loss:.4f}")
        self.log(f"==> Finished Epoch {self.epoch}.")

    def evaluate_one_epoch(self, loader, pose_label=None, avg=None, name=None):
        self.log(f"++> Evaluate at epoch {self.epoch} ...")

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.eval()

        pose = loader._data.poses_nerf
        bs = loader._data.eye_area
        euler_angles_nerf, translation_nerf = quaternion_to_euler_pose(pose)
        euler_angles_array = np.array(euler_angles_nerf.cpu().detach().numpy())
        translation_array = np.array(translation_nerf.cpu().detach().numpy())

        euler_angles_avg, translation_avg, avg_bs, avg_pose_bs = avg[0], avg[1], avg[2], avg[3]
        Xhub_input = torch.tensor(loader._data.Xhub_features, dtype=torch.float32).to(self.device)

        audio_drive_label = torch.cat(((euler_angles_nerf - euler_angles_avg.cpu()),
                                       (translation_nerf - translation_avg.cpu()), (bs - avg_bs.cpu())), dim=1).to(self.device)

        audio_drive_label_centered = (audio_drive_label - self.model.mean_true) / (self.model.std_true + 1e-8)
        if self.opt.torso is True:


            pose_bs_data = [None, audio_drive_label_centered[:, :6]]
            # pose_bs_data = audio_drive_label_centered[:, :13]

            bs_data = [avg_bs.repeat(Xhub_input.shape[0], 1), audio_drive_label_centered[:, 6:13]]

            target_semantics_list = []
            semantic_radius = 13
            for frame_idx in range(audio_drive_label_centered.shape[0]):
                target_semantics = transform_semantic_target(Xhub_input, frame_idx, semantic_radius)
                target_semantics_list.append(target_semantics)
            Xhub_input_ = torch.stack(target_semantics_list, dim=0)

            standar_gaussian_pose = torch.randn((Xhub_input.shape[0], self.opt.noise_dim_pose), dtype=torch.float32,
                                                device=self.device)
            standar_gaussian_bs = torch.randn((Xhub_input.shape[0], 256), dtype=torch.float32, device=self.device)
            standar_gaussian = [standar_gaussian_pose, standar_gaussian_bs]

            with torch.no_grad():
                t = torch.arange(0, Xhub_input.shape[0]).unsqueeze(-1).to(self.device)
                output, matrix_pose, mu_pred, logvar_pred = self.model.forward_audio(poses_bs=pose_bs_data,
                                                                                     bs=bs_data, audio_feat_pose=Xhub_input,
                                                                                     audio_feat_bs=Xhub_input_,
                                                                                     external_code=standar_gaussian,
                                                                                     transform=(euler_angles_avg,
                                                                                                translation_avg,
                                                                                                avg_bs),
                                                                                     t=t)
                label = audio_drive_label_centered

                l1_loss_0 = self.criterionL1(output[:, 0], label[:, 0])

                l1_loss_1 = self.criterionL1(output[:, 1], label[:, 1])

                l1_loss_2 = self.criterionL1(output[:, 2], label[:, 2])

                l1_loss_3 = self.criterionL1(output[:, 3], label[:, 3])

                l1_loss_4 = self.criterionL1(output[:, 4], label[:, 4])

                l1_loss_5 = self.criterionL1(output[:, 5], label[:, 5])

                Lmse_pose = self.criterionL1(output[:, :6], label[:, :6])

            data = {
                "L1 Loss 0": [l1_loss_0.cpu()],
                "L1 Loss 1": [l1_loss_1.cpu()],
                "L1 Loss 2": [l1_loss_2.cpu()],
                "L1 Loss 3": [l1_loss_3.cpu()],
                "L1 Loss 4": [l1_loss_4.cpu()],
                "L1 Loss 5": [l1_loss_5.cpu()],
                "Lmse Pose": [Lmse_pose.cpu()],
            }

            df = pd.DataFrame(data)

            csv_file_path = os.path.join(self.opt.workspace, f"losses_{self.epoch}.csv")
            df.to_csv(csv_file_path, index=False)

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                             bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        target_semantics_list = []
        semantic_radius = 13
        for frame_idx in range(audio_drive_label.shape[0]):
            target_semantics = transform_semantic_target(Xhub_input, frame_idx, semantic_radius)
            target_semantics_list.append(target_semantics)
        Xhub_input_ = torch.stack(target_semantics_list, dim=0)

        with torch.no_grad():
            self.model.eval()
            self.local_step = 0

            for idx, data in enumerate(loader):
                self.local_step += 1
                t = torch.tensor([data['index'][0]], dtype=torch.float32, device=self.device).unsqueeze(0)
                Xhub_input_bs = Xhub_input_[data['index'][0], :, :].unsqueeze(0)
                pose_diff_true = audio_drive_label[data['index'][0], :6].unsqueeze(0)
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, pred_ambient_aud, pred_ambient_eye, pred_uncertainty, truths, loss, loss_raw = self.eval_step(
                        data, avg=(euler_angles_avg, translation_avg, avg_bs, avg_pose_bs), t=t,
                        Xhub_input_bs=Xhub_input_bs, bs=bs, pose_diff_true=pose_diff_true)
                loss_val = loss.item()
                total_loss += loss_val

                # rank-0 saves validation images
                if self.local_rank == 0:

                    save_path = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_rgb.png')
                    save_path_depth = os.path.join(self.workspace, 'validation',
                                                   f'{name}_{self.local_step:04d}_depth.png')
                    save_path_ambient_aud = os.path.join(self.workspace, 'validation',
                                                         f'{name}_{self.local_step:04d}_aud.png')
                    save_path_ambient_eye = os.path.join(self.workspace, 'validation',
                                                         f'{name}_{self.local_step:04d}_eye.png')
                    save_path_uncertainty = os.path.join(self.workspace, 'validation',
                                                         f'{name}_{self.local_step:04d}_uncertainty.png')

                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    if self.opt.color_space == 'linear':
                        preds = linear_to_srgb(preds)

                    if self.opt.portrait:
                        pred = blend_with_mask_cuda(preds[0], data["bg_gt_images"].squeeze(0),
                                                    data["bg_face_mask"].squeeze(0))
                        preds = torch.from_numpy(pred).unsqueeze(0).to(self.device).float()
                    else:
                        pred = preds[0].detach().cpu().numpy()
                    pred_depth = preds_depth[0].detach().cpu().numpy()
                    #         metric.update(preds, truths)
                    pred_ambient_aud = pred_ambient_aud[0].detach().cpu().numpy()
                    pred_ambient_aud /= np.max(pred_ambient_aud)
                    pred_ambient_eye = pred_ambient_eye[0].detach().cpu().numpy()
                    pred_ambient_eye /= np.max(pred_ambient_eye)
                    pred_uncertainty = pred_uncertainty[0].detach().cpu().numpy()
                    # pred_uncertainty = (pred_uncertainty - np
                    pred_uncertainty /= np.max(pred_uncertainty)

                    cv2.imwrite(save_path, cv2.cvtColor((pred * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

                    if not self.opt.torso:
                        cv2.imwrite(save_path_depth, (pred_depth * 255).astype(np.uint8))
                        cv2.imwrite(save_path_ambient_aud, (pred_ambient_aud * 255).astype(np.uint8))
                        cv2.imwrite(save_path_ambient_eye, (pred_ambient_eye * 255).astype(np.uint8))
                        cv2.imwrite(save_path_uncertainty, (pred_uncertainty * 255).astype(np.uint8))
                        # cv2

                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss / self.local_step:.4f})")
                    pbar.update(loader.batch_size)

        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                result = self.metrics[0].measure()
                self.stats["results"].append(
                    result if self.best_mode == 'min' else - result)  # if max mode, use -result
            else:
                self.stats["results"].append(average_loss)  # if no metric, choose best by min loss

            #         self.log(metric.report(), style="blue")
            #             metric.write(self.writer, self.epoch, prefix="evaluate")
            #         metric.clear()

        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    def save_checkpoint(self, name=None, full=False, best=False, remove_old=True):

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        state = {'epoch': self.epoch,
                 'global_step': self.global_step,
                 'stats': self.stats, }

        state['mean_count'] = self.model.mean_count
        state['mean_density'] = self.model.mean_density
        state['mean_density_torso'] = self.model.mean_density_torso

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()

        if not best:

            state['model'] = self.model.state_dict()

            file_path = f"{self.ckpt_path}/{name}.pth"

            if remove_old:
                self.stats["checkpoints"].append(file_path)

                if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                    old_ckpt = self.stats["checkpoints"].pop(0)
                    if os.path.exists(old_ckpt):
                        os.remove(old_ckpt)

            torch.save(state, file_path)

        else:
            if len(self.stats["results"]) > 0:
                # always save as best ckpt
                if True:

                    # snapshot EMA weights for best ckpt
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    # drop density_grid from best ckpt
                    if 'density_grid' in state['model']:
                        del state['model']['density_grid']

                    if self.ema is not None:
                        self.ema.restore()

                    torch.save(state, self.best_path)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")

    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/{self.name}_ep*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)

        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded bare model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")

        if self.ema is not None and 'ema' in checkpoint_dict:
            self.ema.load_state_dict(checkpoint_dict['ema'])

        if 'mean_count' in checkpoint_dict:
            self.model.mean_count = checkpoint_dict['mean_count']
        if 'mean_density' in checkpoint_dict:
            self.model.mean_density = checkpoint_dict['mean_density']
        if 'mean_density_torso' in checkpoint_dict:
            self.model.mean_density_torso = checkpoint_dict['mean_density_torso']

        if model_only:
            return

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        self.global_step = checkpoint_dict['global_step']
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")

        if self.optimizer and 'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer.")

        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")

        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")


def load_wav(path, sr):
    return librosa.core.load(path, sr=sr)[0]


def preemphasis(wav, k):
    return signal.lfilter([1, -k], [1], wav)


def melspectrogram(wav):
    D = _stft(preemphasis(wav, 0.97))
    S = _amp_to_db(_linear_to_mel(np.abs(
        D))) - 20

    return _normalize(S)


def _stft(y):
    return librosa.stft(y=y, n_fft=800, hop_length=200, win_length=800)


def _linear_to_mel(spectogram):
    global _mel_basis
    _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectogram)


def _build_mel_basis():
    return librosa.filters.mel(sr=16000, n_fft=800, n_mels=80, fmin=55, fmax=7600)


def _amp_to_db(x):
    min_level = np.exp(-5 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _normalize(S):
    return np.clip((2 * 4.) * ((S - -100) / (--100)) - 4., -4., 4.)


class AudDataset(object):
    def __init__(self, wavpath):
        wav = load_wav(wavpath, 16000)

        self.orig_mel = melspectrogram(wav).T
        self.data_len = int((self.orig_mel.shape[0] - 16) / 80. * float(25)) + 2

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def crop_audio_window(self, spec, start_frame):
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(80. * (start_frame_num / float(25)))

        end_idx = start_idx + 16
        if end_idx > spec.shape[0]:
            end_idx = spec.shape[0]
            start_idx = end_idx - 16

        return spec[start_idx: end_idx, :]

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        mel = self.crop_audio_window(self.orig_mel.copy(), idx)  # 16×80
        if (mel.shape[0] != 16):
            raise Exception('mel.shape[0] != 16')
        mel = torch.FloatTensor(mel.T).unsqueeze(0)

        return mel


def quaternion_to_euler_pose(quaternion_pose):
    """Extract euler angles and translation from (B, 4, 4) pose matrices."""
    rotation_matrix = quaternion_pose[:, :3, :3]
    translation = quaternion_pose[:, :3, 3]
    euler_angles = rotation_matrix_to_euler(rotation_matrix)

    return euler_angles, translation


def rotation_matrix_to_euler(R_mat):
    """Convert (B, 3, 3) rotation matrices to euler angles."""
