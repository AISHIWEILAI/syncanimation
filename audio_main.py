import argparse
import os

from syncanimation_nerf.provider import NeRFDataset
from syncanimation_nerf.utils import *
from syncanimation_nerf.network import NeRFNetwork

try:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
except AttributeError:
    print('Info. This pytorch version is not support with tf32.')

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='data/May')
    parser.add_argument('--O', action='store_false', help="equals --fp16 --cuda_ray --exp_eye")
    parser.add_argument('--test', action='store_true', help="test mode (load model and test dataset)")
    parser.add_argument('--infer', type=str, default=None, help="inference with audio (Xhubert)")

    parser.add_argument('--test_train', action='store_true', help="test mode (load model and train dataset)")
    parser.add_argument('--data_range', type=int, nargs='*', default=[0, -1], help="data range to use")
    parser.add_argument('--workspace', type=str, default='model/May/May_trial_audio')
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--iters', type=int, default=100000, help="training iters")
    parser.add_argument('--lr', type=float, default=1e-2, help="initial learning rate")
    parser.add_argument('--lr_net', type=float, default=1e-3, help="initial learning rate")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--num_rays', type=int, default=4096 * 16, help="num rays sampled per image for each training step")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=16, help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=16, help="num steps sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=0, help="num steps up-sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")

    parser.add_argument('--warmup_step', type=int, default=10000, help="warm up steps")
    parser.add_argument('--amb_aud_loss', type=int, default=1, help="use ambient aud loss")
    parser.add_argument('--amb_eye_loss', type=int, default=1, help="use ambient eye loss")
    parser.add_argument('--unc_loss', type=int, default=1, help="use uncertainty loss")
    parser.add_argument('--lambda_amb', type=float, default=1e-4, help="lambda for ambient loss")
    parser.add_argument('--pyramid_loss', type=int, default=0, help="use perceptual loss")

    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")

    parser.add_argument('--bg_img', type=str, default='', help="background image")
    parser.add_argument('--fbg', action='store_true', help="frame-wise bg")
    parser.add_argument('--exp_eye', action='store_true', help="explicitly control the eyes")
    parser.add_argument('--fix_eye', type=float, default=-1, help="fixed eye area, negative to disable, set to 0-0.3 for a reasonable eye")
    parser.add_argument('--smooth_eye', action='store_true', help="smooth the eye area sequence")
    parser.add_argument('--bs_area', type=str, default="upper", help="upper or eye")
    parser.add_argument('--au45', action='store_true', help="use openface au45")
    parser.add_argument('--torso_shrink', type=float, default=0.8, help="shrink bg coords to allow more flexibility in deform")

    parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
    parser.add_argument('--preload', type=int, default=0, help="0 means load data from disk on-the-fly, 1 means preload to CPU, 2 means GPU.")
    parser.add_argument('--bound', type=float, default=1, help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=4, help="scale camera location into box[-bound, bound]^3")
    parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")
    parser.add_argument('--dt_gamma', type=float, default=1/256, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.05, help="minimum near distance for camera")
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied (sigma)")
    parser.add_argument('--density_thresh_torso', type=float, default=0.01, help="threshold for density grid to be occupied (alpha)")
    parser.add_argument('--patch_size', type=int, default=1, help="[experimental] render patches in training, so as to apply LPIPS loss. 1 means disabled, use [64, 32, 16] to enable")

    parser.add_argument('--init_lips', action='store_true', help="init lips region")
    parser.add_argument('--finetune_lips', action='store_true', help="use LPIPS and landmarks to fine tune lips region")
    parser.add_argument('--smooth_lips', action='store_true', help="smooth the enc_a in a exponential decay way...")
    parser.add_argument('--bs_loss', action='store_true', help="only use in second phase (training face)")

    parser.add_argument('--torso', action='store_true', help="fix head and train torso")
    parser.add_argument('--head_ckpt', type=str, default='', help="head model")
    parser.add_argument('--torso_ckpt', type=str, default='', help="torso checkpoint for face training")

    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=450, help="GUI width")
    parser.add_argument('--H', type=int, default=450, help="GUI height")
    parser.add_argument('--radius', type=float, default=3.35, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=21.24, help="default GUI camera fovy")
    parser.add_argument('--max_spp', type=int, default=1, help="GUI rendering max sample per pixel")

    parser.add_argument('--att', type=int, default=2, help="audio attention mode (0 = turn off, 1 = left-direction, 2 = bi-direction)")
    parser.add_argument('--aud', type=str, default='', help="audio source (empty will load the default, else should be a path to a npy file)")
    parser.add_argument('--emb', action='store_true', help="use audio class + embedding instead of logits")
    parser.add_argument('--portrait', action='store_true', help="only render face")
    parser.add_argument('--ind_dim', type=int, default=4, help="individual code dim, 0 to turn off")
    parser.add_argument('--ind_num', type=int, default=20000, help="number of individual codes, should be larger than training dataset size")

    parser.add_argument('--ind_dim_torso', type=int, default=8, help="individual code dim, 0 to turn off")

    parser.add_argument('--amb_dim', type=int, default=2, help="ambient dimension")
    parser.add_argument('--part', action='store_true', help="use partial training data (1/10)")
    parser.add_argument('--part2', action='store_true', help="use partial training data (first 15s)")

    parser.add_argument('--train_camera', action='store_true', help="optimize camera pose")
    parser.add_argument('--smooth_path', action='store_true', help="brute-force smooth camera pose trajectory with a window size")
    parser.add_argument('--smooth_path_window', type=int, default=7, help="smoothing window size")

    parser.add_argument('--asr', action='store_true', help="load asr for real-time app")
    parser.add_argument('--asr_wav', type=str, default='', help="load the wav and use as input")
    parser.add_argument('--asr_play', action='store_true', help="play out the audio")

    parser.add_argument('--asr_model', type=str, default='hubert')
    parser.add_argument('--asr_save_feats', action='store_true')
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('-l', type=int, default=10)
    parser.add_argument('-m', type=int, default=50)
    parser.add_argument('-r', type=int, default=10)

    parser.add_argument('--noise_dim_pose', type=int, default=32, help="")
    parser.add_argument('--special', action='store_true', help="start audio2bs")
    parser.add_argument('--bs_start', action='store_true', help="start prediction bs to nominate")
    parser.add_argument('--bs_au45', action='store_true', help="")
    parser.add_argument('--cvae', action='store_true', help="")

    opt = parser.parse_args()

    opt.cvae = True
    opt.bs_au45 = True

    if opt.O:
        opt.fp16 = True
        opt.exp_eye = True

    opt.cuda_ray = True

    if opt.patch_size > 1:
        assert opt.num_rays % (opt.patch_size ** 2) == 0, "patch_size ** 2 should be dividable by num_rays."

    print(opt)

    seed_everything(opt.seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Face/lip training: load torso_ckpt and freeze TORSO_FREEZE_PREFIXES.
    TORSO_FREEZE_PREFIXES = (
        'anchor_points',
        'au2BsNet',
        'au2PoseNet',
        'density_grid_torso',
        'individual_codes_torso',
        'torso_deform_net',
        'torso_encoder',
        'torso_net',
    )

    def extract_torso_state_dict(model_dict):
        return {k: v for k, v in model_dict.items()
                if k.split('.')[0] in TORSO_FREEZE_PREFIXES}

    model = NeRFNetwork(opt)
    torso_dict = None
    if opt.special and opt.torso_ckpt != '':
        model_dict_all = torch.load(opt.torso_ckpt, map_location='cpu')['model']
        torso_dict = extract_torso_state_dict(model_dict_all)

        if opt.finetune_lips is not True:
            missing_keys, unexpected_keys = model.load_state_dict(torso_dict, strict=False)

            if len(missing_keys) > 0:
                print(f"[WARN] missing keys: {missing_keys}")
            if len(unexpected_keys) > 0:
                print(f"[WARN] unexpected keys: {unexpected_keys}")

            for k, v in model.named_parameters():
                if k in torso_dict:
                    print(f'[INFO] freeze {k}, {v.shape}')
                    v.requires_grad = False

    criterion_audio2attr = torch.nn.L1Loss(reduction='mean')
    criterion = torch.nn.L1Loss(reduction='none')

    if opt.test:
        train_loader = NeRFDataset(opt, device=device, type='train').dataloader()
        metrics = [PSNRMeter(), LPIPSMeter(device=device), LMDMeter(backend='fan'),
                   LSEMeter(metric='contrast', device=device), LSEMeter(metric='difference', device=device)]

        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace,
                          criterion=criterion, fp16=opt.fp16, metrics=metrics, use_checkpoint=opt.ckpt)

        if opt.test_train:
            test_set = NeRFDataset(opt, device=device, type='train')
            test_set.training = False
            test_set.num_rays = -1
            test_loader = test_set.dataloader()
        else:
            test_loader = NeRFDataset(opt, device=device, type='test').dataloader()

        model.aud_features = test_loader._data.auds
        model.eye_areas = test_loader._data.eye_area

        if opt.gui:
            from syncanimation_nerf.gui import NeRFGUI
            with NeRFGUI(opt, trainer, test_loader) as gui:
                gui.render()
        else:
            trainer.test(test_loader, train_loader)

    else:
        if opt.torso:
            lr_map = 1e-4
        elif opt.finetune_lips is not True:
            lr_map = 1e-2
        else:
            lr_map = 1e-4
        optimizer = lambda model: torch.optim.AdamW(
            model.get_params(opt.lr, lr_map, opt.lr_net), betas=(0, 0.99), eps=1e-8)

        train_loader = NeRFDataset(opt, device=device, type='train').dataloader()

        assert len(train_loader) < opt.ind_num, \
            f"[ERROR] dataset too many frames: {len(train_loader)}, please increase --ind_num to this number!"

        model.aud_features = train_loader._data.auds
        model.eye_area = train_loader._data.eye_area
        model.poses = train_loader._data.poses

        if opt.finetune_lips:
            scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(
                optimizer, lambda iter: 0.05 ** (iter / opt.iters))
        else:
            scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(
                optimizer, lambda iter: 0.5 ** (iter / opt.iters))

        metrics = [PSNRMeter(), LPIPSMeter(device=device), LMDMeter(backend='fan')]

        eval_interval = max(1, int(5000 / len(train_loader)))
        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace,
                          optimizer=optimizer, criterion=criterion, criterion_audio2attr=criterion_audio2attr,
                          ema_decay=None, fp16=opt.fp16, lr_scheduler=scheduler,
                          scheduler_update_every_step=True, metrics=metrics,
                          use_checkpoint=opt.ckpt, eval_interval=eval_interval,
                          torso_dic=torso_dict if torso_dict is not None else None)
        with open(os.path.join(opt.workspace, 'opt.txt'), 'a') as f:
            f.write(str(opt))

        valid_loader = NeRFDataset(opt, device=device, type='val', downscale=1).dataloader()

        max_epochs = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
        print(f'[INFO] max_epoch = {max_epochs}')
        trainer.train(train_loader, valid_loader, max_epochs)
