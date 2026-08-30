import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import get_encoder
from .renderer import NeRFRenderer


from .audio_pose_syncer import AudioPoseSyncer
from .audio_emotion_syncer import AudioEmotionSyncer


class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, leakyReLU=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding),
            nn.BatchNorm2d(cout)
        )
        if leakyReLU:
            self.act = nn.LeakyReLU(0.02)
        else:
            self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)


class AudioAttNet(nn.Module):
    def __init__(self, dim_aud=64, seq_len=8):
        super(AudioAttNet, self).__init__()
        self.seq_len = seq_len
        self.dim_aud = dim_aud
        self.attentionConvNet = nn.Sequential(  # b x subspace_dim x seq_len
            nn.Conv1d(self.dim_aud, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(8, 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(4, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(2, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True)
        )
        self.attentionNet = nn.Sequential(
            nn.Linear(in_features=self.seq_len, out_features=self.seq_len, bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x: [1, seq_len, dim_aud]
        y = x.permute(0, 2, 1)  # [1, dim_aud, seq_len]
        y = self.attentionConvNet(y)
        y = self.attentionNet(y.view(1, self.seq_len)).view(1, self.seq_len, 1)
        return torch.sum(y * x, dim=1)  # [1, dim_aud]


class AudioEncoder(nn.Module):
    def __init__(self):
        super(AudioEncoder, self).__init__()

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0), )

    def forward(self, x):
        out = self.audio_encoder(x)
        out = out.squeeze(2).squeeze(2)

        return out


class AudioNet(nn.Module):
    def __init__(self, dim_in=29, dim_aud=64, win_size=16):
        super(AudioNet, self).__init__()
        self.win_size = win_size
        self.dim_aud = dim_aud
        self.encoder_conv = nn.Sequential(  # n x 29 x 16
            nn.Conv1d(dim_in, 32, kernel_size=3, stride=2, padding=1, bias=True),  # n x 32 x 8
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1, bias=True),  # n x 32 x 4
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1, bias=True),  # n x 64 x 2
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),  # n x 64 x 1
            nn.LeakyReLU(0.02, True),
        )
        self.encoder_fc1 = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(0.02, True),
            nn.Linear(64, dim_aud),
        )

    def forward(self, x):
        half_w = int(self.win_size / 2)
        x = x[:, :, 8 - half_w:8 + half_w]
        x = self.encoder_conv(x).squeeze(-1)
        x = self.encoder_fc1(x)
        return x


class AudioNet_ave(nn.Module):
    def __init__(self, dim_in=29, dim_aud=64, win_size=16):
        super(AudioNet_ave, self).__init__()
        self.win_size = win_size
        self.dim_aud = dim_aud
        self.encoder_fc1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.02, True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.02, True),
            nn.Linear(128, dim_aud),
        )

    def forward(self, x):
        x = self.encoder_fc1(x).permute(1, 0, 2).squeeze(0)
        return x


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden,
                                 self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=False)
                       )

        self.net = nn.ModuleList(net)

    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)

        return x


class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 opt,
                 audio_dim=32,
                 ):
        super().__init__(opt)

        self.emb = self.opt.emb

        if 'esperanto' in self.opt.asr_model:
            self.audio_in_dim = 44
        elif 'deepspeech' in self.opt.asr_model:
            self.audio_in_dim = 29
        elif 'hubert' in self.opt.asr_model:
            self.audio_in_dim = 1024
        else:
            self.audio_in_dim = 32

        if self.emb:
            self.embedding = nn.Embedding(self.audio_in_dim, self.audio_in_dim)

        self.audio_dim = audio_dim
        if self.opt.asr_model == 'ave':
            self.audio_net = AudioNet_ave(self.audio_in_dim, self.audio_dim)
        else:
            self.audio_net = AudioNet(self.audio_in_dim, self.audio_dim)

        self.att = self.opt.att
        if self.att > 0:
            self.audio_att_net = AudioAttNet(self.audio_dim)

        self.num_levels = 12
        self.level_dim = 1
        self.encoder_xy, self.in_dim_xy = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels,
                                                      level_dim=self.level_dim, base_resolution=64,
                                                      log2_hashmap_size=14, desired_resolution=512 * self.bound)
        self.encoder_yz, self.in_dim_yz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels,
                                                      level_dim=self.level_dim, base_resolution=64,
                                                      log2_hashmap_size=14, desired_resolution=512 * self.bound)
        self.encoder_xz, self.in_dim_xz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels,
                                                      level_dim=self.level_dim, base_resolution=64,
                                                      log2_hashmap_size=14, desired_resolution=512 * self.bound)

        self.in_dim = self.in_dim_xy + self.in_dim_yz + self.in_dim_xz

        self.num_layers = 3
        self.hidden_dim = 64
        self.geo_feat_dim = 64
        if self.opt.au45:
            self.eye_att_net = MLP(self.in_dim, 1, 16, 2)
            self.eye_dim = 1 if self.exp_eye else 0
        else:
            if self.opt.bs_area == "upper":
                self.eye_att_net = MLP(self.in_dim, 7, 64, 2)
                self.eye_dim = 7 if self.exp_eye else 0
            elif self.opt.bs_area == "single":
                self.eye_att_net = MLP(self.in_dim, 4, 64, 2)
                self.eye_dim = 4 if self.exp_eye else 0
            elif self.opt.bs_area == "eye":
                self.eye_att_net = MLP(self.in_dim, 2, 64, 2)
                self.eye_dim = 2 if self.exp_eye else 0
        self.sigma_net = MLP(self.in_dim + self.audio_dim + self.eye_dim, 1 + self.geo_feat_dim, self.hidden_dim,
                             self.num_layers)
        self.num_layers_color = 2
        self.hidden_dim_color = 64
        self.encoder_dir, self.in_dim_dir = get_encoder('spherical_harmonics')
        self.color_net = MLP(self.in_dim_dir + self.geo_feat_dim + self.individual_dim, 3, self.hidden_dim_color,
                             self.num_layers_color)

        self.unc_net = MLP(self.in_dim, 1, 32, 2)

        self.aud_ch_att_net = MLP(self.in_dim, self.audio_dim, 64, 2)

        self.testing = False

        self.au2PoseNet = AudioPoseSyncer(audio_dim=256,
                                         audio_layers=1,
                                         pose_dim=6,
                                         hidden_dim=128,
                                         noise_dim=opt.noise_dim_pose,
                                         dropout=0.1, max_time=30)

        self.au2BsNet = AudioEmotionSyncer(bs_dim=7)

        self.max_true = 0
        self.min_true = 0
        self.mean_true = 0
        self.std_true = 0

        self.mean_pred = 0
        self.avg_pose = {}
        self.avg_bs = 0

        if self.torso or opt.special:
            self.register_parameter('anchor_points',
                                    nn.Parameter(torch.tensor(
                                        [[0.01, 0.01, 0.1, 1], [-0.1, -0.1, 0.1, 1], [0.1, -0.1, 0.1, 1]])))
            self.torso_deform_encoder, self.torso_deform_in_dim = get_encoder('frequency', input_dim=2, multires=8)
            self.anchor_encoder, self.anchor_in_dim = get_encoder('frequency', input_dim=6, multires=3)
            self.torso_deform_net = MLP(self.torso_deform_in_dim + self.anchor_in_dim + self.individual_dim_torso, 2,
                                        32, 3)

            self.torso_encoder, self.torso_in_dim = get_encoder('tiledgrid', input_dim=2, num_levels=16, level_dim=2,
                                                                base_resolution=16, log2_hashmap_size=16,
                                                                desired_resolution=2048)
            self.torso_net = MLP(
                self.torso_in_dim + self.torso_deform_in_dim + self.anchor_in_dim + self.individual_dim_torso, 4, 32, 3)

    def forward_torso(self, x, poses, c=None):
        x = x * self.opt.torso_shrink

        wrapped_anchor = self.anchor_points[None, ...] @ poses.permute(0, 2, 1).inverse()
        wrapped_anchor = (
                    wrapped_anchor[:, :, :2] / wrapped_anchor[:, :, 3, None] / wrapped_anchor[:, :, 2, None]).view(1,
                                                                                                                   -1)
        enc_anchor = self.anchor_encoder(wrapped_anchor)
        enc_x = self.torso_deform_encoder(x)

        if c is not None:
            h = torch.cat([enc_x, enc_anchor.repeat(x.shape[0], 1), c.repeat(x.shape[0], 1)], dim=-1)
        else:
            h = torch.cat([enc_x, enc_anchor.repeat(x.shape[0], 1)], dim=-1)

        dx = self.torso_deform_net(h)

        x = (x + dx).clamp(-1, 1)

        x = self.torso_encoder(x, bound=1)

        h = torch.cat([x, h], dim=-1)

        h = self.torso_net(h)

        alpha = torch.sigmoid(h[..., :1]) * (1 + 2 * 0.001) - 0.001
        color = torch.sigmoid(h[..., 1:]) * (1 + 2 * 0.001) - 0.001

        return alpha, color, dx

    @staticmethod
    @torch.jit.script
    def split_xyz(x):
        xy, yz, xz = x[:, :-1], x[:, 1:], torch.cat([x[:, :1], x[:, -1:]], dim=-1)
        return xy, yz, xz

    def encode_x(self, xyz, bound):
        # x: [N, 3], in [-bound, bound]
        N, M = xyz.shape
        xy, yz, xz = self.split_xyz(xyz)
        feat_xy = self.encoder_xy(xy, bound=bound)
        feat_yz = self.encoder_yz(yz, bound=bound)
        feat_xz = self.encoder_xz(xz, bound=bound)

        return torch.cat([feat_xy, feat_yz, feat_xz], dim=-1)

    def encode_audio(self, a):
        if a is None: return None

        if self.emb:
            a = self.embedding(a).transpose(-1, -2).contiguous()  # [1/8, 29, 16]

        enc_a = self.audio_net(a)  # [8,32]

        if self.att > 0:
            enc_a = self.audio_att_net(enc_a.unsqueeze(0))  # [1, 32]

        return enc_a

    def predict_uncertainty(self, unc_inp):
        if self.testing or not self.opt.unc_loss:
            unc = torch.zeros_like(unc_inp)
        else:
            unc = self.unc_net(unc_inp.detach())

        return unc

    def inverse_transform(self, pose_bs, mean_ar, std_ar):
        z_scores = (pose_bs - 50) / 50 * 3
        return z_scores * std_ar + mean_ar

    def quaternion_to_euler_pose(self, quaternion_pose):
        """Pose matrix -> euler angles and translation."""
        rotation_matrix = quaternion_pose[:, :3, :3]
        translation = quaternion_pose[:, :3, 3]
        euler_angles = self.rotation_matrix_to_euler(rotation_matrix)

        return euler_angles, translation

    def rotation_matrix_to_euler(self, R_mat):
        """Rotation matrix -> euler angles."""
        sy = torch.sqrt(R_mat[:, 0, 0] ** 2 + R_mat[:, 1, 0] ** 2)

        singular = sy < 1e-6

        x = torch.atan2(R_mat[:, 2, 1], R_mat[:, 2, 2])
        y = torch.atan2(-R_mat[:, 2, 0], sy)
        z = torch.atan2(R_mat[:, 1, 0], R_mat[:, 0, 0])

        x[singular] = torch.atan2(-R_mat[singular, 1, 2], R_mat[singular, 1, 1])
        y[singular] = torch.atan2(-R_mat[singular, 2, 0], sy[singular])
        z[singular] = 0

        return torch.stack((x, y, z), dim=1)

    def euler_to_rot_matrix(self, euler_angle):
        """Euler angles -> rotation matrices."""
        B = euler_angle.shape[0]

        roll = euler_angle[:, 0].reshape(-1, 1, 1)
        pitch = euler_angle[:, 1].reshape(-1, 1, 1)
        yaw = euler_angle[:, 2].reshape(-1, 1, 1)

        rot_x = torch.cat((
            torch.cat((torch.ones_like(roll), torch.zeros_like(roll), torch.zeros_like(roll)), dim=2),
            torch.cat((torch.zeros_like(roll), roll.cos(), -roll.sin()), dim=2),
            torch.cat((torch.zeros_like(roll), roll.sin(), roll.cos()), dim=2)
        ), dim=1)

        rot_y = torch.cat((
            torch.cat((pitch.cos(), torch.zeros_like(pitch), pitch.sin()), dim=2),
            torch.cat((torch.zeros_like(pitch), torch.ones_like(pitch), torch.zeros_like(pitch)), dim=2),
            torch.cat((-pitch.sin(), torch.zeros_like(pitch), pitch.cos()), dim=2)
        ), dim=1)

        rot_z = torch.cat((
            torch.cat((yaw.cos(), -yaw.sin(), torch.zeros_like(yaw)), dim=2),
            torch.cat((yaw.sin(), yaw.cos(), torch.zeros_like(yaw)), dim=2),
            torch.cat((torch.zeros_like(yaw), torch.zeros_like(yaw), torch.ones_like(yaw)), dim=2)
        ), dim=1)

        rotation_matrix = torch.bmm(rot_z, torch.bmm(rot_y, rot_x))

        return rotation_matrix

    def rodrigues_rotation_matrix(self, axis, angle):
        """Axis-angle -> rotation matrix (Rodrigues)."""
        axis = axis / torch.norm(axis)
        cos_theta = torch.cos(angle)
        sin_theta = torch.sin(angle)
        K = torch.tensor([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]], dtype=torch.float32, device=sin_theta.device)

        I = torch.eye(3, dtype=torch.float32, device=sin_theta.device)
        R = I + sin_theta * K + (1 - cos_theta) * torch.matmul(K, K)

        return R

    def modify_pose_with_rotation(self, pose, angle):
        """Apply X/Y/Z rotations (degrees) to pose."""
        angle_x = torch.deg2rad(angle[0])
        angle_y = torch.deg2rad(angle[1])
        angle_z = torch.deg2rad(angle[2])
        Rx = self.rodrigues_rotation_matrix(torch.tensor([1, 0, 0], dtype=torch.float32, device=angle.device), angle_x)
        Ry = self.rodrigues_rotation_matrix(torch.tensor([0, 1, 0], dtype=torch.float32, device=angle.device), angle_y)
        Rz = self.rodrigues_rotation_matrix(torch.tensor([0, 0, 1], dtype=torch.float32, device=angle.device), angle_z)

        combined_rotation = torch.matmul(Rz, torch.matmul(Ry, Rx))
        new_pose = pose.clone()
        new_pose[0, :3, :3] = torch.matmul(combined_rotation, pose[0, :3, :3])

        return new_pose

    def euler_pose_to_quaternion(self, euler_angles, translation):
        """Euler + translation -> pose matrices."""
        B = euler_angles.shape[0]

        rotation_matrix = self.euler_to_rot_matrix(euler_angles)
        quaternion_pose = torch.eye(4, dtype=torch.float32).repeat(B, 1, 1).to(euler_angles.device)
        quaternion_pose[:, :3, :3] = rotation_matrix
        quaternion_pose[:, :3, 3] = translation

        return quaternion_pose

    def ngp_to_nerf_matrix(self, new_pose, scale=0.33, offset=[0, 0, 0]):
        B = new_pose.shape[0]
        original_pose = torch.zeros_like(new_pose)
        original_pose[:, 1, 0] = new_pose[:, 0, 0]
        original_pose[:, 1, 1] = -new_pose[:, 0, 1]
        original_pose[:, 1, 2] = -new_pose[:, 0, 2]
        original_pose[:, 1, 3] = (new_pose[:, 0, 3] - offset[0]) / scale

        original_pose[:, 2, 0] = new_pose[:, 1, 0]
        original_pose[:, 2, 1] = -new_pose[:, 1, 1]
        original_pose[:, 2, 2] = -new_pose[:, 1, 2]
        original_pose[:, 2, 3] = (new_pose[:, 1, 3] - offset[1]) / scale

        original_pose[:, 0, 0] = new_pose[:, 2, 0]
        original_pose[:, 0, 1] = -new_pose[:, 2, 1]
        original_pose[:, 0, 2] = -new_pose[:, 2, 2]
        original_pose[:, 0, 3] = (new_pose[:, 2, 3] - offset[2]) / scale

        original_pose[:, 3, 3] = 1.0

        return original_pose

    def nerf_matrix_to_ngp_tensor(self, pose, scale=0.33, offset=[0, 0, 0]):
        B = pose.shape[0]
        new_pose = torch.zeros_like(pose)
        new_pose[:, 0, 0] = pose[:, 1, 0]
        new_pose[:, 0, 1] = -pose[:, 1, 1]
        new_pose[:, 0, 2] = -pose[:, 1, 2]
        new_pose[:, 0, 3] = pose[:, 1, 3] * scale + offset[0]

        new_pose[:, 1, 0] = pose[:, 2, 0]
        new_pose[:, 1, 1] = -pose[:, 2, 1]
        new_pose[:, 1, 2] = -pose[:, 2, 2]
        new_pose[:, 1, 3] = pose[:, 2, 3] * scale + offset[1]

        new_pose[:, 2, 0] = pose[:, 0, 0]
        new_pose[:, 2, 1] = -pose[:, 0, 1]
        new_pose[:, 2, 2] = -pose[:, 0, 2]
        new_pose[:, 2, 3] = pose[:, 0, 3] * scale + offset[2]

        new_pose[:, 3, 3] = 1.0

        return new_pose

    def get_rotation_matrix_x_tensor(self, theta):
        """4x4 rotation about X (radians)."""
        c = torch.cos(theta)
        s = torch.sin(theta)
        rotation_matrix = torch.tensor([
            [1, 0, 0, 0],
            [0, c, -s, 0],
            [0, s, c, 0],
            [0, 0, 0, 1]
        ], dtype=torch.float32, device=theta.device)

        return rotation_matrix

    def get_rotation_matrix_y_tensor(self, theta):
        """4x4 rotation about Y (radians)."""
        c = torch.cos(theta)
        s = torch.sin(theta)
        rotation_matrix = torch.tensor([
            [c, 0, s, 0],
            [0, 1, 0, 0],
            [-s, 0, c, 0],
            [0, 0, 0, 1]
        ], dtype=torch.float32, device=theta.device)

        return rotation_matrix

    def get_rotation_matrix_z_tensor(self, theta):
        """4x4 rotation about Z (radians)."""
        c = torch.cos(theta)
        s = torch.sin(theta)
        rotation_matrix = torch.tensor([
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=torch.float32, device=theta.device)

        return rotation_matrix

    def center_and_rotate_pose_rl_tensor(self, pose_matrix, center_position, rotation_angle_degrees):
        """Recenter pose and apply X/Y/Z rotations (degrees)."""
        rotation_angles_radians = torch.deg2rad(rotation_angle_degrees)

        rotation_matrix_x = self.get_rotation_matrix_x_tensor(rotation_angles_radians[0])
        rotation_matrix_y = self.get_rotation_matrix_y_tensor(rotation_angles_radians[1])
        rotation_matrix_z = self.get_rotation_matrix_z_tensor(rotation_angles_radians[2])

        new_pose_matrix = pose_matrix.clone()
        new_pose_matrix[:, :3, 3] = center_position

        new_pose_matrix = torch.matmul(rotation_matrix_z, torch.matmul(rotation_matrix_y,
                                                                       torch.matmul(rotation_matrix_x,
                                                                                    new_pose_matrix))).to(torch.float32)

        return new_pose_matrix

    def forward_audio(self, poses_bs=None, bs=None, audio_feat_pose=None,
                      audio_feat_bs=None, external_code=None, transform=None, t=None, clip=False):
        if external_code is not None:
            assert len(external_code) == 2
            external_code_pose = external_code[0]
            external_code_bs = external_code[1]
        else:
            external_code_pose = None
            external_code_bs = None

        if self.opt.test is not True:
            if self.opt.torso is True:
                euler, trans, mu_pose, logvar_pose = self.au2PoseNet(pose_bs=poses_bs, audio=audio_feat_pose,
                                                                     t=t,
                                                                     external_code=external_code_pose)
                mu_bs = torch.ones((trans.shape[0], 256), device=trans.device)
                logvar_bs = torch.ones((trans.shape[0], 256), device=trans.device)

                mu = [mu_pose, mu_bs]
                logvar = [logvar_pose, logvar_bs]

                output = torch.cat((euler, trans), dim=1)
                output_ori = output * self.std_true[:, :6] + self.mean_true[:, :6]
                euler_diff, t_diff, arkit = output_ori[:, :3], output_ori[:, 3:6], output_ori[:, 6:13]

                avg_euler, avg_t = transform[0], transform[1]
                euler = avg_euler + euler_diff
                translation = avg_t + t_diff

                matrix_pose = self.euler_pose_to_quaternion(euler, translation)
                return output, matrix_pose, mu, logvar

            elif self.opt.finetune_lips is True:
                euler, trans, mu_pose, logvar_pose = self.au2PoseNet(pose_bs=poses_bs, audio=audio_feat_pose, t=t,
                                                                     external_code=external_code_pose)
                arkit, mu_bs, logvar_bs = self.au2BsNet(audio=audio_feat_bs, bs=bs, t=t,
                                                        external_code=external_code_bs)
                mu = [mu_pose, mu_bs]
                logvar = [logvar_pose, logvar_bs]

                output = torch.cat((euler, trans, arkit), dim=1)
                output_ori = output * self.std_true[:, :13] + self.mean_true[:, :13]
                euler_diff, t_diff, arkit_diff = output_ori[:, :3], output_ori[:, 3:6], output_ori[:, 6:13]

                avg_euler, avg_t, avg_a = transform[0], transform[1], transform[2][:, :7]
                euler = avg_euler + euler_diff
                translation = avg_t + t_diff
                a = avg_a + arkit_diff

                matrix_pose = self.euler_pose_to_quaternion(euler, translation)
                return output, matrix_pose, a, mu, logvar
            else:
                arkit, mu_bs, logvar_bs = self.au2BsNet(audio=audio_feat_bs, bs=bs, t=t,
                                                        external_code=external_code_bs)

                mu_pose = torch.ones((audio_feat_bs.shape[0], self.opt.noise_dim_pose), device=audio_feat_bs.device)
                logvar_pose = torch.ones((audio_feat_bs.shape[0], self.opt.noise_dim_pose), device=audio_feat_bs.device)
                mu = [mu_pose, mu_bs]
                logvar = [logvar_pose, logvar_bs]

                output_ori = arkit * self.std_true[:, 6:] + self.mean_true[:, 6:]
                avg_a = transform[2][:, :7]
                a = avg_a + output_ori

                return arkit, a, mu, logvar

        else:
            euler, trans, mu_pose, logvar_pose = self.au2PoseNet(pose_bs=poses_bs, audio=audio_feat_pose, t=t,
                                                                 external_code=external_code_pose)
            arkit, mu_bs, logvar_bs = self.au2BsNet(audio=audio_feat_bs, bs=bs, t=t,
                                                    external_code=external_code_bs)

        mu = [mu_pose, mu_bs]
        logvar = [logvar_pose, logvar_bs]

        if transform is not None and clip is False:

            output = torch.cat((euler, trans, arkit), dim=1)

            output_ori = output * self.std_true[:, :13] + self.mean_true[:, :13]
            euler_diff, t_diff, arkit_diff = output_ori[:, :3], output_ori[:, 3:6], output_ori[:, 6:13]

            avg_euler, avg_t, avg_a = transform[0], transform[1], transform[2][:, :7]
            euler = avg_euler + euler_diff
            translation = avg_t + t_diff
            a = avg_a + arkit_diff
            matrix_pose = self.euler_pose_to_quaternion(euler, translation)

            return output, matrix_pose, a, mu, logvar

        elif transform is not None and clip is True:
            output = torch.cat((euler, trans, arkit), dim=1)
            output_ori = output * self.std_true[:, :13] + self.mean_true[:, :13]
            euler_diff, t_diff, arkit_diff = output_ori[:, :3], output_ori[:, 3:6], output_ori[:, 6:13]

            avg_euler, avg_t, avg_a = transform[0], transform[1], transform[2][:, :7]
            euler = avg_euler + euler_diff
            translation = avg_t + t_diff
            a = avg_a + arkit_diff

            matrix_pose = self.euler_pose_to_quaternion(euler, translation)
            return output, matrix_pose, a, mu, logvar
        else:
            output = torch.cat((euler, trans, arkit), dim=1)
            return output, mu, logvar

    def forward(self, x, d, enc_a, c, e=None):
        enc_x = self.encode_x(x, bound=self.bound)

        sigma_result = self.density(x, enc_a, e, enc_x)
        sigma = sigma_result['sigma']
        geo_feat = sigma_result['geo_feat']
        aud_ch_att = sigma_result['ambient_aud']
        eye_att = sigma_result['ambient_eye']

        enc_d = self.encoder_dir(d)

        if c is not None:
            h = torch.cat([enc_d, geo_feat, c.repeat(x.shape[0], 1)], dim=-1)
        else:
            h = torch.cat([enc_d, geo_feat], dim=-1)

        h_color = self.color_net(h)
        color = torch.sigmoid(h_color) * (1 + 2 * 0.001) - 0.001

        uncertainty = self.predict_uncertainty(enc_x)
        uncertainty = torch.log(1 + torch.exp(uncertainty))

        return sigma, color, aud_ch_att, eye_att, uncertainty[..., None]

    def density(self, x, enc_a, e=None, enc_x=None):
        if enc_x is None:
            enc_x = self.encode_x(x, bound=self.bound)

        enc_a = enc_a.repeat(enc_x.shape[0], 1)
        aud_ch_att = self.aud_ch_att_net(enc_x)
        enc_w = enc_a * aud_ch_att
        eye_att = None
        if e is not None:
            e = e.repeat(enc_x.shape[0], 1)
            eye_att = self.eye_att_net(enc_x)
            e = e * eye_att
            h = torch.cat([enc_x, enc_w, e], dim=-1)
        else:
            h = torch.cat([enc_x, enc_w], dim=-1)

        h = self.sigma_net(h)

        sigma = torch.exp(h[..., 0])
        geo_feat = h[..., 1:]

        return {
            'sigma': sigma,
            'geo_feat': geo_feat,
            'ambient_aud': aud_ch_att.norm(dim=-1, keepdim=True),
            'ambient_eye': eye_att.norm(dim=-1, keepdim=True),
        }

    def get_params(self, lr, lr_map, lr_net, wd=0):

        if self.torso:
            params = [
                {'params': self.au2BsNet.parameters(), 'lr': lr_map, 'weight_decay': 1e-4},
                {'params': self.au2PoseNet.parameters(), 'lr': lr_map, 'weight_decay': 1e-4},

                {'params': self.torso_encoder.parameters(), 'lr': lr},
                {'params': self.torso_deform_encoder.parameters(), 'lr': lr, 'weight_decay': wd},
                {'params': self.torso_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
                {'params': self.torso_deform_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
                {'params': self.anchor_points, 'lr': lr_net, 'weight_decay': wd}
            ]

            if self.individual_dim_torso > 0:
                params.append({'params': self.individual_codes_torso, 'lr': lr_net, 'weight_decay': wd})

            return params

        params = [
            {'params': self.audio_net.parameters(), 'lr': lr_net, 'weight_decay': wd},

            {'params': self.encoder_xy.parameters(), 'lr': lr},
            {'params': self.encoder_yz.parameters(), 'lr': lr},
            {'params': self.encoder_xz.parameters(), 'lr': lr},

            {'params': self.sigma_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
            {'params': self.color_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
        ]
        if self.opt.finetune_lips is not True:
            params.append({'params': self.au2BsNet.parameters(), 'lr': lr_map, 'weight_decay': 1e-4})
            params.append({'params': self.au2PoseNet.parameters(), 'lr': lr_map, 'weight_decay': 1e-4})

        if self.att > 0:
            params.append({'params': self.audio_att_net.parameters(), 'lr': lr_net * 5, 'weight_decay': 0.0001})
        if self.emb:
            params.append({'params': self.embedding.parameters(), 'lr': lr})
        if self.individual_dim > 0:
            params.append({'params': self.individual_codes, 'lr': lr_net, 'weight_decay': wd})
        if self.train_camera:
            params.append({'params': self.camera_dT, 'lr': 1e-5, 'weight_decay': 0})
            params.append({'params': self.camera_dR, 'lr': 1e-5, 'weight_decay': 0})

        params.append({'params': self.aud_ch_att_net.parameters(), 'lr': lr_net, 'weight_decay': wd})
        params.append({'params': self.unc_net.parameters(), 'lr': lr_net, 'weight_decay': wd})
        params.append({'params': self.eye_att_net.parameters(), 'lr': lr_net, 'weight_decay': wd})

        return params