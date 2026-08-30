import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from .audio_pose_syncer import AudioCondPoseDecoder


class au2attrNet(nn.Module):
    def __init__(self, hidden_nc, layer):
        super( au2attrNet, self).__init__()

        self.layer = layer
        nonlinearity = nn.LeakyReLU(0.1)

        self.leakyrelu = nn.LeakyReLU()

        for i in range(layer):
            net = nn.Sequential(
                nonlinearity,
                torch.nn.Linear(hidden_nc, hidden_nc),
            )
            setattr(self, 'encoder' + str(i), net)

        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.output_nc = hidden_nc

        self.norm1 = nn.LayerNorm(int(hidden_nc * 2))
        self.norm2 = nn.LayerNorm(int(hidden_nc * 2))
        self.norm3 = nn.LayerNorm(int(hidden_nc * 2))

        self.fc_e_1 = nn.Linear(hidden_nc, int(hidden_nc * 2))
        self.fc_e_2 = nn.Linear(int(hidden_nc * 2), 3)
        self.fc_t_1 = nn.Linear(hidden_nc, int(hidden_nc * 2))
        self.fc_t_2 = nn.Linear(int(hidden_nc * 2), 3)

        self.fc_a_1 = nn.Linear(hidden_nc, int(hidden_nc * 2))
        self.fc_a_2 = nn.Linear(int(hidden_nc * 2), 52)

    def forward(self, audio):

        out = audio
        for i in range(self.layer):
            model = getattr(self, 'encoder' + str(i))
            out = model(out) + out

        euler = self.fc_e_1(out)
        euler = self.norm1(self.leakyrelu(euler))
        euler = self.fc_e_2(euler)

        trans = self.fc_t_1(out)
        trans = self.norm2(self.leakyrelu(trans))
        trans = self.fc_t_2(trans)

        arkit = self.fc_a_1(out)
        arkit = self.norm3(self.leakyrelu(arkit))
        arkit = self.fc_a_2(arkit)

        return euler, trans, arkit

class au2attrNetV2(nn.Module):
    def __init__(self, hidden_nc, layer):
        super(au2attrNetV2, self).__init__()

        self.layer = layer
        nonlinearity = nn.LeakyReLU(0.1)
        self.dp = nn.Dropout(0.1)
        self.leakyrelu = nn.LeakyReLU()

        self.first = nn.Sequential(
            torch.nn.Conv1d(hidden_nc, hidden_nc, kernel_size=7, padding=0, bias=True),
            nn.InstanceNorm1d(hidden_nc),
        )

        for i in range(layer):
            net=nn.Sequential(nonlinearity,
                              torch.nn.Conv1d(hidden_nc, hidden_nc, kernel_size=3, padding=0, dilation=3),
                              nn.InstanceNorm1d(hidden_nc ),
                              nn.LeakyReLU(0.1)
            )
            setattr(self, 'encoder' + str(i), net)

        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.output_nc = hidden_nc

        self.norm1 = nn.LayerNorm(int(hidden_nc // 2))
        self.norm2 = nn.LayerNorm(int(hidden_nc // 2))
        self.norm3 = nn.LayerNorm(int(hidden_nc // 2))

        self.norm4 = nn.InstanceNorm1d(hidden_nc // 2)

        self.fc_e_1 = nn.Linear(hidden_nc, int(hidden_nc // 2))
        self.fc_e_2 = nn.Linear(int(hidden_nc // 2), 3)

        self.fc_t_1 = nn.Linear(hidden_nc, int(hidden_nc // 2))
        self.fc_t_21 = nn.Linear(int(hidden_nc // 2), 1)
        self.fc_t_22 = nn.Linear(int(hidden_nc // 2), 1)
        self.fc_t_23 = nn.Linear(int(hidden_nc // 2), 1)

        self.fc_a_1 = nn.Linear(hidden_nc, int(hidden_nc // 2))
        self.fc_a_2 = nn.Linear(int(hidden_nc // 2), 7)

        self.apply(self.init_weights)

    def init_weights(self, m):
        """Init module weights."""
        if isinstance(m, nn.Conv1d):
            torch.nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)


    def forward(self, audio):
        out = self.first(audio)
        for i in range(self.layer):
            model = getattr(self, 'encoder' + str(i))
            out = model(out) + out[:, :, 3:-3]
            out = self.dp(out)


        out = self.pooling(out)
        out = out.view(out.shape[0], -1)
        euler = self.fc_e_1(out)
        euler = self.norm1(self.leakyrelu(euler))
        euler = self.fc_e_2(euler)

        trans = self.fc_t_1(out)
        trans = self.norm2(self.leakyrelu(trans))

        trans_1 = self.fc_t_21(trans)
        trans_2 = self.fc_t_22(trans)
        trans_3 = self.fc_t_23(trans)
        trans = torch.cat([trans_1, trans_2, trans_3], dim=1)

        arkit = self.fc_a_1(out)
        arkit = self.norm3(self.leakyrelu(arkit))
        arkit = self.fc_a_2(arkit)
        return euler, trans, arkit

class au2attrNetV3(nn.Module):
    def __init__(self, hidden_nc, layer):
        super(au2attrNetV3, self).__init__()

        self.layer = layer
        nonlinearity = nn.LeakyReLU(0.1)
        self.dp = nn.Dropout(0.2)
        self.leakyrelu = nn.LeakyReLU()

        self.first = nn.Sequential(
            torch.nn.Conv1d(hidden_nc, hidden_nc // 2, kernel_size=7, padding=0, bias=True),
            nn.InstanceNorm1d(hidden_nc),
        )

        for i in range(layer):
            net=nn.Sequential(nonlinearity,
                              torch.nn.Conv1d(hidden_nc // 2, hidden_nc // 2, kernel_size=3, padding=0, dilation=3),
                              nn.InstanceNorm1d(hidden_nc // 2),
                              nn.LeakyReLU(0.1)
            )
            setattr(self, 'encoder' + str(i), net)

        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.output_nc = hidden_nc

        self.norm1 = nn.LayerNorm(int(hidden_nc // 4))
        self.norm2 = nn.LayerNorm(int(hidden_nc // 4))
        self.norm3 = nn.LayerNorm(int(hidden_nc // 4))

        self.norm4 = nn.InstanceNorm1d(hidden_nc // 2)

        self.fc_e_1 = nn.Linear(hidden_nc // 2, int(hidden_nc // 4))
        self.fc_e_2 = nn.Linear(int(hidden_nc // 4), 3)

        self.fc_t_1 = nn.Linear(hidden_nc // 2, int(hidden_nc // 4))
        self.fc_t_21 = nn.Linear(int(hidden_nc // 4), 1)
        self.fc_t_22 = nn.Linear(int(hidden_nc // 4), 1)
        self.fc_t_23 = nn.Linear(int(hidden_nc // 4), 1)

        self.fc_a_1 = nn.Linear(hidden_nc // 2, int(hidden_nc // 4))
        self.fc_a_2 = nn.Linear(int(hidden_nc // 4), 7)

        self.apply(self.init_weights)

    def xyz_to_spherical(self, xyz):
        """XYZ -> spherical (r, theta, phi)."""
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        r = torch.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = torch.acos(z / (r + 1e-8))
        phi = torch.atan2(y, x)

        spherical = torch.stack((r, theta, phi), dim=-1)
        return spherical

    def spherical_to_xyz(self, spherical):
        """Spherical (r, theta, phi) -> XYZ."""
        r, theta, phi = spherical[:, 0], spherical[:, 1], spherical[:, 2]
        x = r * torch.sin(theta) * torch.cos(phi)
        y = r * torch.sin(theta) * torch.sin(phi)
        z = r * torch.cos(theta)

        xyz = torch.stack((x, y, z), dim=-1)
        return xyz

    def init_weights(self, m):
        """Init module weights."""
        if isinstance(m, nn.Conv1d):
            torch.nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)


    def forward(self, audio):
        out = self.first(audio)
        for i in range(self.layer):
            model = getattr(self, 'encoder' + str(i))
            out = model(out) + self.norm4(out[:, :, 3:-3])
            out = self.dp(out)


        out = self.pooling(out)
        out = out.view(out.shape[0], -1)

        trans = self.fc_t_1(out)
        trans = self.norm2(self.leakyrelu(trans))
        trans = self.dp(trans)

        trans_1 = self.fc_t_21(trans)
        trans_2 = self.fc_t_22(trans)
        trans_3 = self.fc_t_23(trans)
        trans = torch.cat([trans_1, trans_2, trans_3], dim=1)

        output = trans
        return output


class au2attrNetV4(nn.Module):
    def __init__(self,
                 decoder,
                 mu_var_dim=32,
                 audio_feat_dim=256,
                 pose_dim=6,
                 hidden_dim=256,
                 arkit_dim=7):
        super(au2attrNetV4, self).__init__()

        self.decoder = decoder

        self.query_linear = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.InstanceNorm1d(1),
        )
        self.key_linear = nn.Sequential(
            nn.Linear(audio_feat_dim, hidden_dim),
            nn.InstanceNorm1d(1),
        )
        self.value_linear = nn.Sequential(
            nn.Linear(audio_feat_dim, hidden_dim),
            nn.InstanceNorm1d(1),
        )

        self.pose_output = nn.Sequential(
            nn.Linear(hidden_dim, pose_dim),
            nn.InstanceNorm1d(1),
        )

        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_feat_dim, 128),
            nn.InstanceNorm1d(1),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.InstanceNorm1d(1),
            nn.ReLU()
        )
        self.arkit_output = nn.Linear(64, arkit_dim)

    def forward(self, audio_feat, mu, var):
        latent_feat = self.decoder(mu, var)

        query = self.query_linear(latent_feat).unsqueeze(1)
        key = self.key_linear(audio_feat).unsqueeze(1)
        value = self.value_linear(audio_feat).unsqueeze(1)

        attention_weights = torch.bmm(query, key.transpose(1, 2)) / (query.shape[-1] ** 0.5)
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_out = torch.bmm(attention_weights, value).squeeze(1)

        pose = self.pose_output(attention_out)

        audio_encoded = self.audio_encoder(audio_feat)
        arkit = self.arkit_output(audio_encoded)

        return pose, arkit


class au2attrNetV5(nn.Module):
    def __init__(self,
                 audio_feat_dim=256,
                 pose_dim=6,
                 hidden_dim=256,
                 arkit_dim=7,
                 num_heads=8,
                 dropout=0.1,
                 noise_dim=128,
                 checkpoint_dir=None):
        super(au2attrNetV5, self).__init__()

        decoder = AudioCondPoseDecoder(cfg=None, noise_dim=noise_dim)
        pretrain_decoder = load_decoder(decoder, checkpoint_dir=checkpoint_dir)
        self.decoder = pretrain_decoder


        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim,
                                                     num_heads=num_heads,
                                                     dropout=dropout)

        self.query_linear = nn.Sequential(nn.Linear(pose_dim, hidden_dim),
                                          nn.InstanceNorm1d(1),
                                          nn.LeakyReLU())
        self.key_linear = nn.Sequential(nn.Linear(audio_feat_dim, hidden_dim),
                                          nn.InstanceNorm1d(1),
                                          nn.LeakyReLU())
        self.value_linear = nn.Sequential(nn.Linear(audio_feat_dim, hidden_dim),
                                          nn.InstanceNorm1d(1),
                                          nn.LeakyReLU())

        self.pose_output = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.InstanceNorm1d(1),
            nn.LeakyReLU(),
            nn.Linear(128, arkit_dim)
        )

        self.arkit_output = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.InstanceNorm1d(1),
            nn.ReLU(),
            nn.Linear(128, arkit_dim)
        )

    def forward(self, audio_feat, external_code, conditions):
        latent_feat = self.decoder(external_code, conditions)

        query = self.query_linear(latent_feat).unsqueeze(0)  # (1, batch, hidden_dim)
        key = self.key_linear(audio_feat).unsqueeze(0)  # (1, batch, hidden_dim)
        value = self.value_linear(audio_feat).unsqueeze(0)  # (1, batch, hidden_dim)

        attention_out, _ = self.cross_attention(query, key, value)  # (1, batch, hidden_dim)
        attention_out = attention_out.squeeze(0)

        pose = self.pose_output(attention_out)
        eluer, translation = pose[:, :3], pose[:, 3:]
        arkit = self.arkit_output(key.squeeze(0)) # (batch, arkit_dim)

        return eluer, translation, arkit

class Filter(nn.Module):

    def __init__(self, increase_factor=2, window_size=11, max_5=1.0, max_6=1.0):
        super(Filter, self).__init__()
        self.increase_factor = increase_factor
        self.window_size = window_size
        self.max_5 = max_5
        self.max_6 = max_6
        if self.window_size % 2 == 0:
            raise ValueError("Window size should be odd to maintain symmetry in smoothing.")

    def forward(self, bs):
        """Smooth first 5 blendshape dims; threshold last 2."""
        if bs.shape[1] < 7:
            raise ValueError("Input tensor must have at least 7 columns (features).")

        first_five = bs[:, :5]
        last_two = bs[:, 5:]

        smoothed_first_five = torch.zeros_like(first_five)
        for i in range(self.window_size // 2, first_five.shape[0] - self.window_size // 2):
            smoothed_first_five[i, :] = torch.mean(
                first_five[i - self.window_size // 2:i + self.window_size // 2 + 1, :], dim=0
            )

        means = last_two.mean(dim=0)
        stds = last_two.std(dim=0)
        dynamic_thresholds = means + 2.5 * stds

        last_two[:, 0][last_two[:, 0] > dynamic_thresholds[0]] = self.max_5 + 0.05
        last_two[:, 1][last_two[:, 1] > dynamic_thresholds[1]] = self.max_6 + 0.05


        smoothed_first_five[:self.window_size // 2, :] = first_five[:self.window_size // 2, :]
        smoothed_first_five[-self.window_size // 2:, :] = first_five[-self.window_size // 2:, :]

        bs = torch.cat([smoothed_first_five, last_two], dim=1)

        return bs





class ContrastiveVideoLoss:
    def __init__(self, n=2, temperature=0.07):
        """Contrastive loss for video continuity."""
        self.n = n
        self.temperature = temperature
        self.criterionL1 = torch.nn.L1Loss()

    def contrastive_video_loss(self, predictions, target_idx):
        """InfoNCE contrastive loss."""
        batchsize, feat_dim = predictions.shape

        target_sample = predictions[target_idx].unsqueeze(0)  # (1, 6)

        pos_indices = list(range(max(0, target_idx - self.n), min(batchsize, target_idx + self.n + 1)))
        pos_indices.remove(target_idx)
        pos_samples = predictions[pos_indices]  # (2n, 6)

        pos_sim = F.cosine_similarity(target_sample, pos_samples) / self.temperature  # (2n)

        neg_indices = [i for i in range(batchsize) if i not in pos_indices + [target_idx]]
        neg_samples = predictions[neg_indices]
        neg_sim = F.cosine_similarity(target_sample, neg_samples) / self.temperature

        pos_exp = torch.exp(pos_sim)
        neg_exp = torch.exp(neg_sim)
        contrastive_loss = -torch.log(pos_exp.sum() / (pos_exp.sum() + neg_exp.sum()))

        return contrastive_loss

    def __call__(self, predictions, target_idx, target, idx):
        """Weighted contrastive + L1 + variance + frame-diff loss."""
        if idx < 22:
            w1, w2, w3 = 5, 20, 1
        elif idx <= 26:
            w1, w2, w3 = 3, 25, 0.5
        else:
            w1, w2, w3 = 3, 1, 0


        variance_loss = torch.var(predictions[:, 5])
        if idx > 26:
            variance_loss = 0
            target_sample = predictions[target_idx, 3:5].unsqueeze(0)  # (1, 2)
            l1_loss = self.criterionL1(target_sample, target[target_idx, 3:5].unsqueeze(0))
            contrastive_loss = self.contrastive_video_loss(predictions[:, 3:6], target_idx)
        else:
            target_sample = predictions[target_idx, 3:6].unsqueeze(0)  # (1, 3)
            l1_loss = self.criterionL1(target_sample, target[target_idx, 3:6].unsqueeze(0))
            contrastive_loss = self.contrastive_video_loss(predictions[:, 3:5], target_idx)

        pred_diff = predictions[1:, 3:5] - predictions[:-1, 3:5]
        target_diff = target[1:, 3:5] - target[:-1, 3:5]

        frame_diff_loss = self.criterionL1(pred_diff, target_diff)

        frame_diff_weight = min(1.0, idx / 31)
        frame_diff_loss = frame_diff_loss * frame_diff_weight * 10
        return w1 * contrastive_loss + w2 * l1_loss + w3 * variance_loss + frame_diff_loss


class ContrastiveVideoLossV2:
    def __init__(self, n=2, temperature=0.07):
        """L1 + KL loss for pose/BS prediction."""
        self.n = n
        self.temperature = temperature
        self.criterionL1 = torch.nn.L1Loss()



    def __call__(self, predictions, mu_var, target_idx, target, idx):
        """L1 + KL divergence loss."""
        l1_loss = self.criterionL1(predictions[:, :], target[:, :])
        mu_pred, logvar_pred = mu_var[0], mu_var[1]
        kl_loss = 0.5 * (-logvar_pred + mu_pred ** 2 + torch.exp(logvar_pred) - 1).mean() * 0.1
        loss = l1_loss + kl_loss
        return loss


class BSLoss(nn.Module):
    def __init__(self):
        super(BSLoss, self).__init__()

        self.criterionL1 = torch.nn.L1Loss()
        self.range_penalty_multiplier = 0.1

    def forward(self, bs, all_bs_from_image, mu_var):
        """L1 + KL loss for blendshapes."""
        mse1 = self.criterionL1(bs[:, :6], all_bs_from_image[:, :6])
        mse2 = self.criterionL1(bs[:, 6:], all_bs_from_image[:, 6:])
        mse = self.criterionL1(bs[:, :], all_bs_from_image[:, :])
        mu_pred, logvar_pred = mu_var[0], mu_var[1]
        kl_loss = 0.5 * (-logvar_pred + mu_pred ** 2 + torch.exp(logvar_pred) - 1).mean() * 0.1
        loss = mse + kl_loss
        return loss



def load_decoder(decoder, checkpoint_dir):
    model_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('best_model_epoch_') and f.endswith('.pt')]
    epochs = [int(f.split('_')[-1].split('.')[0]) for f in model_files]
    max_epoch = max(epochs)
    latest_model_path = os.path.join(checkpoint_dir, f'best_model_epoch_{max_epoch}.pt')

    state_dict = torch.load(latest_model_path)
    decoder_params = {key.replace('decoder.', ''): value for key, value in state_dict.items() if
                      key.startswith('decoder')}

    missing_keys, unexpected_keys = decoder.load_state_dict(decoder_params, strict=True)
    if len(missing_keys) > 0:
        print(f"[WARN] missing keys: {missing_keys}")
    if len(unexpected_keys) > 0:
        print(f"[WARN] unexpected keys: {unexpected_keys}")
    for param in decoder.parameters():
        param.requires_grad = False
    return decoder

if __name__ == '__main__':
    pass
