"""Audio-driven head pose (SyncAnimation Sec. 3.1)."""
import torch
import torch.nn.functional as F
from torch import nn

from .layers import FCNormRelu, ConvNormRelu


class AudioCondPoseEncoder(nn.Module):
    def __init__(self, audio_feats=128, cfg=None) -> None:
        super().__init__()

        leaky = True
        norm = "BN"
        out_channels = 128 * 2
        hidden_channels = 384
        in_channels = 3 * 2 + 7 + audio_feats

        self.block1 = FCNormRelu(in_features=in_channels,
                                 out_features=hidden_channels, norm=norm, leaky=leaky)

        self.block2 = FCNormRelu(in_features=hidden_channels + audio_feats,
                                 out_features=hidden_channels, norm=norm, leaky=leaky)

        self.block3 = FCNormRelu(in_features=hidden_channels + audio_feats,
                                 out_features=hidden_channels, norm=norm, leaky=leaky)

        self.block4 = FCNormRelu(in_features=hidden_channels + audio_feats,
                                 out_features=hidden_channels, norm=norm, leaky=leaky)

        self.block5 = FCNormRelu(in_features=hidden_channels + audio_feats,
                                 out_features=hidden_channels, norm=norm, leaky=leaky)

        self.block6 = FCNormRelu(in_features=hidden_channels + audio_feats,
                                 out_features=hidden_channels, norm=norm, leaky=leaky)

        self.block7 = FCNormRelu(in_features=hidden_channels + audio_feats,
                                 out_features=out_channels, norm=norm, leaky=leaky)

    def forward(self, x, condition):
        x = torch.cat([x, condition], dim=1)
        x = self.block1(x)

        x = torch.cat([x, condition], dim=1)
        x = self.block2(x)

        x = torch.cat([x, condition], dim=1)
        x = self.block3(x)

        x = torch.cat([x, condition], dim=1)
        x = self.block4(x)

        x = torch.cat([x, condition], dim=1)
        x = self.block5(x)

        x = torch.cat([x, condition], dim=1)
        x = self.block6(x)

        x = torch.cat([x, condition], dim=1)
        x = self.block7(x)

        mu = x[:, 0::2]
        logvar = x[:, 1::2]
        return mu, logvar


class AudioCondPoseDecoder(nn.Module):
    def __init__(self, audio_feats=128, noise_dim=128) -> None:
        super().__init__()

        leaky = True
        norm = "BN"
        hidden_channels = 384
        in_channels = noise_dim + audio_feats
        out_channels = 3 + 3 + 7
        self.block1 = FCNormRelu(in_features=in_channels, out_features=hidden_channels, norm=norm, leaky=leaky)
        self.block2 = FCNormRelu(in_features=hidden_channels + audio_feats, out_features=hidden_channels, norm=norm, leaky=leaky)
        self.block3 = FCNormRelu(in_features=hidden_channels + audio_feats, out_features=hidden_channels, norm=norm, leaky=leaky)
        self.block4 = FCNormRelu(in_features=hidden_channels + audio_feats, out_features=hidden_channels, norm=norm, leaky=leaky)
        self.block5 = FCNormRelu(in_features=hidden_channels + audio_feats, out_features=hidden_channels, norm=norm, leaky=leaky)
        self.block6 = FCNormRelu(in_features=hidden_channels + audio_feats, out_features=hidden_channels, norm=norm, leaky=leaky)

        self.output_layer = FCNormRelu(in_features=hidden_channels + audio_feats, out_features=out_channels, norm=norm, leaky=leaky)

    def forward(self, x, condition):
        x = torch.cat([x, condition], dim=1)
        x = self.block1(x)

        x = torch.cat([x, condition], dim=1)
        x = self.block2(x)

        x = torch.cat([x, condition], dim=1)
        x = self.block3(x)

        x = torch.cat([x, condition], dim=1)
        x = self.block4(x)

        x = torch.cat([x, condition], dim=1)
        x = self.block5(x)

        x = torch.cat([x, condition], dim=1)
        x = self.block6(x)

        x = torch.cat([x, condition], dim=1)
        x = self.output_layer(x)

        return x


class AudioCondPoseAutoencoder(nn.Module):
    def __init__(self, layer=3, hidden_nc=256, dropout=0.15) -> None:
        super().__init__()
        self.layer = layer
        self.encoder = AudioCondPoseEncoder(audio_feats=hidden_nc // 2)
        self.decoder = AudioCondPoseDecoder(audio_feats=hidden_nc // 2)

        self.audio_first = nn.Sequential(torch.nn.Conv1d(hidden_nc, hidden_nc // 2, kernel_size=7, padding=0, bias=True),
                                                 nn.InstanceNorm1d(hidden_nc)
                                                )
        self.norm4 = nn.InstanceNorm1d(hidden_nc // 2)
        for i in range(layer):
            net = nn.Sequential(nn.LeakyReLU(),
                                torch.nn.Conv1d(hidden_nc // 2, hidden_nc // 2, kernel_size=3, padding=0, dilation=3),
                                nn.InstanceNorm1d(hidden_nc // 2),
                                nn.LeakyReLU(0.1)
                                )
            setattr(self, 'audio_encoder' + str(i), net)
        self.pooling = nn.AdaptiveAvgPool1d(1)

        self.audio_dp = nn.Dropout(dropout)



    def forward(self, x, audio_features, external_code=None):

        out = self.audio_first(audio_features)
        for i in range(self.layer):
            audio_encoder = getattr(self, 'audio_encoder' + str(i))
            out = audio_encoder(out) + self.norm4(out[:, :, 3:-3])
            out = self.audio_dp(out)

        out = self.pooling(out)
        condition = out.view(out.shape[0], -1)


        if external_code is not None:
            x_recon = self.decoder(external_code, condition)
            eluer, translation, arkit = x_recon[:, :3], x_recon[:, 3:6], x_recon[:, 6:]

            return eluer, translation, arkit, 0, 1
        mu, logvar = self.encoder(x, condition)

        eps = torch.randn(logvar.shape, device=logvar.device)
        code = mu + torch.exp(0.5 * logvar) * eps

        x_recon = self.decoder(code, condition)
        eluer, translation, arkit = x_recon[:, :3], x_recon[:, 3:6], x_recon[:, 6:]

        return eluer, translation, arkit, mu.squeeze(-1), logvar.squeeze(-1)

class PoseVAEEncoder(nn.Module):
    def __init__(self, in_channels=3 * 2 + 7, noise_dim=32) -> None:
        super().__init__()

        leaky = True
        norm = "BN"
        out_channels = noise_dim * 2
        hidden_channels = 256
        in_channels = in_channels

        self.block1 = FCNormRelu(in_features=in_channels,
                                 out_features=hidden_channels, norm=norm, leaky=leaky)

        self.block2 = FCNormRelu(in_features=hidden_channels,
                                 out_features=hidden_channels, norm=norm, leaky=leaky)

        self.block3 = FCNormRelu(in_features=hidden_channels,
                                 out_features=hidden_channels, norm=norm, leaky=leaky)

        self.block4 = FCNormRelu(in_features=hidden_channels,
                                 out_features=hidden_channels, norm=norm, leaky=leaky)

        self.block5 = FCNormRelu(in_features=hidden_channels,
                                 out_features=hidden_channels, norm=norm, leaky=leaky)

        self.block6 = FCNormRelu(in_features=hidden_channels,
                                 out_features=hidden_channels, norm=norm, leaky=leaky)

        self.block7 = FCNormRelu(in_features=hidden_channels,
                                 out_features=out_channels, norm=norm, leaky=leaky)

    def forward(self, x):
        x = self.block1(x)

        x = self.block2(x)

        x = self.block3(x)

        x = self.block4(x)

        x = self.block5(x)

        x = self.block6(x)

        x = self.block7(x)
        mu = x[:, 0::2]
        logvar = x[:, 1::2]
        return mu, logvar


class PoseVAEDecoder(nn.Module):
    def __init__(self, in_channels=32, out_channels=6) -> None:
        super().__init__()

        leaky = True
        norm = "BN"
        hidden_channels = 256
        out_channels = out_channels
        in_channels = in_channels

        self.block1 = FCNormRelu(in_features=in_channels, out_features=hidden_channels, norm=norm, leaky=leaky)
        self.block2 = FCNormRelu(in_features=hidden_channels, out_features=hidden_channels, norm=norm, leaky=leaky)
        self.block3 = FCNormRelu(in_features=hidden_channels, out_features=hidden_channels, norm=norm, leaky=leaky)
        self.block4 = FCNormRelu(in_features=hidden_channels, out_features=hidden_channels, norm=norm, leaky=leaky)
        self.block5 = FCNormRelu(in_features=hidden_channels, out_features=hidden_channels, norm=norm, leaky=leaky)
        self.block6 = FCNormRelu(in_features=hidden_channels, out_features=hidden_channels, norm=norm, leaky=leaky)

        self.output_layer = FCNormRelu(in_features=hidden_channels, out_features=out_channels, norm=norm, leaky=leaky)

    def forward(self, x):
        x = self.block1(x)

        x = self.block2(x)

        x = self.block3(x)

        x = self.block4(x)

        x = self.block5(x)

        x = self.block6(x)

        x = self.output_layer(x)

        return x


class PoseVAE(nn.Module):
    def __init__(self, in_channels=13, layer=3, noise_dim=32, dropout=0) -> None:
        super().__init__()
        self.layer = layer
        self.encoder = PoseVAEEncoder(in_channels=in_channels, noise_dim=noise_dim)
        self.decoder = PoseVAEDecoder(in_channels=noise_dim, out_channels=in_channels)

    def forward(self, x=None, external_code=None):

        if external_code is not None:
            x_recon = self.decoder(external_code)

            return x_recon, 0, 1

        mu, logvar = self.encoder(x)

        eps = torch.randn(logvar.shape, device=logvar.device)
        code = mu + torch.exp(0.5 * logvar) * eps

        x_recon = self.decoder(code)


        return x_recon, mu, logvar

class UNet_1D_bs(nn.Module):
    def __init__(self, audio_dim=256, pose_dim=13, hidden_dim=256, time_feats=2):
        super().__init__()

        leaky = True
        norm = "BN"
        in_features = audio_dim + pose_dim

        self.e0 = FCNormRelu(in_features=in_features, out_features=hidden_dim, norm=norm, leaky=leaky)
        self.e1 = FCNormRelu(in_features=hidden_dim, out_features=hidden_dim // 2, norm=norm, leaky=leaky)
        self.e2 = FCNormRelu(in_features=hidden_dim // 2, out_features=hidden_dim // 4, norm=norm, leaky=leaky)
        self.e3 = FCNormRelu(in_features=hidden_dim // 4, out_features=hidden_dim // 8, norm=norm, leaky=leaky)
        self.e4 = FCNormRelu(in_features=hidden_dim // 8, out_features=hidden_dim // 16, norm=norm, leaky=leaky)
        self.e5 = FCNormRelu(in_features=hidden_dim // 16, out_features=hidden_dim // 32, norm=norm, leaky=leaky)
        self.e6 = FCNormRelu(in_features=hidden_dim // 32, out_features=hidden_dim // 64, norm=norm, leaky=leaky)

        self.d5 = FCNormRelu(in_features=hidden_dim // 32, out_features=hidden_dim // 16, norm=norm, leaky=leaky)
        self.d4 = FCNormRelu(in_features=hidden_dim // 16, out_features=hidden_dim // 8, norm=norm, leaky=leaky)
        self.d3 = FCNormRelu(in_features=hidden_dim // 8, out_features=hidden_dim // 4, norm=norm, leaky=leaky)
        self.d2 = FCNormRelu(in_features=hidden_dim // 4, out_features=hidden_dim // 2, norm=norm, leaky=leaky)
        self.d1 = FCNormRelu(in_features=hidden_dim // 2, out_features=hidden_dim, norm=norm, leaky=leaky)

        self.time_embed_e0 = nn.Linear(time_feats, hidden_dim)
        self.time_embed_e1 = nn.Linear(time_feats, hidden_dim // 2)
        self.time_embed_e2 = nn.Linear(time_feats, hidden_dim // 4)
        self.time_embed_e3 = nn.Linear(time_feats, hidden_dim // 8)

    def forward(self, x, t):
        e0 = self.e0(x) + self.time_embed_e0(t)
        e1 = self.e1(e0)+ self.time_embed_e1(t)
        e2 = self.e2(e1)+ self.time_embed_e2(t)
        e3 = self.e3(e2) + self.time_embed_e3(t)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)

        d5 = self.d5(F.interpolate(e6.unsqueeze(1), size=e5.size(-1), mode='linear').squeeze(1) + e5)
        d4 = self.d4(F.interpolate(d5.unsqueeze(1), size=e4.size(-1), mode='linear').squeeze(1) + e4) + self.time_embed_e3(t)
        d3 = self.d3(F.interpolate(d4.unsqueeze(1), size=e3.size(-1), mode='linear').squeeze(1) + e3) + self.time_embed_e2(t)
        d2 = self.d2(F.interpolate(d3.unsqueeze(1), size=e2.size(-1), mode='linear').squeeze(1) + e2) + self.time_embed_e1(t)
        d1 = self.d1(F.interpolate(d2.unsqueeze(1), size=e1.size(-1), mode='linear').squeeze(1) + e1) + self.time_embed_e0(t)

        return d1

class PoseUNet1D(nn.Module):
    def __init__(self, audio_dim=256, pose_dim=13, hidden_dim=256):
        super().__init__()

        leaky = True
        norm = "BN"
        in_features = audio_dim + pose_dim + 256 + 256

        self.e0 = FCNormRelu(in_features=in_features, out_features=hidden_dim, norm=norm, leaky=leaky)
        self.e1 = FCNormRelu(in_features=hidden_dim, out_features=hidden_dim // 2, norm=norm, leaky=leaky)
        self.e2 = FCNormRelu(in_features=hidden_dim // 2, out_features=hidden_dim // 4, norm=norm, leaky=leaky)
        self.e3 = FCNormRelu(in_features=hidden_dim // 4, out_features=hidden_dim // 8, norm=norm, leaky=leaky)
        self.e4 = FCNormRelu(in_features=hidden_dim // 8, out_features=hidden_dim // 16, norm=norm, leaky=leaky)
        self.e5 = FCNormRelu(in_features=hidden_dim // 16, out_features=hidden_dim // 32, norm=norm, leaky=leaky)
        self.e6 = FCNormRelu(in_features=hidden_dim // 32, out_features=hidden_dim // 64, norm=norm, leaky=leaky)

        self.d5 = FCNormRelu(in_features=hidden_dim // 32, out_features=hidden_dim // 16, norm=norm, leaky=leaky)
        self.d4 = FCNormRelu(in_features=hidden_dim // 16, out_features=hidden_dim // 8, norm=norm, leaky=leaky)
        self.d3 = FCNormRelu(in_features=hidden_dim // 8, out_features=hidden_dim // 4, norm=norm, leaky=leaky)
        self.d2 = FCNormRelu(in_features=hidden_dim // 4, out_features=hidden_dim // 2, norm=norm, leaky=leaky)
        self.d1 = FCNormRelu(in_features=hidden_dim // 2, out_features=hidden_dim, norm=norm, leaky=leaky)

    def forward(self, x):
        e0 = self.e0(x)
        e1 = self.e1(e0)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)

        d5 = self.d5(F.interpolate(e6.unsqueeze(1), size=e5.size(-1), mode='linear').squeeze(1) + e5)
        d4 = self.d4(F.interpolate(d5.unsqueeze(1), size=e4.size(-1), mode='linear').squeeze(1) + e4)
        d3 = self.d3(F.interpolate(d4.unsqueeze(1), size=e3.size(-1), mode='linear').squeeze(1) + e3)
        d2 = self.d2(F.interpolate(d3.unsqueeze(1), size=e2.size(-1), mode='linear').squeeze(1) + e2)
        d1 = self.d1(F.interpolate(d2.unsqueeze(1), size=e1.size(-1), mode='linear').squeeze(1) + e1)

        return d1

class PoseAttrMapNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, norm="BN", leaky=True):
        super(PoseAttrMapNet, self).__init__()

        self.euler_fc = FCNormRelu(
            in_features=in_channels,
            out_features=hidden_channels,
            norm=norm,
            leaky=leaky
        )
        self.euler_yr = nn.Linear(hidden_channels, 2)
        self.euler_p = nn.Linear(hidden_channels, 1)

        self.translation_fc = FCNormRelu(
            in_features=in_channels,
            out_features=hidden_channels,
            norm=norm,
            leaky=leaky
        )
        self.translation_xy = nn.Linear(hidden_channels, 2)
        self.translation_z = nn.Linear(hidden_channels, 1)

    def forward(self, pose_output):
        euler_feat = self.euler_fc(pose_output)
        pitch = self.euler_p(euler_feat)
        euler = self.euler_yr(euler_feat)

        euler = torch.cat((pitch, euler), dim=1)

        translation_feat = self.translation_fc(pose_output)
        xy = self.translation_xy(translation_feat)
        z = self.translation_z(translation_feat)

        translation = torch.cat((xy, z), dim=1)

        return euler, translation

class AudioPoseSyncer(nn.Module):
    def __init__(self, audio_dim=256, audio_layers=3, pose_dim=13, hidden_dim=256, dropout=0.1, max_time=30, noise_dim=32):
        super(AudioPoseSyncer, self).__init__()

        self.autoencoder = PoseVAE(in_channels=pose_dim, noise_dim=noise_dim)
        self.unet_pose = PoseUNet1D(audio_dim=audio_dim // 2, pose_dim=6, hidden_dim=hidden_dim)

        self.label_mapp = FCNormRelu(in_features=5,
                                     out_features=256,
                                     norm="BN",
                                     leaky=True,
                                     drop_out=0.60)

        self.pitch_mapp = FCNormRelu(in_features=1,
                                     out_features=256,
                                     norm="BN",
                                     leaky=True,
                                     drop_out=0.60)


        hidden_nc = audio_dim // 2
        self.audio_first = FCNormRelu(in_features=audio_dim,
                                      out_features=hidden_nc,
                                      norm="BN",
                                      leaky=True
                                      )
        self.norm4 = nn.BatchNorm1d(hidden_nc)
        self.audio_layers = audio_layers
        for i in range(audio_layers):
            net = nn.Sequential(FCNormRelu(in_features=hidden_nc,
                                           out_features=hidden_nc,
                                           norm="BN",
                                           leaky=True
                                           )
                                )
            setattr(self, 'audio_encoder' + str(i), net)

        self.attr_map_net = PoseAttrMapNet(in_channels=hidden_dim, hidden_channels=hidden_dim // 2)

    def forward(self, audio, pose_bs=None, t=None, external_code=None):
        """Audio + pose -> euler, translation, VAE mu/logvar."""
        pose_bs_diff, pose_diff_true = pose_bs[0], pose_bs[1]

        pose_bs, mu_pred, logvar_pred = self.autoencoder(x=pose_bs_diff, external_code=external_code)

        audio_out = self.audio_first(audio)
        for i in range(self.audio_layers):
            audio_encoder = getattr(self, 'audio_encoder' + str(i))
            audio_out = audio_encoder(audio_out) + self.norm4(audio_out)

        pose_bs_expanded = pose_bs.expand(audio.size(0), -1)
        pose_diff_true_expanded = pose_diff_true.expand(audio.size(0), -1)
        pose_diff_true_map = self.label_mapp(pose_diff_true_expanded[:, 1:])
        pitch_diff_true_map = self.pitch_mapp(pose_diff_true_expanded[:, 0].unsqueeze(1))

        pose_diff_true_expanded = torch.cat([pitch_diff_true_map, pose_diff_true_map], dim=1)
        pose_input = torch.cat([audio_out, pose_bs_expanded[:, :],pose_diff_true_expanded], dim=1)

        pose_output = self.unet_pose(pose_input)

        euler, translation = self.attr_map_net(pose_output)

        return euler, translation, mu_pred, logvar_pred
