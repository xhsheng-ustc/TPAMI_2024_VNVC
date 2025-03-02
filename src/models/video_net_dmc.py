import torch
import math
import torch.nn as nn
from pytorch_msssim import MS_SSIM
from .video_net import ME_Spynet, GDN, flow_warp, ResBlock, bilineardownsacling, UNet
from ..entropy_models.video_entropy_models import BitEstimator, GaussianEncoder
from ..layers.layers import subpel_conv3x3,conv3x3
import time
from ..entropy_models.video_entropy_models import EntropyCoder
from ..utils.stream_helper import get_downsampled_shape, encode_p, decoder_p, filesize


class FeatureExtractor(nn.Module):
    def __init__(self, channel=64):
        super().__init__()
        self.conv1 = nn.Conv2d(channel, channel, 3, stride=1, padding=1)
        self.res_block1 = ResBlock(channel)
        self.conv2 = nn.Conv2d(channel, channel, 3, stride=2, padding=1)
        self.res_block2 = ResBlock(channel)
        self.conv3 = nn.Conv2d(channel, channel, 3, stride=2, padding=1)
        self.res_block3 = ResBlock(channel)

    def forward(self, feature):
        layer1 = self.conv1(feature)
        layer1 = self.res_block1(layer1)

        layer2 = self.conv2(layer1)
        layer2 = self.res_block2(layer2)

        layer3 = self.conv3(layer2)
        layer3 = self.res_block3(layer3)

        return layer1, layer2, layer3


class MultiScaleContextFusion(nn.Module):
    def __init__(self, channel_in=64, channel_out=64):
        super().__init__()
        self.conv3_up = subpel_conv3x3(channel_in, channel_out, 2)
        self.res_block3_up = ResBlock(channel_out)
        self.conv3_out = nn.Conv2d(channel_out, channel_out, 3, padding=1)
        self.res_block3_out = ResBlock(channel_out)
        self.conv2_up = subpel_conv3x3(channel_out * 2, channel_out, 2)
        self.res_block2_up = ResBlock(channel_out)
        self.conv2_out = nn.Conv2d(channel_out * 2, channel_out, 3, padding=1)
        self.res_block2_out = ResBlock(channel_out)
        self.conv1_out = nn.Conv2d(channel_out * 2, channel_out, 3, padding=1)
        self.res_block1_out = ResBlock(channel_out)

    def forward(self, context1, context2, context3):
        context3_up = self.conv3_up(context3)
        context3_up = self.res_block3_up(context3_up)
        context3_out = self.conv3_out(context3)
        context3_out = self.res_block3_out(context3_out)
        context2_up = self.conv2_up(torch.cat((context3_up, context2), dim=1))
        context2_up = self.res_block2_up(context2_up)
        context2_out = self.conv2_out(torch.cat((context3_up, context2), dim=1))
        context2_out = self.res_block2_out(context2_out)
        context1_out = self.conv1_out(torch.cat((context2_up, context1), dim=1))
        context1_out = self.res_block1_out(context1_out)
        context1 = context1 + context1_out
        context2 = context2 + context2_out
        context3 = context3 + context3_out
        return context1, context2, context3


class ResEncoder(nn.Module):
    def __init__(self, channel_N=64, channel_M=96):
        super().__init__()
        self.conv1 = nn.Conv2d(channel_N + 3, channel_N, 3, stride=2, padding=1)
        self.gdn1 = GDN(channel_N)
        self.res1 = ResBlock(channel_N * 2, bottleneck=False, slope=0.1,
                             start_from_relu=False, end_with_relu=True)
        self.conv2 = nn.Conv2d(channel_N * 2, channel_N, 3, stride=2, padding=1)
        self.gdn2 = GDN(channel_N)
        self.res2 = ResBlock(channel_N * 2, bottleneck=False, slope=0.1,
                             start_from_relu=False, end_with_relu=True)
        self.conv3 = nn.Conv2d(channel_N * 2, channel_N, 3, stride=2, padding=1)
        self.gdn3 = GDN(channel_N)
        self.conv4 = nn.Conv2d(channel_N, channel_M, 3, stride=2, padding=1)

    def forward(self, x, context1, context2, context3):
        feature = self.conv1(torch.cat([x, context1], dim=1))
        feature = self.gdn1(feature)
        feature = self.res1(torch.cat([feature, context2], dim=1))
        feature = self.conv2(feature)
        feature = self.gdn2(feature)
        feature = self.res2(torch.cat([feature, context3], dim=1))
        feature = self.conv3(feature)
        feature = self.gdn3(feature)
        feature = self.conv4(feature)
        return feature


class ResDecoder(nn.Module):
    def __init__(self, channel_N=64, channel_M=96):
        super().__init__()
        self.up1 = subpel_conv3x3(channel_M, channel_N, 2)
        self.gdn1 = GDN(channel_N, inverse=True)
        self.up2 = subpel_conv3x3(channel_N, channel_N, 2)
        self.gdn2 = GDN(channel_N, inverse=True)
        self.res1 = ResBlock(channel_N * 2, bottleneck=False, slope=0.1,
                             start_from_relu=False, end_with_relu=True)
        self.up3 = subpel_conv3x3(channel_N * 2, channel_N, 2)
        self.gdn3 = GDN(channel_N, inverse=True)
        self.res2 = ResBlock(channel_N * 2, bottleneck=False, slope=0.1,
                             start_from_relu=False, end_with_relu=True)
        self.up4 = subpel_conv3x3(channel_N * 2, channel_N, 2)
        self.conv_out = nn.Conv2d(channel_N * 2, channel_N, 3, stride=1, padding=1)
    def forward(self, x, context1, context2, context3):
        feature = self.up1(x)
        feature = self.gdn1(feature)
        feature = self.up2(feature)
        feature = self.gdn2(feature)
        feature = self.res1(torch.cat([feature, context3], dim=1))
        feature = self.up3(feature)
        feature = self.gdn3(feature)
        feature = self.res2(torch.cat([feature, context2], dim=1))
        feature = self.up4(feature)
        feature = self.conv_out(torch.cat([feature, context1], dim=1))
        return feature


class TemporalPriorEncoder(nn.Module):
    def __init__(self, channel_N=64, channel_M=96):
        super().__init__()
        self.conv1 = nn.Conv2d(channel_N, channel_N, 3, stride=2, padding=1)
        self.gdn1 = GDN(channel_N)
        self.conv2 = nn.Conv2d(channel_N * 2, channel_M, 3, stride=2, padding=1)
        self.gdn2 = GDN(channel_M)
        self.conv3 = nn.Conv2d(channel_M + channel_N, channel_M * 3 // 2, 3, stride=2, padding=1)
        self.gdn3 = GDN(channel_M * 3 // 2)
        self.conv4 = nn.Conv2d(channel_M * 3 // 2, channel_M * 2, 3, stride=2, padding=1)

    def forward(self, context1, context2, context3):
        feature = self.conv1(context1)
        feature = self.gdn1(feature)
        feature = self.conv2(torch.cat([feature, context2], dim=1))
        feature = self.gdn2(feature)
        feature = self.conv3(torch.cat([feature, context3], dim=1))
        feature = self.gdn3(feature)
        feature = self.conv4(feature)
        return feature


class ReconGeneration(nn.Module):
    def __init__(self, channel=64):
        super().__init__()
        self.unet_1 = UNet(channel)
        self.unet_2 = UNet(channel)
        self.recon_conv = nn.Conv2d(channel, 3, 3, stride=1, padding=1)

    def forward(self, feature):
        feature = self.unet_1(feature)
        feature = self.unet_2(feature)
        recon = self.recon_conv(feature)
        return recon


class DMC(nn.Module):
    def __init__(self, win_size=11):
        super().__init__()
        channel_mv = 128
        channel_N = 64
        channel_M = 96

        self.channel_mv = channel_mv
        self.channel_N = channel_N
        self.channel_M = channel_M

        self.mv_encoder = nn.Sequential(
            nn.Conv2d(3+channel_N, channel_mv, 3, stride=2, padding=1),
            GDN(channel_mv),
            ResBlock(channel_mv, start_from_relu=False),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(channel_mv, channel_mv, 3, stride=2, padding=1),
            GDN(channel_mv),
            ResBlock(channel_mv, start_from_relu=False),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(channel_mv, channel_mv, 3, stride=2, padding=1),
            GDN(channel_mv),
            ResBlock(channel_mv, start_from_relu=False),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(channel_mv, channel_mv, 3, stride=2, padding=1),
        )

        self.mv_prior_encoder = nn.Sequential(
            nn.Conv2d(channel_mv, channel_N, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel_N, channel_N, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel_N, channel_N, 3, stride=2, padding=1),
        )

        self.mv_prior_decoder = nn.Sequential(
            nn.ConvTranspose2d(channel_N, channel_mv, 3,
                               stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(channel_mv, channel_mv * 3 // 2, 3,
                               stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(channel_mv * 3 // 2, channel_mv * 2, 3, stride=1, padding=1)
        )

        self.mv_decoder = nn.Sequential(
            nn.ConvTranspose2d(channel_mv, channel_mv, 3,
                               stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            ResBlock(channel_mv, start_from_relu=False),
            GDN(channel_mv, inverse=True),
            nn.ConvTranspose2d(channel_mv, channel_mv, 3,
                               stride=2, padding=1, output_padding=1),
            GDN(channel_mv, inverse=True),
            nn.ConvTranspose2d(channel_mv, channel_mv, 3,
                               stride=2, padding=1, output_padding=1),
            GDN(channel_mv, inverse=True),
            nn.ConvTranspose2d(channel_mv, 2, 3, stride=2, padding=1, output_padding=1),
        )

        self.feature_adaptor_I_1 = nn.Conv2d(3, channel_N, 3, stride=1, padding=1)
        self.feature_adaptor_I_2 = nn.Conv2d(3, channel_N, 3, stride=1, padding=1)
        self.feature_adaptor_P = nn.Conv2d(channel_N, channel_N, 1)
        self.feature_extractor = FeatureExtractor()
        self.context_fusion_net = MultiScaleContextFusion()

        self.res_encoder = ResEncoder()

        self.res_prior_encoder = nn.Sequential(
            nn.Conv2d(channel_M, channel_N, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel_N, channel_N, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel_N, channel_N, 3, stride=2, padding=1),
        )

        self.res_prior_decoder = nn.Sequential(
            conv3x3(channel_N, channel_M),
            nn.LeakyReLU(),
            subpel_conv3x3(channel_M, channel_M, 2),
            nn.LeakyReLU(),
            conv3x3(channel_M, channel_M * 3 // 2),
            nn.LeakyReLU(),
            subpel_conv3x3(channel_M * 3 // 2, channel_M * 3 // 2, 2),
            nn.LeakyReLU(),
            conv3x3(channel_M * 3 // 2, channel_M * 2),
        )

        self.temporal_prior_encoder = TemporalPriorEncoder()

        self.res_entropy_parameter = nn.Sequential(
            nn.Conv2d(channel_M * 12 // 3, channel_M * 10 // 3, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel_M * 10 // 3, channel_M * 8 // 3, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel_M * 8 // 3, channel_M * 6 // 3, 3, stride=1, padding=1),
        )

        self.res_decoder = ResDecoder()

        self.recon_generation_net = ReconGeneration()

        self.entropy_coder = None
        self.bit_estimator_z = BitEstimator(channel_N)
        self.bit_estimator_z_mv = BitEstimator(channel_N)
        self.gaussian_encoder = GaussianEncoder()
        self.ms_ssim_loss = MS_SSIM(data_range=1.0, win_size=win_size)

    def load_dict(self, pretrained_dict, strict=True):
        result_dict = {}
        for key, weight in pretrained_dict.items():
            result_key = key
            if key[:7] == "module.":
                result_key = key[7:]
            result_dict[result_key] = weight

        self.load_state_dict(result_dict, strict=strict)

    def multi_scale_feature_extractor(self, ref, feature):
        if feature is None:
            feature = self.feature_adaptor_I_2(ref)
        else:
            feature = self.feature_adaptor_P(feature)
        return self.feature_extractor(feature)

    def motion_compensation(self, ref, feature, mv):
        warpframe = flow_warp(ref, mv)
        mv2 = bilineardownsacling(mv) / 2
        mv3 = bilineardownsacling(mv2) / 2
        ref_feature1, ref_feature2, ref_feature3 = self.multi_scale_feature_extractor(ref, feature)
        context1 = flow_warp(ref_feature1, mv)
        context2 = flow_warp(ref_feature2, mv2)
        context3 = flow_warp(ref_feature3, mv3)
        context1, context2, context3 = self.context_fusion_net(context1, context2, context3)
        return context1, context2, context3, warpframe

    @staticmethod
    def get_y_bits_probs(y, sigma):
        mu = torch.zeros_like(sigma)
        sigma = sigma.clamp(1e-5, 1e10)
        gaussian = torch.distributions.laplace.Laplace(mu, sigma)
        probs = gaussian.cdf(y + 0.5) - gaussian.cdf(y - 0.5)
        total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-5) / math.log(2.0), 0, 50))
        return total_bits, probs

    @staticmethod
    def get_z_bits_probs(z, bit_estimator):
        prob = bit_estimator(z + 0.5) - bit_estimator(z - 0.5)
        total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))
        return total_bits, prob

    def update(self, force=False):
        self.entropy_coder = EntropyCoder()
        self.bit_estimator_z_mv.update(force=force, entropy_coder=self.entropy_coder)
        self.bit_estimator_z.update(force=force, entropy_coder=self.entropy_coder)
        self.gaussian_encoder.update(force=force, entropy_coder=self.entropy_coder)

    def encode_decode(self, x, ref_frame, ref_feature, output_path=None,
                      pic_width=None, pic_height=None):
        # pic_width and pic_height may be different from x's size. x here is after padding
        # x_hat has the same size with x
        encoded = self.forward_one_frame(x, ref_frame, ref_feature)
        result = {
            "x_hat": encoded['recon_image'],
            "feature": encoded['feature'],
            "bit_y": encoded['bit_y'].item(),
            "bit_z": encoded['bit_z'].item(),
            "bit_mv_y": encoded['bit_mv_y'].item(),
            "bit_mv_z": encoded['bit_mv_z'].item(),
            "bit": encoded['bit'].item(),
            "decoding_time": 0
        }
        return result

    def quant(self, x):
        return torch.round(x)

    def forward_one_frame(self, x, ref_frame, ref_feature):
        if ref_feature is None:
            ref_feature = self.feature_adaptor_I_1(ref_frame)

        mv_y = self.mv_encoder(torch.cat((x, ref_feature), dim=1))
        mv_z = self.mv_prior_encoder(mv_y)
        mv_z_hat = self.quant(mv_z)
        mv_params = self.mv_prior_decoder(mv_z_hat)
        mv_scales_hat, mv_means_hat = mv_params.chunk(2, 1)

        mv_y_res = mv_y - mv_means_hat
        mv_y_q = self.quant(mv_y_res)
        mv_y_hat = mv_y_q + mv_means_hat

        mv_hat = self.mv_decoder(mv_y_hat)
        context1, context2, context3, warp_frame = self.motion_compensation(
            ref_frame, ref_feature, mv_hat)

        y = self.res_encoder(x, context1, context2, context3)
        z = self.res_prior_encoder(y)
        z_hat = self.quant(z)
        hierarchical_params = self.res_prior_decoder(z_hat)
        temporal_params = self.temporal_prior_encoder(context1, context2, context3)

        params = torch.cat((temporal_params, hierarchical_params), dim=1)
        gaussian_params = self.res_entropy_parameter(params)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_res = y - means_hat
        y_q = self.quant(y_res)
        y_hat = y_q + means_hat

        feature = self.res_decoder(y_hat, context1, context2, context3)
        recon_image = self.recon_generation_net(feature)

        mse_loss = torch.mean((recon_image - x).pow(2))
        me_mse = torch.mean((warp_frame - x).pow(2))
        recon_ms_ssim = self.ms_ssim_loss(recon_image, x)
        me_ms_ssim = self.ms_ssim_loss(warp_frame, x)

        y_for_bit = y_q
        mv_y_for_bit = mv_y_q
        z_for_bit = z_hat
        mv_z_for_bit = mv_z_hat
        total_bits_y, _ = self.get_y_bits_probs(y_for_bit, scales_hat)
        total_bits_mv_y, _ = self.get_y_bits_probs(mv_y_for_bit, mv_scales_hat)
        total_bits_z, _ = self.get_z_bits_probs(z_for_bit, self.bit_estimator_z)
        total_bits_mv_z, _ = self.get_z_bits_probs(mv_z_for_bit, self.bit_estimator_z_mv)

        im_shape = x.size()
        pixel_num = im_shape[0] * im_shape[2] * im_shape[3]
        bpp_y = total_bits_y / pixel_num
        bpp_z = total_bits_z / pixel_num
        bpp_mv_y = total_bits_mv_y / pixel_num
        bpp_mv_z = total_bits_mv_z / pixel_num

        bpp = bpp_y + bpp_z + bpp_mv_y + bpp_mv_z

        return {"bpp_mv_y": bpp_mv_y,
                "bpp_mv_z": bpp_mv_z,
                "bpp_y": bpp_y,
                "bpp_z": bpp_z,
                "bpp": bpp,
                "me_mse": me_mse,
                "mse_loss": mse_loss,
                "me_ms_ssim": me_ms_ssim,
                "recon_ms_ssim": recon_ms_ssim,
                "recon_image": recon_image,
                "feature": feature,
                "warp_frame": warp_frame,
                "temporal_params": temporal_params,
                "y": y,
                "z": z,
                "mv_y": mv_y,
                "mv_z": mv_z,
                "bit_y": total_bits_y,
                "bit_z": total_bits_z,
                "bit_mv_y": total_bits_mv_y,
                "bit_mv_z": total_bits_mv_z,
                "bit": total_bits_y + total_bits_z + total_bits_mv_y + total_bits_mv_z,
                "mv": mv_hat
                }
