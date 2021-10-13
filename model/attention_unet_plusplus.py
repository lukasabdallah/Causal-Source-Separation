from model.grid_attention import GridAttentionBlock2D
from utils import *


class DownSamplingBlock(nn.Module):
    def __init__(self, in_channel, out_channel, dropout=0.0):
        super(DownSamplingBlock, self).__init__()
        layers = [nn.LeakyReLU(0.2),
                  nn.utils.spectral_norm(
                      nn.Conv2d(in_channel, out_channel, (4, 4), (2, 2), (1, 1), bias=False)),
                  nn.InstanceNorm2d(out_channel, affine=True)
                  ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UpSamplingBlock(nn.Module):
    def __init__(self, in_channel, out_channel, dropout=0.0):
        super(UpSamplingBlock, self).__init__()
        layers = [
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.ConvTranspose2d(
                in_channel, out_channel, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(out_channel, affine=True)
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class StandardUnit(nn.Module):
    def __init__(self, in_channel, out_channel, dropout=0.0):
        super(StandardUnit, self).__init__()

        layers = [nn.LeakyReLU(0.2),
                  nn.ConstantPad2d((1, 2, 1, 2), 0),
                  nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, (4, 4), (1, 1)))]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.ConstantPad2d((1, 2, 1, 2), 0))
        layers.append(nn.utils.spectral_norm(
            nn.Conv2d(out_channel, out_channel, (4, 4), (1, 1), bias=False)))
        layers.append(nn.InstanceNorm2d(out_channel, affine=True))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class FinalUnetActivate(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FinalUnetActivate, self).__init__()
        layers = [
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.ConvTranspose2d(
                in_channel, out_channel, (4, 4), (2, 2), (1, 1))),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class AttUNetPlusPlus(nn.Module):
    def __init__(self, in_channel, ngf):
        super(AttUNetPlusPlus, self).__init__()
        self.conv1_1 = nn.utils.spectral_norm(
            nn.Conv2d(in_channel, ngf, (4, 4), (2, 2), (1, 1)))
        self.conv2_1 = DownSamplingBlock(ngf, ngf*2)
        self.conv3_1 = DownSamplingBlock(ngf*2, ngf*4)
        self.conv4_1 = DownSamplingBlock(ngf*4, ngf*8)
        self.conv5_1 = DownSamplingBlock(ngf*8, ngf*8)
        self.conv6_1 = DownSamplingBlock(ngf * 8, ngf * 8)

        self.up1_2 = UpSamplingBlock(ngf*2, ngf)
        self.ga1_2 = GridAttentionBlock2D(ngf, ngf*2)
        self.conv1_2 = StandardUnit(ngf*2, ngf)

        self.up2_2 = UpSamplingBlock(ngf*4, ngf*2)
        self.ga2_2 = GridAttentionBlock2D(ngf*2, ngf*4)
        self.down2_2 = DownSamplingBlock(ngf, ngf*2)
        self.conv2_2 = StandardUnit(ngf*6, ngf*2)
        self.up1_3 = UpSamplingBlock(ngf*2, ngf)
        self.ga1_3 = GridAttentionBlock2D(ngf, ngf*2)
        self.conv1_3 = StandardUnit(ngf*3, ngf)

        self.up3_2 = UpSamplingBlock(ngf*8, ngf*4)
        self.ga3_2 = GridAttentionBlock2D(ngf*4, ngf*8)
        self.down3_2 = DownSamplingBlock(ngf*2, ngf*4)
        self.conv3_2 = StandardUnit(ngf*12, ngf*4)
        self.up2_3 = UpSamplingBlock(ngf*4, ngf*2)
        self.ga2_3 = GridAttentionBlock2D(ngf*2, ngf*4)
        self.down2_3 = DownSamplingBlock(ngf, ngf*2)
        self.conv2_3 = StandardUnit(ngf*8, ngf*2)
        self.up1_4 = UpSamplingBlock(ngf*2, ngf)
        self.ga1_4 = GridAttentionBlock2D(ngf, ngf*2)
        self.conv1_4 = StandardUnit(ngf*4, ngf)

        self.up4_2 = UpSamplingBlock(ngf*8, ngf*8, 0.3)
        self.ga4_2 = GridAttentionBlock2D(ngf*8, ngf*8)
        self.down4_2 = DownSamplingBlock(ngf*4, ngf*8, 0.3)
        self.conv4_2 = StandardUnit(ngf*24, ngf*8, 0.3)
        self.up3_3 = UpSamplingBlock(ngf*8, ngf*4)
        self.ga3_3 = GridAttentionBlock2D(ngf*4, ngf*8)
        self.down3_3 = DownSamplingBlock(ngf*2, ngf*4)
        self.conv3_3 = StandardUnit(ngf*16, ngf*4)
        self.up2_4 = UpSamplingBlock(ngf*4, ngf*2)
        self.ga2_4 = GridAttentionBlock2D(ngf*2, ngf*4)
        self.down2_4 = DownSamplingBlock(ngf, ngf*2)
        self.conv2_4 = StandardUnit(ngf*10, ngf*2)
        self.up1_5 = UpSamplingBlock(ngf*2, ngf)
        self.ga1_5 = GridAttentionBlock2D(ngf, ngf*2)
        self.conv1_5 = StandardUnit(ngf*5, ngf)

        self.up5_2 = UpSamplingBlock(ngf*8, ngf*8, 0.5)
        self.ga5_2 = GridAttentionBlock2D(ngf*8, ngf*8)
        self.down5_2 = DownSamplingBlock(ngf*8, ngf*8, 0.3)
        self.conv5_2 = StandardUnit(ngf*24, ngf*8, 0.5)
        self.up4_3 = UpSamplingBlock(ngf*8, ngf*8, 0.3)
        self.ga4_3 = GridAttentionBlock2D(ngf*8, ngf*8)
        self.down4_3 = DownSamplingBlock(ngf*4, ngf*8, 0.3)
        self.conv4_3 = StandardUnit(ngf*32, ngf*8)
        self.up3_4 = UpSamplingBlock(ngf*8, ngf*4)
        self.ga3_4 = GridAttentionBlock2D(ngf*4, ngf*8)
        self.down3_4 = DownSamplingBlock(ngf*2, ngf*4)
        self.conv3_4 = StandardUnit(ngf*20, ngf*4)
        self.up2_5 = UpSamplingBlock(ngf*4, ngf*2)
        self.ga2_5 = GridAttentionBlock2D(ngf*2, ngf*4)
        self.down2_5 = DownSamplingBlock(ngf, ngf*2)
        self.conv2_5 = StandardUnit(ngf*12, ngf*2)
        self.up1_6 = UpSamplingBlock(ngf*2, ngf)
        self.ga1_6 = GridAttentionBlock2D(ngf, ngf*2)
        self.conv1_6 = StandardUnit(ngf*6, ngf)

        self.FUA_mag = FinalUnetActivate(ngf, 1)
        self.FUA_phase = FinalUnetActivate(ngf, 2)

    def forward(self, complex_stft, mag_stft):                       # 256 x 256
        # x = torch.cat([complex_stft, mag_stft], 1)
        x = mag_stft
        conv1_1 = self.conv1_1(x)               # 128 x 128
        conv2_1 = self.conv2_1(conv1_1)         # 64 x 64
        conv3_1 = self.conv3_1(conv2_1)         # 32 x 32
        conv4_1 = self.conv4_1(conv3_1)         # 16 x 16
        conv5_1 = self.conv5_1(conv4_1)         # 8 x 8
        conv6_1 = self.conv6_1(conv5_1)         # 4 x 4

        conv1_1, mask1_1 = self.ga1_2(conv1_1, conv2_1)
        up1_2 = self.up1_2(conv2_1)
        conv1_2 = torch.cat((up1_2, conv1_1), dim=1)
        conv1_2 = self.conv1_2(conv1_2)

        conv2_1, mask2_1 = self.ga2_2(conv2_1, conv3_1)
        up2_2 = self.up2_2(conv3_1)
        down2_2 = self.down2_2(conv1_2)
        conv2_2 = torch.cat((up2_2, conv2_1, down2_2), dim=1)
        conv2_2 = self.conv2_2(conv2_2)
        conv1_2, mask1_2 = self.ga1_3(conv1_2, conv2_2)
        up1_3 = self.up1_3(conv2_2)
        conv1_3 = torch.cat((up1_3, conv1_1, conv1_2), dim=1)
        conv1_3 = self.conv1_3(conv1_3)

        conv3_1, mask3_1 = self.ga3_2(conv3_1, conv4_1)
        up3_2 = self.up3_2(conv4_1)
        down3_2 = self.down3_2(conv2_2)
        conv3_2 = torch.cat((up3_2, conv3_1, down3_2), dim=1)
        conv3_2 = self.conv3_2(conv3_2)
        conv2_2, mask2_2 = self.ga2_3(conv2_2, conv3_2)
        up2_3 = self.up2_3(conv3_2)
        down2_3 = self.down2_3(conv1_3)
        conv2_3 = torch.cat((up2_3, conv2_1, conv2_2, down2_3), dim=1)
        conv2_3 = self.conv2_3(conv2_3)
        conv1_3, mask1_3 = self.ga1_4(conv1_3, conv2_3)
        up1_4 = self.up1_4(conv2_3)
        conv1_4 = torch.cat((up1_4, conv1_1, conv1_2, conv1_3), dim=1)
        conv1_4 = self.conv1_4(conv1_4)

        conv4_1, mask4_1 = self.ga4_2(conv4_1, conv5_1)
        up4_2 = self.up4_2(conv5_1)
        down4_2 = self.down4_2(conv3_2)
        conv4_2 = torch.cat((up4_2, conv4_1, down4_2), dim=1)
        conv4_2 = self.conv4_2(conv4_2)
        conv3_2, mask3_2 = self.ga3_3(conv3_2, conv4_2)
        up3_3 = self.up3_3(conv4_2)
        down3_3 = self.down3_3(conv2_3)
        conv3_3 = torch.cat((up3_3, conv3_1, conv3_2, down3_3), dim=1)
        conv3_3 = self.conv3_3(conv3_3)
        conv2_3, mask2_3 = self.ga2_4(conv2_3, conv3_3)
        up2_4 = self.up2_4(conv3_3)
        down2_4 = self.down2_4(conv1_4)
        conv2_4 = torch.cat((up2_4, conv2_1, conv2_2, conv2_3, down2_4), dim=1)
        conv2_4 = self.conv2_4(conv2_4)
        conv1_4, mask1_4 = self.ga1_5(conv1_4, conv2_4)
        up1_5 = self.up1_5(conv2_4)
        conv1_5 = torch.cat((up1_5, conv1_1, conv1_2, conv1_3, conv1_4), dim=1)
        conv1_5 = self.conv1_5(conv1_5)

        conv5_1, mask5_1 = self.ga5_2(conv5_1, conv6_1)
        up5_2 = self.up5_2(conv6_1)
        down5_2 = self.down5_2(conv4_2)
        conv5_2 = torch.cat((up5_2, conv5_1, down5_2), dim=1)
        conv5_2 = self.conv5_2(conv5_2)
        conv4_2, mask4_2 = self.ga4_3(conv4_2, conv5_2)
        up4_3 = self.up4_3(conv5_2)
        down4_3 = self.down4_3(conv3_3)
        conv4_3 = torch.cat((up4_3, conv4_1, conv4_2, down4_3), dim=1)
        conv4_3 = self.conv4_3(conv4_3)
        conv3_3, mask3_3 = self.ga3_4(conv3_3, conv4_3)
        up3_4 = self.up3_4(conv4_3)
        down3_4 = self.down3_4(conv2_4)
        conv3_4 = torch.cat((up3_4, conv3_1, conv3_2, conv3_3, down3_4), dim=1)
        conv3_4 = self.conv3_4(conv3_4)
        conv2_4, mask2_4 = self.ga2_5(conv2_4, conv3_4)
        up2_5 = self.up2_5(conv3_4)
        down2_5 = self.down2_5(conv1_5)
        conv2_5 = torch.cat(
            (up2_5, conv2_1, conv2_2, conv2_3, conv2_4, down2_5), dim=1)
        conv2_5 = self.conv2_5(conv2_5)
        conv1_5, mask1_5 = self.ga1_6(conv1_5, conv2_5)
        up1_6 = self.up1_6(conv2_5)
        conv1_6 = torch.cat(
            (up1_6, conv1_1, conv1_2, conv1_3, conv1_4, conv1_5), dim=1)
        conv1_6 = self.conv1_6(conv1_6)

        out_mag = self.FUA_mag(conv1_6)
        # out_complex = self.FUA_phase(conv1_5)

        # mask_list = [mask1_1, mask1_2, mask1_3, mask1_4, mask1_5, mask2_1, mask2_2, mask2_3, mask2_4, mask3_1,
        #              mask3_2, mask3_3, mask4_1, mask4_2, mask5_1]

        # return out_mag, out_complex, mask_list
        return out_mag
