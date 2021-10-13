import torch
from torch import nn
from torch.nn import functional as F


class _GridAttentionBlockND(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels=None, dimension=2, mode='concatenation',
                 sub_sample_factor=(2,2)):
        super(_GridAttentionBlockND, self).__init__()

        assert mode in ['concatenation', 'concatenation_debug', 'concatenation_residual']

        # Downsampling rate for the input featuremap
        if isinstance(sub_sample_factor, tuple): self.sub_sample_factor = sub_sample_factor
        elif isinstance(sub_sample_factor, list): self.sub_sample_factor = tuple(sub_sample_factor)
        else: self.sub_sample_factor = tuple([sub_sample_factor]) * dimension

        # Default parameter set
        self.mode = mode
        self.dimension = dimension
        self.sub_sample_kernel_size = self.sub_sample_factor

        # Number of channels (pixel dimensions)
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        else:
            raise NotImplemented
        conv_nd = nn.Conv2d
        #bn = nn.BatchNorm2d
        InstanceNorm = nn.InstanceNorm2d
        self.upsample_mode = 'bilinear'

        # Output transform
        self.W = nn.Sequential(
            nn.utils.spectral_norm(conv_nd(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1,
                                           stride=1, padding=0, bias=False)),
            InstanceNorm(self.in_channels, affine=True),
        )

        # Theta^T * x_ij + Phi^T * gating_signal + bias
        self.theta = nn.Sequential(nn.utils.spectral_norm(conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                                    kernel_size=1, stride=1, padding=0, bias=False)),
                                   InstanceNorm(self.inter_channels, affine=True)
                                   )
        self.phi = nn.Sequential(nn.utils.spectral_norm(conv_nd(in_channels=self.gating_channels, out_channels=self.inter_channels,
                                    kernel_size=1, stride=1, padding=0, bias=False)),
                                 InstanceNorm(self.inter_channels, affine=True)
                                 )
        self.psi = nn.Sequential(nn.utils.spectral_norm(conv_nd(in_channels=self.inter_channels, out_channels=1,
                                    kernel_size=1, stride=1, padding=0, bias=False)),
                                 InstanceNorm(1, affine=True)
                                 )

        # self.theta_channel = nn.utils.spectral_norm(conv_nd(in_channels=self.in_channels, out_channels=self.in_channels,
        #                      kernel_size=self.sub_sample_kernel_size, stride=self.sub_sample_factor, padding=0, bias=False))
        # self.phi_channel = nn.utils.spectral_norm(conv_nd(in_channels=self.gating_channels, out_channels=self.in_channels,
        #                                           kernel_size=1, stride=1, padding=0, bias=True))
        # self.linear_1 = nn.utils.spectral_norm(nn.Linear(self.in_channels, self.in_channels//8))
        # self.linear_2 = nn.utils.spectral_norm(nn.Linear(self.in_channels//8, self.in_channels))


        # Define the operation
        if mode == 'concatenation':
            self.operation_function = self._concatenation
        elif mode == 'concatenation_debug':
            self.operation_function = self._concatenation_debug
        elif mode == 'concatenation_residual':
            self.operation_function = self._concatenation_residual
        else:
            raise NotImplementedError('Unknown operation function.')

    def forward(self, x, g):
        '''
        :param x: (b, c, t, h, w)
        :param g: (b, g_d)
        :return:
        '''

        output = self.operation_function(x, g)
        return output

    def _concatenation(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        g = F.upsample(g, size=theta_x_size[2:], mode=self.upsample_mode)
        phi_g = self.phi(g)
        # mu = torch.cat([theta_x, phi_g], 1)
        f = F.leaky_relu(theta_x + phi_g, 0.2, inplace=True)
        # f = F.leaky_relu(mu, 0.2, inplace=True)
        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = F.sigmoid(self.psi(f))

        # upsample the attentions and multiply
        # sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
        y = sigm_psi_f.expand_as(x) * x
        # y = channel_attention * y
        W_y = self.W(y)
        # W_y = W_y + x

        return W_y, sigm_psi_f

    def _concatenation_debug(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
        f = F.softplus(theta_x + phi_g)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = F.sigmoid(self.psi(f))

        # upsample the attentions and multiply
        sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f


    def _concatenation_residual(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
        f = F.relu(theta_x + phi_g, inplace=True)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        f = self.psi(f).view(batch_size, 1, -1)
        sigm_psi_f = F.softmax(f, dim=2).view(batch_size, 1, *theta_x.size()[2:])

        # upsample the attentions and multiply
        sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f


class GridAttentionBlock2D(_GridAttentionBlockND):
    def __init__(self, in_channels, gating_channels, inter_channels=None, mode='concatenation',
                 sub_sample_factor=(2,2)):
        super(GridAttentionBlock2D, self).__init__(in_channels,
                                                   inter_channels=inter_channels,
                                                   gating_channels=gating_channels,
                                                   dimension=2, mode=mode,
                                                   sub_sample_factor=sub_sample_factor,
                                                   )