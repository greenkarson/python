from day03.utils import *
from torchviz import make_dot, make_dot_from_trace

class MBConv(nn.Module):

    def __init__(self,image_size, in_channels, out_channels, expand, kernel_size, stride, drop_ratio):
        super().__init__()

        mid_channels = in_channels * expand
        if expand == 1:
            self.expand_conv = nn.Identity()
        else:
            self.expand_conv = Conv2dSamePaddingBNSwish(image_size, in_channels, mid_channels, 1, bias=False)
        self.depthwise_conv = Conv2dSamePaddingBNSwish(image_size, mid_channels, mid_channels, kernel_size,
                                                        stride=stride, group=mid_channels, bias=False)

        self.se = SEModule(mid_channels, int(in_channels * 0.25))
        self.project_conv = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels, 1e-3, 0.01)
        )
        self.skip = stride == 1 and in_channels == out_channels
        self.dropout = nn.Dropout2d(drop_ratio)

    def forward(self, x):
        h = self.expand_conv(x)
        h = self.depthwise_conv(h)
        h = self.se(h)
        h = self.project_conv(h)
        if self.skip:
            h = self.dropout(h)
            h = h + x
        return h


class MBBlock(nn.Module):

    def __init__(self, image_size, in_channels, out_channels, expand, kernel_size, stride, num_repeat, drop_ratio):
        super().__init__()
        layer = [MBConv(image_size, in_channels, out_channels, expand, kernel_size, stride, drop_ratio)]
        for i in range(1, num_repeat):
            layer.append(MBConv(image_size, out_channels, out_channels, expand, kernel_size, 1, drop_ratio))
        self.layer = nn.Sequential(*layer)

    def forward(self, x):
        return self.layer(x)


class EfficientNet(nn.Module):

    def __init__(self, image_size, width_coeff, depth_coeff, drop_ratio=0.2):
        super().__init__()

        im_s = (image_size, image_size)

        self.input_layer = Conv2dSamePaddingBNSwish(im_s, 3, re_ch(32, width_coeff), 3, stride=2, bias=False)

        self.block = nn.Sequential(
            MBBlock(im_s, re_ch(32, width_coeff), re_ch(16, width_coeff), 1, 3, 1, re_dp(1, depth_coeff), drop_ratio),
            MBBlock(im_s, re_ch(16, width_coeff), re_ch(24, width_coeff), 6, 3, 2, re_dp(2, depth_coeff), drop_ratio),
            MBBlock(im_s, re_ch(24, width_coeff), re_ch(40, width_coeff), 6, 5, 2, re_dp(2, depth_coeff), drop_ratio),
            MBBlock(im_s, re_ch(40, width_coeff), re_ch(80, width_coeff), 6, 3, 2, re_dp(3, depth_coeff), drop_ratio),
            MBBlock(im_s, re_ch(80, width_coeff), re_ch(112, width_coeff), 6, 5, 1, re_dp(3, depth_coeff), drop_ratio),
            MBBlock(im_s, re_ch(112, width_coeff), re_ch(192, width_coeff), 6, 5, 2, re_dp(4, depth_coeff), drop_ratio),
            MBBlock(im_s, re_ch(192, width_coeff), re_ch(320, width_coeff), 6, 3, 1, re_dp(1, depth_coeff), drop_ratio),
        )

        self.feature_layer = Conv2dSamePaddingBNSwish(im_s, re_ch(320, width_coeff),
                                                      re_ch(1280, width_coeff), 1, bias=False)


    def forward(self, x):
        h = self.input_layer(x)
        h = self.block(h)
        h = self.feature_layer(h)
        return h


if __name__ == '__main__':
    net = EfficientNet(224, 1.6, 2.2, 0.4)
    x = torch.rand(1, 3, 224, 224)
    y = net(x)
    g = make_dot(net(x), params=dict(net.named_parameters()))
    print(y.shape)
    print(net)
    g.view()
