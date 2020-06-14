from torch import nn
import torch


class Converlutional(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, padding):
        super(Converlutional, self).__init__()
        self.converlutional_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.converlutional_layer(x)


class Residual(nn.Module):
    def __init__(self, in_channels):
        super(Residual, self).__init__()
        self.residual_layer = nn.Sequential(
            Converlutional(in_channels, in_channels // 2, kernel=1, stride=1, padding=0),
            Converlutional(in_channels // 2, in_channels, kernel=3, stride=1, padding=1)
        )

    def forward(self, x):
        return x + self.residual_layer(x)


class Upsample(nn.Module):
    def __init__(self):
        super(Upsample, self).__init__()

    def forward(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample, self).__init__()
        self.downsample_layer = nn.Sequential(
            Converlutional(in_channels, out_channels, kernel=3, stride=2, padding=1)
        )

    def forward(self, x):
        return self.downsample_layer(x)


class ConverlutionalSet(nn.Module):
    def __init__(self, in_channels, out_channnels):
        super(ConverlutionalSet, self).__init__()
        self.converlutonal_set_layer = nn.Sequential(
            Converlutional(in_channels, out_channnels, kernel=1, stride=1, padding=0),
            Converlutional(out_channnels, in_channels, kernel=3, stride=1, padding=1),

            Converlutional(in_channels, out_channnels, kernel=1, stride=1, padding=0),
            Converlutional(out_channnels, in_channels, kernel=3, stride=1, padding=1),

            Converlutional(in_channels, out_channnels, kernel=1, stride=1, padding=0)
        )

    def forward(self, x):
        return self.converlutonal_set_layer(x)


class Darknet53(nn.Module):
    def __init__(self):
        super(Darknet53, self).__init__()
        self.predict_52 = nn.Sequential(
            Converlutional(3, 32, kernel=3, stride=1, padding=1),
            Downsample(32, 64),

            Residual(64),
            Downsample(64, 128),

            Residual(128),
            Residual(128),
            Downsample(128,256),

            Residual(256),
            Residual(256),
            Residual(256),
            Residual(256),
            Residual(256),
            Residual(256),
            Residual(256),
            Residual(256),
        )

        self.predict_26 = nn.Sequential(
            Downsample(256, 512),

            Residual(512),
            Residual(512),
            Residual(512),
            Residual(512),
            Residual(512),
            Residual(512),
            Residual(512),
            Residual(512),
        )

        self.predict_13 = nn.Sequential(
            Downsample(512, 1024),

            Residual(1024),
            Residual(1024),
            Residual(1024),
            Residual(1024)
        )

        self.convolutionalset_13 = nn.Sequential(
            ConverlutionalSet(1024, 512),
        )

        self.detection_13 = nn.Sequential(
            Converlutional(512, 1024, kernel=3, stride=1, padding=1),
            nn.Conv2d(1024, 45, kernel_size=1, stride=1, padding=0)
        )

        self.up_26 = nn.Sequential(
            Converlutional(512, 256, kernel=1, stride=1, padding=0),
            Upsample()
        )

        self.convolutionalset_26 = nn.Sequential(
            ConverlutionalSet(768,256)
        )

        self.detection_26 = nn.Sequential(
            Converlutional(256, 512, kernel=3, stride=1, padding=1),
            nn.Conv2d(512, 45, kernel_size=1, stride=1, padding=0)
        )

        self.up_52 = nn.Sequential(
            Converlutional(256, 128, kernel=1, stride=1, padding=0),
            Upsample()
        )

        self.convolutionalset_52 = nn.Sequential(
            ConverlutionalSet(384, 128)
        )

        self.detection_52 = nn.Sequential(
            Converlutional(128, 256, kernel=3, stride=1, padding=1),
            nn.Conv2d(256, 45, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        predict_52 = self.predict_52(x)
        predict_26 = self.predict_26(predict_52)
        predict_13 = self.predict_13(predict_26)

        convolutionalset_13 = self.convolutionalset_13(predict_13)
        # print(convolutionalset_13.shape)

        detection_13 = self.detection_13(convolutionalset_13)
        # print(detection_13.shape)

        up_26_out = self.up_26(convolutionalset_13)
        # print(up_26_out.shape)

        route_26_out = torch.cat((up_26_out,predict_26), dim=1)
        # print(route_26_out.shape)

        convolutionalset_26 = self.convolutionalset_26(route_26_out)
        # print(convolutionalset_26.shape)

        detection_26 = self.detection_26(convolutionalset_26)
        # print(detection_26.shape)

        up_52_out = self.up_52(convolutionalset_26)
        # print(up_52_out.shape)
        route_52_out = torch.cat((up_52_out, predict_52), dim=1)
        # print(route_52_out.shape)

        convolutionalset_52 = self.convolutionalset_52(route_52_out)
        # print(convolutionalset_52.shape)
        detection_52 = self.detection_52(convolutionalset_52)
        # print(detection_52.shape)

        return detection_13, detection_26, detection_52


if __name__ == '__main__':
    x = torch.randn(1, 3, 416, 416)
    net = Darknet53()
    y_13, y_26, y_52 = net(x)
    print(y_13.shape)
    print(y_26.shape)
    print(y_52.shape)