import torch
import torch.nn.functional as F
import torchvision.models as models


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(self.out_channels),
            torch.nn.ReLU(True),
        )
    
    def forward(self,x):
        return self.block(x)


class Upsampling(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        conv_blocks = []
        for i in range(len(in_channels) - 1):
            conv_blocks.append(ConvBlock(self.in_channels[i], self.in_channels[i+1]))
        conv_blocks.append(ConvBlock(self.in_channels[-1], self.out_channels))
        self.conv_blocks = torch.nn.ModuleList(conv_blocks)
    
    def forward(self, x1, x2):
        x1_up = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=False)
        output = torch.cat([x2, x1_up], dim=1)
        for conv in self.conv_blocks:
            output = conv(output)
        return output


class UNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        vgg.avgpool = torch.nn.Identity()
        vgg.features[30] = torch.nn.Identity()
        for param in vgg.features:
            param.requires_grad = False
        self.vgg = vgg

        self.up_1 = Upsampling([1024,   256, 128], 17)
        self.up_2 = Upsampling([256+17, 128,  64], 17)
        self.up_3 = Upsampling([128+17, 64], 17)
        self.up_4 = Upsampling([64+17,  32], 17)
        
        self.out_layer6 = torch.nn.Sequential(
            torch.nn.Conv2d(17, 17, 1),
            torch.nn.Sigmoid(),
        )
        self.out_layer7 = torch.nn.Sequential(
            torch.nn.Conv2d(17, 17, 1),
            torch.nn.Sigmoid(),
        )
        self.out_layer8 = torch.nn.Sequential(
            torch.nn.Conv2d(17, 17, 1),
            torch.nn.Sigmoid(),
        )
        self.out_layer9 = torch.nn.Sequential(
            torch.nn.Conv2d(17, 17, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self,x):
        x1 = self.vgg.features[0:4](x)
        x2 = self.vgg.features[4:9](x1)
        x3 = self.vgg.features[9:16](x2)
        x4 = self.vgg.features[16:23](x3)
        x5 = self.vgg.features[23:30](x4)
        x6 = self.up_1(x5, x4)
        x7 = self.up_2(x6, x3)
        x8 = self.up_3(x7, x2)
        x9 = self.up_4(x8, x1)

        outx6 = self.out_layer6(x6)
        outx7 = self.out_layer7(x7)
        outx8 = self.out_layer8(x8)
        outx9 = self.out_layer9(x9)

        return (
            (outx6[:,:16], outx6[:,16:]),
            (outx7[:,:16], outx7[:,16:]),
            (outx8[:,:16], outx8[:,16:]),
            (outx9[:,:16], outx9[:,16:]),
        )


def main():
    net = UNet(3,17)
    x = torch.rand(1, 3, 800, 600)
    print("x.shape: ",x.shape)
    (F3_estimate,S3_estimate),(F2_estimate,S2_estimate),(F1_estimate,S1_estimate),(F0_estimate,S0_estimate) = net.forward(x)
    print("F3.shape: ",F3_estimate.shape)
    print("S3.shape: ",S3_estimate.shape)
    print("F2.shape: ",F2_estimate.shape)
    print("S2.shape: ",S2_estimate.shape)
    print("F1.shape: ",F1_estimate.shape)
    print("S1.shape: ",S1_estimate.shape)
    print("F0.shape: ",F0_estimate.shape)
    print("S0.shape: ",S0_estimate.shape)


if __name__ == "__main__":
    main()