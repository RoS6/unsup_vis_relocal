import torch

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=0) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = torch.nn.Sequential(torch.nn.Conv2d(self.in_channels, self.out_channels, kernel_size= kernel_size, padding= 0,stride=1),
                                     torch.nn.BatchNorm2d(self.out_channels),
                                     torch.nn.ReLU(True))
    def forward(self,x):
        return self.block(x)


class Upsampling(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=0,conv_layer = 3):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.blockseq = torch.nn.ModuleList()
        # self.blockseq = torch.nn.ModuleList([ConvBlock(self.in_channels[0], self.out_channels, kernel_size=kernel_size, padding=0)])
        for i in range(conv_layer-1):
            self.blockseq.append(ConvBlock(self.in_channels[i], self.in_channels[i+1], kernel_size=kernel_size, padding=0))
        self.blockseq.append(ConvBlock(self.in_channels[-1], self.out_channels, kernel_size=kernel_size, padding=0))
        # initialize sequentials / 
    def forward(self, x1,x2):
        x1up = self.upsample(x1)
        if x1up.shape != x2.shape:
            x1up = torch.nn.functional.pad(x1up, (0, x2.shape[3] - x1up.shape[3], 0, x2.shape[2] - x1up.shape[2]))

        x = torch.cat([x2, x1up], dim=1)
        output = x
        counter = 1
        for conv in self.blockseq:
            output = conv(output)
            counter +=1 
        
        return output

class UNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        import torchvision.models as models
        vgg = models.vgg16(pretrained=True)
        vgg.avgpool = torch.nn.Identity()
        vgg.features[30] = torch.nn.Identity()
        for param in vgg.features:
            param.requires_grad = False
        self.vgg = vgg

        self.up_1 = Upsampling([1024,256,128], 17,conv_layer = 3)
        self.up_2 = Upsampling([256+17,128,64], 17,conv_layer = 3)
        self.up_3 = Upsampling([128+17,64], 17,conv_layer=2)
        self.up_4 = Upsampling([64+17,32], 17,conv_layer=2)
        
        self.out_layer6 = torch.nn.Sequential(torch.nn.Conv2d(17,17, kernel_size=1, padding= 0,stride=1),torch.nn.Sigmoid())
        self.out_layer7 = torch.nn.Sequential(torch.nn.Conv2d(17,17, kernel_size=1, padding= 0,stride=1),torch.nn.Sigmoid())
        self.out_layer8 = torch.nn.Sequential(torch.nn.Conv2d(17,17, kernel_size=1, padding= 0,stride=1),torch.nn.Sigmoid())
        self.out_layer9 = torch.nn.Sequential(torch.nn.Conv2d(17,17, kernel_size=1, padding= 0,stride=1),torch.nn.Sigmoid())

    
    def forward(self,x):
        x1 = self.vgg.features[0:4](x)
        x2 = self.vgg.features[4:9](x1)
        x3 = self.vgg.features[9:16](x2)
        x4 = self.vgg.features[16:23](x3)
        x5 = self.vgg.features[23:30](x4)
        x6 = self.up_1(x5,x4)
        x7 = self.up_2(x6,x3)
        x8 = self.up_3(x7,x2)
        x9 = self.up_4(x8,x1)
        # print("x1",x1.shape,"x2",x2.shape,"x3",x3.shape,"x4",x4.shape,"x5",x5.shape,"x6",x6.shape,"x7",x7.shape,"x8",x8.shape,"x9",x9.shape)

        outx6 = self.out_layer6(x6)
        # padding, till the size of outputs are same as image size/downsampled size
        if outx6.shape != x4.shape:
            outx6 = torch.nn.functional.pad(outx6, (0, x4.shape[3] - outx6.shape[3], 0, x4.shape[2] - outx6.shape[2]),'replicate')
        outx7 = self.out_layer7(x7)
        if outx7.shape != x3.shape:
            outx7 = torch.nn.functional.pad(outx7, (0, x3.shape[3] - outx7.shape[3], 0, x3.shape[2] - outx7.shape[2]),'replicate')
        outx8 = self.out_layer8(x8)
        if outx8.shape != x2.shape:
            outx8 = torch.nn.functional.pad(outx8, (0, x2.shape[3] - outx8.shape[3], 0, x2.shape[2] - outx8.shape[2]),'replicate')
        outx9 = self.out_layer9(x9)
        if outx9.shape != x.shape:
            outx9 = torch.nn.functional.pad(outx9, (0, x.shape[-1] - outx9.shape[-1], 0, x.shape[-2] - outx9.shape[-2]),'replicate')
        
        # print("x9.shape: ",x9.shape)
        # (F3,S3),(F2,S2),(F1,S1),(F0,S0)
        return (outx6[:,:16],outx6[:,16:]),(outx7[:,:16],outx7[:,16:]),(outx8[:,:16],outx8[:,16:]),(outx9[:,:16],outx9[:,16:])
        

def main():
    net = UNet(3,17)
    net.train()
    x = torch.rand(1, 3, 800, 600)
    print("x.shape: ",x.shape)
    import numpy
    (F3_estimate,S3_estimate),(F2_estimate,S2_estimate),(F1_estimate,S1_estimate),(F0_estimate,S0_estimate) = net.forward(x)
    # torch.set_printoptions(threshold=numpy.inf)
    print("x",x)
    print("F3.shape: ",F3_estimate[:,:,-50:,-50:])
    print("S3.shape: ",S3_estimate[:,:,-50:,-50:])
    print("F2.shape: ",F2_estimate.shape)
    print("S2.shape: ",S2_estimate.shape)
    print("F1.shape: ",F1_estimate.shape)
    print("S1.shape: ",S1_estimate.shape)
    print("F0.shape",F0_estimate.shape)
    print("S0.shape",S0_estimate.shape)
if __name__ == "__main__":
    main()