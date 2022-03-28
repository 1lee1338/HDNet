import torch
import torch.nn as nn
import torch.nn.functional as F
from core.models.segbase import SegBaseModel

def get_hdnet(dataset='bcdd', backbone='resnet50', **kwargs):
    from ..data.dataloader import datasets
    model = HDNet(datasets[dataset].NUM_CLASS, backbone=backbone,  **kwargs)
    return model

class HDNet(nn.Module):
    def __init__(self, nclass , backbone='resnet50',  **kwargs):
        super(HDNet, self).__init__()
        self.head = hdnet(nclass, backbone=backbone,**kwargs)

    def forward(self, x):
        outputs = []
        m_c = []
        m_b = []
        x, y, z = self.head(x)
        outputs.append(x)
        m_c.append(y)
        m_b.append(z)

        return tuple(outputs),tuple(m_c),tuple(m_b)


class hdnet(SegBaseModel):

    def __init__(self, num_classes = 2, backbone='resnet50',  **kwargs):
        super(hdnet, self).__init__(num_classes,backbone,  **kwargs)

    # compress channels
        self.conv_s1 = conv1_norm_relu(256, 64)
        self.conv_s2 = conv1_norm_relu(512, 128)
        self.conv_s3 = conv1_norm_relu(1024, 256)
        self.conv_s4 = conv1_norm_relu(2048, 512)
    #from deep to low
        self.s4_conv = conv1_norm_relu(512, 256)
        self.s3_conv = conv1_norm_relu(256, 128)
        self.s2_conv = conv1_norm_relu(128, 64)

        self.body_3 = conv1_norm_relu(256, 128)
        self.body_2 = conv1_norm_relu(256, 64)
        self.gate_body23 = gatehead(128,128)
        self.gate_body12 = gatehead(64,64)

        self.edge_3 = conv1_norm_relu(256, 64)
        self.edge_2 = conv1_norm_relu(128, 64)
        self.edge_12 = conv1_norm_relu(128, 64)
        self.gate_edge12 = gatehead(64,64)
        self.gate_edge23 = gatehead(64,64)

    #semantic decoupled
        self.squeeze_body_edge_1 = Disentangle(64, Norm2d)
        self.squeeze_body_edge_2 = Disentangle(128, Norm2d)
        self.squeeze_body_edge_3 = Disentangle(256, Norm2d)
    #body\edge aspp
        self.aspp_body = ASPPModule(128, reduction_dim=64, output_stride=16)
        self.bot_aspp_body = conv1_norm_relu(192, 128)
        self.aspp_edge = ASPPModule(128, reduction_dim=64, output_stride=16)
        self.bot_aspp_edge = conv1_norm_relu(192, 128)

     # out
        #edge
        self.edge_out = nn.Sequential(
            nn.Conv2d(128,64, kernel_size=3, padding=1, bias=False),
            Norm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, bias=False) )
        # body
        self.body_out = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            Norm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1, bias=False) )
        # seg
        self.final_seg = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            Norm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            Norm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1, bias=False))
    def forward(self, inp, gts=None):
        x_size = inp.size()
        s1_features, s2_features, s3_features, s4_features = self.base_forward(inp)
    #compress channels
        s1_features = self.conv_s1(s1_features)
        s2_features = self.conv_s2(s2_features)
        s3_features = self.conv_s3(s3_features)
        s4_features = self.conv_s4(s4_features)

    #semantic decoupled
        s4_feats = self.s4_conv(s4_features)
        s4_feats = Upsample(s4_feats,(32,32))
        seg_body_3, seg_edge_3 = self.squeeze_body_edge_3(s4_feats,s3_features)

        s3_feats = self.s3_conv(s3_features)
        s3_feats = Upsample(s3_feats,(64,64))
        seg_body_2, seg_edge_2 = self.squeeze_body_edge_2(s3_feats, s2_features)

        s2_feats = self.s2_conv(s2_features)
        s2_feats = Upsample(s2_feats,(128,128))
        seg_body_1, seg_edge_1 = self.squeeze_body_edge_1(s2_feats, s1_features)

    #fuse different level features
        #edge
        seg_edge_2 = self.edge_2(seg_edge_2)
        seg_edge_2 = Upsample(seg_edge_2, (128, 128))
        seg_edge_1,seg_edge_2 = self.gate_edge12(seg_edge_1,seg_edge_2)
        seg_edge_12 = torch.cat([seg_edge_1, seg_edge_2], dim=1)
        seg_edge_12 = self.edge_12(seg_edge_12)
        seg_edge_3 = self.edge_3(seg_edge_3)
        seg_edge_3 = Upsample(seg_edge_3, (128, 128))
        seg_edge_12,seg_edge_3 = self.gate_edge23(seg_edge_12,seg_edge_3)
        seg_edge_123 = torch.cat([seg_edge_12, seg_edge_3], dim=1)

        seg_edge = self.aspp_edge(seg_edge_123)
        seg_edge = self.bot_aspp_edge(seg_edge)
        #body
        seg_body_3 = self.body_3(seg_body_3)
        seg_body_3 = Upsample(seg_body_3,(64,64))
        seg_body_2,seg_body_3 = self.gate_body23(seg_body_2,seg_body_3)
        seg_body_23 = torch.cat([seg_body_2, seg_body_3], dim=1)
        seg_body_23 = self.body_2(seg_body_23)
        seg_body_23 = Upsample(seg_body_23,(128,128))
        seg_body_1,seg_body_23 = self.gate_body12(seg_body_1,seg_body_23)
        seg_body_123 = torch.cat([seg_body_1, seg_body_23], dim=1)

        seg_body = self.aspp_body(seg_body_123)
        seg_body = self.bot_aspp_body(seg_body)
        # seg
        seg_out = seg_body + seg_edge
        seg_final = self.final_seg(seg_out)

    #out
        seg_final_out = Upsample(seg_final, x_size[2:])

        #edge
        seg_edge_out = Upsample(self.edge_out(seg_edge), x_size[2:])

        #body
        seg_body_out = Upsample(self.body_out(seg_body), x_size[2:])

        return seg_final_out,seg_body_out, seg_edge_out

class gatehead(nn.Module):
    def __init__(self,inchannels,outchannels):
        super(gatehead, self).__init__()
        self.gate_x = nn.Sequential(
            nn.Conv2d(inchannels, outchannels,kernel_size=1),
            nn.Sigmoid() )
        self.gate_y = nn.Sequential(
            nn.Conv2d(inchannels, outchannels, kernel_size=1),
            nn.Sigmoid())
    def forward(self,x,y):

        g_x = self.gate_x(x)
        g_y = self.gate_y(y)
        x = x + x * g_x + (1-g_x) * g_y * y
        y = y + y * g_y + (1-g_y) * g_x * x

        return x,y

def Upsample(x, size):

    return nn.functional.interpolate(x, size=size, mode='bilinear',
                                     align_corners=True)

def Norm2d(inplanes):
    return nn.BatchNorm2d(inplanes)


class conv1_norm_relu(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(conv1_norm_relu, self).__init__()
        self.inchannels = inchannels
        self.outchannels = outchannels
        self.conv = nn.Conv2d(inchannels, outchannels, kernel_size=1, bias=False)
        self.norm2d = nn.BatchNorm2d(outchannels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm2d(x)
        x = self.relu(x)

        return x

class ASPPModule(nn.Module):
    def __init__(self, in_dim, reduction_dim=256, output_stride=16,
                 rates=(2, 5)):
        super(ASPPModule, self).__init__()
        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise 'output stride of {} not supported'.format(output_stride)
        self.features = []
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, bias=False),
                torch.nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)
        self.conv1 = nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=3,dilation=1, padding=1, bias=False),
                                   torch.nn.BatchNorm2d(reduction_dim),
                                   nn.ReLU(inplace=True))

    def forward(self, x):
        img_features = self.conv1(x)
        out = img_features
        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out

class Disentangle(nn.Module):
    def __init__(self, inplane, norm_layer):
        super(Disentangle, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(inplane, inplane, kernel_size=3, groups=inplane, stride=2),
            norm_layer(inplane),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplane, inplane, kernel_size=3, groups=inplane, stride=2),
            norm_layer(inplane),
            nn.ReLU(inplace=True) )
        self.flow_make = nn.Conv2d(inplane *2 , 2, kernel_size=3, padding=1, bias=False)
    def forward(self, x,y):
        size = x.size()[2:]
        flow = self.flow_make(torch.cat([x, y], dim=1))
        seg_flow_warp = self.flow_warp(x, flow, size)
        seg_edge = x - seg_flow_warp
        return seg_flow_warp, seg_edge

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()
        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        # new
        h_grid = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w_gird = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w_gird.unsqueeze(2), h_grid.unsqueeze(2)), 2)

        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid)
        return output



if __name__ == '__main__':

    img = torch.randn(4, 3, 512, 512)
    model = HDNet( 2)
    output,a,b = model(img)
    print(output[0].size())
    print(a[0].size())
    print(b[0].size())
