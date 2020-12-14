import torch
import torch.nn as nn
import torch.nn.functional as F

from my_utils import count_ssim


def conv_bn_act(in_size, out_size, kernel_size=4, stride=2, dilation=1, padding=None, bias=False,
                is_bn=True, p_drop=None, act=None, is_up=False):

    padding=int((kernel_size - 1)/2)

    return nn.Sequential(
        nn.Conv2d(in_size, out_size, kernel_size, stride=stride, padding=padding, bias=bias) \
            if not is_up else \
            nn.ConvTranspose2d(in_size, out_size, kernel_size, stride=stride, padding=padding, bias=bias),
        nn.BatchNorm2d(out_size, eps=1e-05, momentum=0.1, affine=True) if is_bn else nn.Identity(),
        nn.Dropout(p_drop) if p_drop is not None else nn.Identity(),
        act if act is not None else nn.Identity()
    )



class UNet(nn.Module):
    def __init__(self, in_size=64):
        super(UNet, self).__init__()

        enc_act, dec_act = nn.LeakyReLU(0.2), nn.ReLU()
        self.encoder = nn.ModuleList([conv_bn_act(3,         in_size, is_bn=False, act=enc_act),
                                      conv_bn_act(in_size,   in_size*2, act=enc_act),
                                      conv_bn_act(in_size*2, in_size*4, act=enc_act),
                                      conv_bn_act(in_size*4, in_size*8, act=enc_act)
                                     ] +
                                     [conv_bn_act(in_size*8, in_size*8, act=enc_act)
                                      for i in range(3)] +
                                     [conv_bn_act(in_size*8, in_size*8, is_bn=False, act=enc_act)])

        self.first_dec_layer = conv_bn_act(in_size*8, in_size*8, p_drop=0.5, act=dec_act, is_up=True)

        self.decoder = nn.ModuleList([conv_bn_act(in_size*16, in_size*8, p_drop=0.5, act=dec_act, is_up=True)
                                      for i in range(3)] +
                                     [conv_bn_act(in_size*16, in_size*4, act=dec_act, is_up=True),
                                      conv_bn_act(in_size*8,  in_size*2, act=dec_act, is_up=True),
                                      conv_bn_act(in_size*4,  in_size,   act=dec_act, is_up=True),
                                      conv_bn_act(in_size*2,  3, bias=True, is_bn=False, act=nn.Tanh(), is_up=True)])


    def forward(self, x):
        N = len(self.encoder) - 1

        enc_outputs = []
        for layer in self.encoder:
            enc_outputs.append(x)
            x = layer(x)

        x = self.first_dec_layer(x)

        for i, layer in enumerate(self.decoder):
            x_concat = torch.cat([x, enc_outputs[N-i]], dim=1)
            x = layer(x_concat)
        return x



# real_x,y ; fake_x,y . Making x->y model
class Pix2Pix(nn.Module):
    def __init__(self, in_size=64):
        super(Pix2Pix, self).__init__()

        self.netG = UNet(in_size)
        self.netD = nn.Sequential(conv_bn_act(6, in_size, is_bn=False, bias=True, act=nn.LeakyReLU(0.2)),
                                  conv_bn_act(in_size, in_size*2, act=nn.LeakyReLU(0.2)),
                                  conv_bn_act(in_size*2, in_size*4, act=nn.LeakyReLU(0.2)),
                                  conv_bn_act(in_size*4, in_size*8, stride=1, act=nn.LeakyReLU(0.2)),
                                  conv_bn_act(in_size*8, 1, stride=1, bias=True, is_bn=False)     # no sigmoid cause of BCEwithlogits
                                 )
        self.cgan_loss = nn.BCEWithLogitsLoss()
        self.l1_loss  = nn.L1Loss()


    def generate(self, real_x):
        fake_y = self.netG(real_x)
        return fake_y


    def my_backward(self, real_x, real_y, optimizer_D, optimizer_G, lambda_par=100):
        fake_y = self.generate(real_x)

        # enable backprop for D if disabled
        for param in self.netD.parameters():
            param.requires_grad = True
        optimizer_D.zero_grad()

        ### DICRIMINATOR BACKWARD
        # real-fake
        real_fake = torch.cat((real_x, fake_y), 1)
        prediction_D = self.netD(real_fake.detach())     # detach for G backward NOT to be used
        prediction_D = prediction_D.view(prediction_D.size(0), -1)
        loss_D_fake = self.cgan_loss(prediction_D, torch.zeros_like(prediction_D))

        # real-real
        real_real = torch.cat((real_x, real_y), 1)
        prediction_D = self.netD(real_real)
        prediction_D = prediction_D.view(prediction_D.size(0), -1)
        loss_D_real = self.cgan_loss(prediction_D, torch.ones_like(prediction_D))

        loss_D = (loss_D_fake + loss_D_real) / 2
        loss_D.backward()
        optimizer_D.step()

        ###
        # disable backprop for D for saving Time and Resourses
        for param in self.netD.parameters():
            param.requires_grad = False

        ### GENERATOR BACKWARD
        optimizer_G.zero_grad()

        prediction_D = self.netD(real_fake)               # don't use detach for G backward TO be used
        prediction_D = prediction_D.view(prediction_D.size(0), -1)
        loss_cgan = self.cgan_loss(prediction_D, torch.ones_like(prediction_D))

        loss_L1 = self.l1_loss(fake_y, real_y) * lambda_par
        loss_G = loss_cgan + loss_L1
        loss_G.backward()
        optimizer_G.step()

        ssim = count_ssim(fake_y.detach(), real_y.detach())

        return loss_D.item(), loss_G.item(), ssim


    ##### Same as my_backward, but without optimizers and .barckward() for losses
    def validation(self, real_x, real_y, lambda_par=100):
        fake_y = self.generate(real_x)

        ### DICRIMINATOR BACKWARD
        # real-fake
        real_fake = torch.cat((real_x, fake_y), 1)
        prediction_D = self.netD(real_fake.detach())
        prediction_D = prediction_D.view(prediction_D.size(0), -1)
        loss_D_fake = self.cgan_loss(prediction_D, torch.zeros_like(prediction_D))

        # real-real
        real_real = torch.cat((real_x, real_y), 1)
        prediction_D = self.netD(real_real)
        prediction_D = prediction_D.view(prediction_D.size(0), -1)
        loss_D_real = self.cgan_loss(prediction_D, torch.ones_like(prediction_D))

        loss_D = (loss_D_fake + loss_D_real) / 2

        ### GENERATOR BACKWARD
        prediction_D = self.netD(real_fake)
        prediction_D = prediction_D.view(prediction_D.size(0), -1)
        loss_cgan = self.cgan_loss(prediction_D, torch.ones_like(prediction_D))
        loss_L1 = self.l1_loss(fake_y, real_y) * lambda_par
        loss_G = loss_cgan + loss_L1

        ssim = count_ssim(fake_y.detach(), real_y.detach())

        return loss_D.item(), loss_G.item(), ssim
