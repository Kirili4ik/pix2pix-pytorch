import torch
from torchvision import transforms

import wandb


@torch.no_grad()
def validate(model, loader, is_saving=False, device='cuda'):
    model.netD.eval()
    total_loss_D, total_loss_G = 0, 0
    total_ssim = 0
    for i, batch in enumerate(loader):
        real_x, real_y = batch['x'].to(device), batch['target'].to(device)

        loss_D, loss_G, ssim = model.validation(real_x, real_y)

        total_loss_D = total_loss_D + loss_D
        total_loss_G = total_loss_G + loss_G
        total_ssim   = total_ssim   + ssim

    if is_saving:
        wandb.log({'examples':
                  [wandb.Image(transforms.ToPILImage()(real_x[0]*0.5 + 0.5), caption='real_x'),
                  wandb.Image(transforms.ToPILImage()(model.generate(real_x)[0]*0.5 + 0.5), caption='generated'),
                  wandb.Image(transforms.ToPILImage()(real_y[0]*0.5 + 0.5), caption='real_y')
                  ]})

    return total_loss_D / len(loader), total_loss_G / len(loader), total_ssim / len(loader)


### same as validate, but saves every second example
@torch.no_grad()
def test(model, loader, device='cuda'):
    model.netD.eval()
    total_loss_D, total_loss_G = 0, 0
    for i, batch in enumerate(loader):
        real_x, real_y = batch['x'].to(device), batch['target'].to(device)

        loss_D, loss_G, ssim = model.validation(real_x, real_y)

        total_loss_D = total_loss_D + loss_D
        total_loss_G = total_loss_G + loss_G

        if i % 2 == 0:
            wandb.log({'examples':
                  [wandb.Image(transforms.ToPILImage()(real_x[0]*0.5 + 0.5), caption='real_x'),
                  wandb.Image(transforms.ToPILImage()(model.generate(real_x)[0]*0.5 + 0.5), caption='generated'),
                  wandb.Image(transforms.ToPILImage()(real_y[0]*0.5 + 0.5), caption='real_y')
                  ]})


    return total_loss_D / len(loader), total_loss_G / len(loader)
