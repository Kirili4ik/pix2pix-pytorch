import torch
from torch.utils.data import DataLoader
import os



from my_utils import set_seed, count_parameters
from my_dataset import MyDataset, my_transforms_tr, my_transforms_val
from model import Pix2Pix
from val_test import validate, test

import wandb
from tqdm import tqdm





if __name__ == "__main__":

    #os.system("wget http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz")
    #os.system("tar -xzf facades.tar.gz")

    BATCH_SIZE = 1
    NUM_EPOCHS = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    set_seed(21)

    ### dataset creation
    train_set = MyDataset(root_dir='../facades/train/', transform=my_transforms_tr)
    val_set   = MyDataset(root_dir='../facades/val/', transform=my_transforms_val)
    test_set  = MyDataset(root_dir='../facades/test/', transform=my_transforms_val)

    ### dataloaders
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=2, pin_memory=True)

    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE,
                            shuffle=False,
                            num_workers=1, pin_memory=True)

    test_loader = DataLoader(test_set, batch_size=1,
                            shuffle=False,
                            num_workers=1, pin_memory=True)
    ### model creatng
    model = Pix2Pix()
    model = model.to(device)
    print(count_parameters(model))

    ### init weights of model: conv, linear layers to N(0, 0.02), biases to 0
    for m in model.parameters():
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            init.normal_(m.weight.data, 0.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    ### WANDB
    wandb.init(project='dl-ht3')
    wandb.watch(model)

    ### create optimizers
    opt_D = torch.optim.Adam(model.netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_G = torch.optim.Adam(model.netG.parameters(), lr=0.0002, betas=(0.5, 0.999))


    ### Train Loop
    for ep_num in tqdm(range(NUM_EPOCHS)):
        model.train()
        total_loss_D, total_loss_G = 0, 0
        total_ssim = 0
        for i, batch in enumerate(train_loader):
            real_x, real_y = batch['x'].to(device), batch['target'].to(device)

            loss_D, loss_G, ssim = model.my_backward(real_x, real_y, opt_D, opt_G)

            total_loss_D = total_loss_D + loss_D
            total_loss_G = total_loss_G + loss_G
            total_ssim = total_ssim + ssim
            torch.nn.utils.clip_grad_norm_(model.parameters(), 15)
            if i == 3:
                break

        if ep_num % 20 == 0:
            val_loss_D, val_loss_G, val_ssim = validate(model, val_loader, is_saving=True, device=device)
        else:
            val_loss_D, val_loss_G, val_ssim = validate(model, val_loader, device=device)

        if ep_num >= 150 and ep_num % 25 == 0:
            a, b = test(model, test_loader, device)

        wandb.log({'train_D':total_loss_D/len(train_loader),
                   'train_G_l1+cgan':total_loss_G/len(train_loader),
                   'train_ssim':total_ssim/len(train_loader),
                   'val_D':val_loss_D,
                   'val_G_l1+cgan':val_loss_G,
                   'val_ssim':val_ssim})

    ### save model
    torch.save({
        'model_state_dict': model.state_dict(),
    }, 'checkpoint')
