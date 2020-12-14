import torch
import torchvision
from torchvision import transforms
import random
import numpy as np



def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


def my_imshow(inp, tar):
    """Imshow for Tensor."""
    inp = transforms.ToPILImage()(inp)
    tar = transforms.ToPILImage()(tar)

    f, axarr = plt.subplots(1, 2)
    f.set_figheight(5)
    f.set_figwidth(10)

    axarr[0].imshow(inp)
    axarr[1].imshow(tar)

    plt.show()
    plt.pause(0.1)


def count_psnr(fake, real):
    mse_loss = F.mse_loss(fake*0.5 + 0.5, real, reduction='none')
    return torch.mean(10 * torch.log10(1 / mse_loss)).item()


def count_cov(x, y):
    x -= torch.mean(x, dim=1, keepdim=True)
    y -= torch.mean(y, dim=1, keepdim=True)

    return x * y


# based on wiki formula
def count_ssim(x, y):
    mu_x = torch.mean(x, dim=[2,3]).squeeze()
    mu_y = torch.mean(y, dim=[2,3]).squeeze()

    sig_x = torch.var(x, dim=[2,3]).squeeze()
    sig_y = torch.var(y, dim=[2,3]).squeeze()

    sig_xy = torch.mean(count_cov(x.reshape(3, -1), y.reshape(3, -1)), 1)

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim = ((2 * mu_x * mu_y + c1) * (2 * sig_xy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sig_x + sig_y + c2))

    return torch.mean(ssim)


