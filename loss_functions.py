import torch
import torch.nn.functional as F
import math

def MSEloss(yhat, y):
    """
    Computes the Mean Squared Difference between prediction and label.
    Parameters    yhat (tensor): tensor of the prediction values
                  y (tensor): tensor of the true labels
    """
    squared_diffs = (yhat - y)**2
    return squared_diffs.mean()

def gaussian(window_size, sigma):
    """
    Creates a gaussian kernel. 
    Taken from https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/
    Parameters:     window_size (int): kernel size
                    sigma (float): standard deviation of window
    """
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def SSIMloss(yhat, y, window_size=11):
    """
    Computes the SSIM loss for a given prediction and labels. The code is based the pytorch-ssim package and 
    is slightly adjusted for the MNIST dataset. https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/
    Parameters:     yhat (tensor): tensor of the prediction values
                    y (tensor):  tensor of the true labels
                    window_size (int): default value is 11
    """
    # Get amount of channels
    channel = yhat.shape[1]
    
    # create a gauss weighting window
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.autograd.Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    
    # Compute local mean per channel
    mu1 = F.conv2d(yhat, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(y, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2
    
    # Compute local sigma per channel
    sigma1_sq = F.conv2d(yhat*yhat, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(y*y, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(yhat*y, window, padding = window_size//2, groups = channel) - mu1_mu2
    
    # Stability constant for luminance
    C1 = 0.01**2
    # Stability contant for contrast
    C2 = 0.03**2
    
    # Compute the similarity index map
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    
    return 1-ssim_map.mean()