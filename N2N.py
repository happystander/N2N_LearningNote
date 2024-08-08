from typing import Optional, List
from skimage.metrics import structural_similarity as ssim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange 
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model import *
from scipy.ndimage import convolve
# Load test image from URL

from io import BytesIO
from utils import *

device = "cuda"
proxies = {
'http': 'http://localhost:7890',
'https': 'https://localhost:7890',
}


# 4 test images from the Kodak24 dataset

#url = "https://drive.google.com/uc?export=download&id=18LcKoV4SYusF16wKqwNBJwYpTXE9myie"
#url = "https://drive.google.com/uc?export=download&id=176lM7ONjvyC83GcllCod-j1RPqjLoRoG"
#url = "https://drive.google.com/uc?export=download&id=1UIh9CwXSCf01JmAXgJo0LPtw5TUkWUU-"
url = "https://drive.google.com/uc?export=download&id=1j1OOzvGhet_GHJCaXbfisiW8uGDxI7ty"



def add_noise(x,noise_level):

    if noise_type == 'gauss':
        noisy = x + torch.normal(0, noise_level/255, x.shape)
        noisy = torch.clamp(noisy,0,1)

    elif noise_type == 'poiss':
        noisy = torch.poisson(noise_level * x)/noise_level

    elif noise_type == "saltAndPepper":
        noisy = salt_and_pepper(x,noise_level)

    return noisy

#============pair_downsampler============
# "orignal":0,
# "remoteWeight":1,
# "remoteNegativeWeight":2,
# "threeSizeSample":3,
# "stripe":4,
# "maxAndAvgPool":5,
# "avgPoolAndRandomPool":6,
# "maxPoolAndRandomPool":7,
#========================================

def downsampler_tool(img,sampleType = 0,upSample = False):
    # img has shape B C H W
    c = img.shape[1]

    # 上采样
    if upSample:
        m = torch.nn.Upsample(scale_factor=2, mode='bilinear')
        img = m(img)

    if sampleType == downsamplerType["original"]:
        # original
        filter1 = torch.FloatTensor([[[[0, 0.5], [0.5, 0]]]]).to(img.device)
        filter1 = filter1.repeat(c, 1, 1, 1)

        filter2 = torch.FloatTensor([[[[0.5, 0], [0, 0.5]]]]).to(img.device)
        filter2 = filter2.repeat(c, 1, 1, 1)

        output1 = F.conv2d(img, filter1, stride=2, groups=c)
        output2 = F.conv2d(img, filter2, stride=2, groups=c)

    elif sampleType == downsamplerType["remoteWeight"]:
        # 边缘权重
        filter1 = torch.FloatTensor([[[[0.1, 0.5], [0.5, 0.1]]]]).to(img.device)
        filter1 = filter1.repeat(c, 1, 1, 1)

        filter2 = torch.FloatTensor([[[[0.5, 0.1], [0.1, 0.5]]]]).to(img.device)
        filter2 = filter2.repeat(c, 1, 1, 1)

        output1 = F.conv2d(img, filter1, stride=2, groups=c) / 1.2
        output2 = F.conv2d(img, filter2, stride=2, groups=c) / 1.2


    elif sampleType == downsamplerType["remoteNegativeWeight"]:
        # original  负边缘权重
        filter1 = torch.FloatTensor([[[[-0.1, 0.5], [0.5, -0.1]]]]).to(img.device)
        filter1 = filter1.repeat(c, 1, 1, 1)

        filter2 = torch.FloatTensor([[[[0.5, -0.1], [-0.1, 0.5]]]]).to(img.device)
        filter2 = filter2.repeat(c, 1, 1, 1)

        output1 = F.conv2d(img, filter1, stride=2, groups=c) / 0.8
        output2 = F.conv2d(img, filter2, stride=2, groups=c) / 0.8

    elif sampleType == downsamplerType["threeSizeSample"]:
        # 3X3 sample
        filter1 = torch.FloatTensor([[[[0, 0.5, 0], [0.5, 0.01, 0.5], [0, 0.5, 0]]]]).to(img.device)
        filter1 = filter1.repeat(c, 1, 1, 1)

        filter2 = torch.FloatTensor([[[[0.5, 0, 0.5], [0, 0.01, 0], [0.5, 0, 0.5]]]]).to(img.device)
        filter2 = filter2.repeat(c, 1, 1, 1)

        output1 = F.conv2d(img, filter1, stride=3, groups=c) / 2.01
        output2 = F.conv2d(img, filter2, stride=3, groups=c) / 2.01

    elif sampleType == downsamplerType["stripe"]:
        # 2X2 条纹
        filter1 = torch.FloatTensor([[[[0, 0.5], [0, 0.5]]]]).to(img.device)
        filter1 = filter1.repeat(c, 1, 1, 1)

        filter2 = torch.FloatTensor([[[[0.5, 0], [0.5, 0]]]]).to(img.device)
        filter2 = filter2.repeat(c, 1, 1, 1)

        output1 = F.conv2d(img, filter1, stride=2, groups=c)
        output2 = F.conv2d(img, filter2, stride=2, groups=c)

    elif sampleType == downsamplerType["maxAndAvgPool"]:
        # 池化
        output1 = F.max_pool2d(img, kernel_size=2, stride=2)
        output2 = F.avg_pool2d(img, kernel_size=2, stride=2)

    elif sampleType == downsamplerType["avgPoolAndRandomPool"]:
        # 随机池化
        output1 = F.avg_pool2d(img, kernel_size=2, stride=2)
        randomPooler = StochasticPool2DLayer(maxpool=False, training=True)
        output2 = randomPooler(img)

    elif sampleType == downsamplerType["maxPoolAndRandomPool"]:
        # 随机池化
        output1 = F.max_pool2d(img, kernel_size=2, stride=2)
        randomPooler = StochasticPool2DLayer(maxpool=False, training=True)
        output2 = randomPooler(img)

    elif sampleType == downsamplerType["RRandomPoolAndRandomPool"]:
        # 随机池化
        randomPooler1 = StochasticPool2DLayer(maxpool=False, training=True)
        output1 = randomPooler1(img)
        randomPooler2 = StochasticPool2DLayer(maxpool=False, training=True)
        output2 = randomPooler2(img)

    elif sampleType == downsamplerType["threeSizeSamplePlus"]:
        counter = 10
        filter1 = torch.FloatTensor([[[[-10, -3, 0],
                                       [-3, counter, 3],
                                       [0, 3, 10]]]]).to(img.device)
        filter1 = filter1.repeat(c, 1, 1, 1)

        # filter2 = torch.FloatTensor([[[[0, 1,2],
        #                                [1, 2, -1],
        #                                [-2, -1,0]]]]).to(img.device)
        filter2 = torch.FloatTensor([[[[0, 3, 10],
                                       [3, counter, -3],
                                       [-10, -3, 0]]]]).to(img.device)
        filter2 = filter2.repeat(c, 1, 1, 1)

        output1 = F.conv2d(img, filter1, stride=3, groups=c) / counter
        output2 = F.conv2d(img, filter2, stride=3, groups=c) / counter

    elif sampleType == downsamplerType["stripeReverse"]:
        # 2X2 条纹
        filter1 = torch.FloatTensor([[[[0.5, 0.5], [0, 0]]]]).to(img.device)
        filter1 = filter1.repeat(c, 1, 1, 1)

        filter2 = torch.FloatTensor([[[[0, 0], [0.5, 0.5]]]]).to(img.device)
        filter2 = filter2.repeat(c, 1, 1, 1)

        output1 = F.conv2d(img, filter1, stride=2, groups=c)
        output2 = F.conv2d(img, filter2, stride=2, groups=c)
    else:
        raise f"None {sampleType} type..."

    return output1, output2


def pair_downsampler(img,sampleType = 0,upSample = False,mini_epoch = 0,lastSampleType = None):
    #img has shape B C H W
    c = img.shape[1]

    # 上采样
    if upSample:
        m = torch.nn.Upsample(scale_factor=2, mode='bilinear')
        img = m(img)

    if lastSampleType != None:
        return *downsampler_tool(img,lastSampleType, upSample),lastSampleType

    if use_step_filter and (mini_epoch < 1000 ) :
        if mini_epoch != None and mini_epoch % 2 == 0:
            sampleType = downsamplerType["stripe"] # todo change

        if mini_epoch != None and mini_epoch % 3 == 0:
            sampleType = downsamplerType["stripeReverse"] # todo change

        if mini_epoch != None and mini_epoch % 5 == 0:
            sampleType = downsamplerType["remoteWeight"] # todo change

        if mini_epoch != None and mini_epoch % 10 == 0:
            sampleType = downsamplerType["remoteNegativeWeight"] # todo change threeSizeSamplePlus

        if mini_epoch != None and mini_epoch % 20 == 0:
            sampleType = downsamplerType["threeSizeSample"] # todo change remoteNegativeWeight

        return *downsampler_tool(img,sampleType, upSample),sampleType

    if 500<= mini_epoch <= 3000:

        return *downsampler_tool(img,downsamplerType["stripeReverse"], upSample),downsamplerType["stripeReverse"]

    # if 2500 <= mini_epoch <= 3000:
    #     return *downsampler_tool(img, sampleType, upSample), sampleType

    if use_max_calhist:
        calhistMethod :List[str] =["stripeReverse"] #["stripeReverse","threeSizeSample","avgPoolAndRandomPool"]
        ssim_list = []
        res_list = []
        for downsampleMethod in calhistMethod:
            o1,o2 = downsampler_tool(img,sampleType=downsamplerType[downsampleMethod], upSample=upSample)
            res_list.append([o1,o2])
            # o1_np = torch.squeeze(o1).detach().cpu().numpy()
            # o2_np = torch.squeeze(o2).detach().cpu().numpy()
            #
            # # 确保图像至少是7x7，并且win_size是合适的
            # min_size = min(o1_np.shape[:2])  # 假设图像是HxW或CxHxW格式
            # win_size = min(11, min_size)  # 使用11x11窗口，但如果图像较小，则减小窗口大小
            # if win_size % 2 == 0:  # 确保win_size是奇数
            #     win_size -= 1
            #
            # score, diff = ssim(o1_np, o2_np, full=True, win_size=win_size,data_range=1)
            score = mse(o1,o2).detach().cpu()

            ssim_list.append(score)

        max_idx = np.argmax(ssim_list)
        print("max_idx",max_idx)

        return *res_list[max_idx], downsamplerType[calhistMethod[max_idx]]

def mse(gt: torch.Tensor, pred:torch.Tensor)-> torch.Tensor:
    loss = torch.nn.MSELoss()
    return loss(gt,pred)

def loss_func(noisy_img,epoch = None):
    noisy1, noisy2,lastSampleType = pair_downsampler(noisy_img,sampleType=global_sampleType,upSample=global_upsampleFlag,mini_epoch=epoch)
    pred1 =  noisy1 - model(noisy1)
    pred2 =  noisy2 - model(noisy2)


    loss_res = 1/2*(mse(noisy1,pred2)+mse(noisy2,pred1))

    noisy_denoised =  noisy_img - model(noisy_img)
    denoised1, denoised2,lastSampleType= pair_downsampler(noisy_denoised,
                                                          sampleType=global_sampleType,
                                                          upSample=global_upsampleFlag,
                                                          mini_epoch=epoch,
                                                          lastSampleType=lastSampleType)

    loss_cons=1/2*(mse(pred1,denoised1) + mse(pred2,denoised2))

    loss = loss_res + loss_cons

    return loss

def train(optimizer, noisy_img, epoch = Optional[int]):

  loss = loss_func(noisy_img,epoch)

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  return loss.item()

def test(model, noisy_img, clean_img):

    with torch.no_grad():
        pred = torch.clamp(noisy_img - model(noisy_img),0,1)
        MSE = mse(clean_img, pred).item()
        PSNR = 10*np.log10(1/MSE)

    return PSNR

def denoise(model, noisy_img):

    with torch.no_grad():
        pred = torch.clamp( noisy_img - model(noisy_img),0,1)

    return pred

def trainALL():
    PSNRList = []
      # 训练
    global epoch
    for epoch in trange(max_epoch):
        model.train()
        loss_val = train(optimizer, noisy_img,epoch)
        scheduler.step()
        if epoch % 10 == 0:
            model.eval()
            PSNR = test(model, noisy_img, clean_img)
            PSNRList.append(PSNR)
            print(f"epoch:{epoch}: loss:{loss_val} PSNR:{PSNR}")
    return PSNRList


# config
max_epoch = 3000     # training epochs
lr = 0.001           # learning rate 0.001
step_size = 1000     # number of epochs at which learning rate decays
gamma = 0.5          # factor by which learning rate decays
downsamplerType = {
    "original":0,
    "remoteWeight":1,
    "remoteNegativeWeight":2,
    "threeSizeSample":3,
    "stripe":4,
    "maxAndAvgPool":5,
    "avgPoolAndRandomPool":6,
    "maxPoolAndRandomPool":7,
    "RRandomPoolAndRandomPool":8,
    "threeSizeSamplePlus":9,
    "stripeReverse":10
}

# 添加噪声
noise_type = 'gauss' # Either 'gauss' or 'poiss'
noise_level = 25    # Pixel range is 0-255 for Gaussian, and 0-1 for Poission
typeName = "avgPoolAndRandomPool"
global_sampleType = downsamplerType[typeName]
global_upsampleFlag = False
use_step_filter = True
use_max_calhist = True

if __name__ == '__main__':

    # response = requests.get(url,proxies=proxies)
    fp = open("00023.00023",'rb')
    data = fp.read()
    path=BytesIO(data)
    clean_img = torch.load(path).unsqueeze(0)
    print(clean_img)
    noisy_img = add_noise(clean_img, noise_level)

    # noise_type = 'saltAndPepper' # Either 'gauss' or 'poiss' or
    # noise_level = 0.05   # Pixel range is 0-255 for Gaussian, and 0-1 for Poission ,0-1 for salt and pepper
    # noisy_img = add_noise(clean_img, noise_level)

    clean_img = clean_img.to(device)
    noisy_img = noisy_img.to(device)

    # 定义模型
    n_chan = clean_img.shape[1]
    model = network(n_chan)
    model = model.to(device)
    print("The number of parameters of the network is: ",  sum(p.numel() for p in model.parameters() if p.requires_grad))
    # 下采样分析
    img1, img2,_ = pair_downsampler(noisy_img)

    img0 = noisy_img.cpu().squeeze(0).permute(1,2,0)
    img1 = img1.cpu().squeeze(0).permute(1,2,0)
    img2 = img2.cpu().squeeze(0).permute(1,2,0)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # 训练
    PSNRList = trainALL()

    # 模型测试
    PSNR = test(model, noisy_img, clean_img)
    print(PSNR)

    # 预测
    denoised_img = denoise(model, noisy_img)

    denoised = denoised_img.cpu().squeeze(0).permute(1,2,0)
    clean = clean_img.cpu().squeeze(0).permute(1,2,0)
    noisy = noisy_img.cpu().squeeze(0).permute(1,2,0)

    fig, ax = plt.subplots(1, 6,figsize=(14, 4))

    ax[0].imshow(img0)
    ax[0].set_title('Noisy Img')
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    ax[1].imshow(img1)
    ax[1].set_title('First downsampled')
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    ax[2].imshow(img2)
    ax[2].set_title('Second downsampled')
    ax[2].set_xticks([])
    ax[2].set_yticks([])

    ax[3].imshow(clean)
    ax[3].set_xticks([])
    ax[3].set_yticks([])
    ax[3].set_title('Ground Truth')

    ax[4].imshow(noisy)
    ax[4].set_xticks([])
    ax[4].set_yticks([])
    ax[4].set_title('Noisy Img')
    try:
        noisy_psnr = 10*np.log10(1/mse(noisy_img,clean_img).item())
    except :
        noisy_psnr = 0.0
    ax[4].set(xlabel= str(round(noisy_psnr,2)) + ' dB')

    ax[5].imshow(denoised)
    ax[5].set_xticks([])
    ax[5].set_yticks([])
    ax[5].set_title('Denoised Img')
    ax[5].set(xlabel= str(round(PSNR,2)) + ' dB')

    plt.suptitle(f"sample:{typeName} noise:{noise_type} level:{noise_level}")
    # plt.savefig(f"{typeName}.jpg")
    plt.show()


    plt.plot(PSNRList)
    plt.xlabel("step")
    plt.ylabel("PSNR")
    plt.title(f"{typeName}-train")
    # plt.savefig(f"{typeName}_train.jpg")
    plt.show()