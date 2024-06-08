import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model import *
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
def pair_downsampler(img,sampleType = 0,upSample = False):
    #img has shape B C H W
    c = img.shape[1]

    # 上采样
    if upSample:
        m = torch.nn.Upsample(scale_factor=2, mode='bilinear')
        img = m(img)

    if sampleType == downsamplerType["orignal"]:
        # original
        filter1 = torch.FloatTensor([[[[0 ,0.5],[0.5, 0]]]]).to(img.device)
        filter1 = filter1.repeat(c,1, 1, 1)

        filter2 = torch.FloatTensor([[[[0.5 ,0],[0, 0.5]]]]).to(img.device)
        filter2 = filter2.repeat(c,1, 1, 1)

        output1 = F.conv2d(img, filter1, stride=2, groups=c)
        output2 = F.conv2d(img, filter2, stride=2, groups=c)

    elif sampleType == downsamplerType["remoteWeight"]:
        #边缘权重
        filter1 = torch.FloatTensor([[[[0.1 ,0.5],[0.5, 0.1]]]]).to(img.device)
        filter1 = filter1.repeat(c,1, 1, 1)

        filter2 = torch.FloatTensor([[[[0.5 ,0.1],[0.1, 0.5]]]]).to(img.device)
        filter2 = filter2.repeat(c,1, 1, 1)

        output1 = F.conv2d(img, filter1, stride=2, groups=c)/1.2
        output2 = F.conv2d(img, filter2, stride=2, groups=c)/1.2


    elif sampleType == downsamplerType["remoteNegativeWeight"]:
        # original  负边缘权重
        filter1 = torch.FloatTensor([[[[-0.1 ,0.5],[0.5, -0.1]]]]).to(img.device)
        filter1 = filter1.repeat(c,1, 1, 1)

        filter2 = torch.FloatTensor([[[[0.5 ,-0.1],[-0.1, 0.5]]]]).to(img.device)
        filter2 = filter2.repeat(c,1, 1, 1)

        output1 = F.conv2d(img, filter1, stride=2, groups=c)/0.8
        output2 = F.conv2d(img, filter2, stride=2, groups=c)/0.8

    elif sampleType == downsamplerType["threeSizeSample"]:
        #3X3 sample
        filter1 = torch.FloatTensor([[[[0 ,0.5,0],[0.5, 0.01,0.5],[0 ,0.5,0]]]]).to(img.device)
        filter1 = filter1.repeat(c,1, 1, 1)

        filter2 = torch.FloatTensor([[[[0.5 ,0,0.5],[0, 0.01,0],[0.5 ,0,0.5]]]]).to(img.device)
        filter2 = filter2.repeat(c,1, 1, 1)

        output1 = F.conv2d(img, filter1, stride=3, groups=c)/2.01
        output2 = F.conv2d(img, filter2, stride=3, groups=c)/2.01
    
    elif sampleType == downsamplerType["stripe"]:
        # 2X2 条纹
        filter1 = torch.FloatTensor([[[[0 ,0.5],[0, 0.5]]]]).to(img.device)
        filter1 = filter1.repeat(c,1, 1, 1)

        filter2 = torch.FloatTensor([[[[0.5 ,0],[0.5, 0]]]]).to(img.device)
        filter2 = filter2.repeat(c,1, 1, 1)

        output1 = F.conv2d(img, filter1, stride=2, groups=c)
        output2 = F.conv2d(img, filter2, stride=2, groups=c)

    elif sampleType == downsamplerType["maxAndAvgPool"]:
        # 池化
        output1 = F.max_pool2d(img,kernel_size=2, stride=2)
        output2 = F.avg_pool2d(img,kernel_size=2, stride=2)

    elif sampleType == downsamplerType["avgPoolAndRandomPool"]:
        # 随机池化
        output1 = F.avg_pool2d(img,kernel_size=2, stride=2)
        randomPooler =  StochasticPool2DLayer(maxpool=False, training=True)
        output2 = randomPooler(img)

    elif sampleType == downsamplerType["maxPoolAndRandomPool"]:
        # 随机池化
        output1 = F.max_pool2d(img,kernel_size=2, stride=2)
        randomPooler =  StochasticPool2DLayer(maxpool=False, training=True)
        output2 = randomPooler(img)

    elif sampleType == downsamplerType["RRandomPoolAndRandomPool"]:
        # 随机池化
        randomPooler1 =  StochasticPool2DLayer(maxpool=False, training=True)
        output1 = randomPooler1(img)
        randomPooler2 =  StochasticPool2DLayer(maxpool=False, training=True)
        output2 = randomPooler2(img)

    else:
        raise f"None {sampleType} type..."
    return output1, output2

def mse(gt: torch.Tensor, pred:torch.Tensor)-> torch.Tensor:
    loss = torch.nn.MSELoss()
    return loss(gt,pred)

def loss_func(noisy_img):
    noisy1, noisy2 = pair_downsampler(noisy_img,sampleType=global_sampleType,upSample=global_upsampleFlag)
    pred1 =  noisy1 - model(noisy1)
    pred2 =  noisy2 - model(noisy2)


    loss_res = 1/2*(mse(noisy1,pred2)+mse(noisy2,pred1))

    noisy_denoised =  noisy_img - model(noisy_img)
    denoised1, denoised2 = pair_downsampler(noisy_denoised,sampleType=global_sampleType,upSample=global_upsampleFlag)

    loss_cons=1/2*(mse(pred1,denoised1) + mse(pred2,denoised2))

    loss = loss_res + loss_cons

    return loss

def train( optimizer, noisy_img):

  loss = loss_func(noisy_img)

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
    for epoch in trange(max_epoch):
        model.train()
        loss_val = train(optimizer, noisy_img)
        scheduler.step()
        if epoch % 100 == 0:
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
    "orignal":0,
    "remoteWeight":1,
    "remoteNegativeWeight":2,
    "threeSizeSample":3,
    "stripe":4,
    "maxAndAvgPool":5,
    "avgPoolAndRandomPool":6,
    "maxPoolAndRandomPool":7,
}

# 添加噪声
noise_type = 'gauss' # Either 'gauss' or 'poiss'
noise_level = 25    # Pixel range is 0-255 for Gaussian, and 0-1 for Poission
typeName = "avgPoolAndRandomPool"
global_sampleType = downsamplerType[typeName]
global_upsampleFlag = False

if __name__ == '__main__':

    # response = requests.get(url,proxies=proxies)
    fp = open("00023.00023",'rb')
    data = fp.read()
    path=BytesIO(data)
    clean_img = torch.load(path).unsqueeze(0)
    print(clean_img.shape) #B C H W

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
    img1, img2 = pair_downsampler(noisy_img)

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