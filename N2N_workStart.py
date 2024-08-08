import os

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import cv2
import torch.optim as optim
from model import *
from utils import *

#========================== process ===============================================
def loadSingleImgByPath(filePath):
    img = cv2.imread(filePath)
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    img = cv2.resize(img,ConfigInfo.input_size)
    img = np.transpose(img,(2,0,1))
    img = img.reshape((1,*img.shape))
    return torch.Tensor(img)

#  添加噪声
def add_noise(x,noise_level,noise_type):
    if noise_type == 'gauss':
        noisy = x + torch.normal(0, noise_level/255, x.shape)
        noisy = torch.clamp(noisy,0,1)

    elif noise_type == 'poiss':
        noisy = torch.poisson(noise_level * x)/noise_level

    elif noise_type == "saltAndPepper":
        noisy = salt_and_pepper(x,noise_level)

    return noisy
#==============================================================================


# ============pair_downsampler============
# "orignal":0,
# "remoteWeight":1,
# "remoteNegativeWeight":2,
# "threeSizeSample":3,
# "stripe":4,
# "maxAndAvgPool":5,
# "avgPoolAndRandomPool":6,
# "maxPoolAndRandomPool":7,
# ========================================
def pair_downsampler_tools(img, sampleType=0, upSample=False):
    # img has shape B C H W
    c = img.shape[1]

    # 上采样
    if upSample:
        m = torch.nn.Upsample(scale_factor=2, mode='bilinear')
        img = m(img)

    if sampleType == ConfigInfo.downsamplerType["orignal"]:
        # original
        filter1 = torch.FloatTensor([[[[0, 0.5], [0.5, 0]]]]).to(img.device)
        filter1 = filter1.repeat(c, 1, 1, 1)

        filter2 = torch.FloatTensor([[[[0.5, 0], [0, 0.5]]]]).to(img.device)
        filter2 = filter2.repeat(c, 1, 1, 1)

        output1 = F.conv2d(img, filter1, stride=2, groups=c)
        output2 = F.conv2d(img, filter2, stride=2, groups=c)

    elif sampleType == ConfigInfo.downsamplerType["remoteWeight"]:
        # 边缘权重
        filter1 = torch.FloatTensor([[[[0.1, 0.5], [0.5, 0.1]]]]).to(img.device)
        filter1 = filter1.repeat(c, 1, 1, 1)

        filter2 = torch.FloatTensor([[[[0.5, 0.1], [0.1, 0.5]]]]).to(img.device)
        filter2 = filter2.repeat(c, 1, 1, 1)

        output1 = F.conv2d(img, filter1, stride=2, groups=c) / 1.2
        output2 = F.conv2d(img, filter2, stride=2, groups=c) / 1.2


    elif sampleType == ConfigInfo.downsamplerType["remoteNegativeWeight"]:
        # original  负边缘权重
        filter1 = torch.FloatTensor([[[[-0.1, 0.5], [0.5, -0.1]]]]).to(img.device)
        filter1 = filter1.repeat(c, 1, 1, 1)

        filter2 = torch.FloatTensor([[[[0.5, -0.1], [-0.1, 0.5]]]]).to(img.device)
        filter2 = filter2.repeat(c, 1, 1, 1)

        output1 = F.conv2d(img, filter1, stride=2, groups=c) / 0.8
        output2 = F.conv2d(img, filter2, stride=2, groups=c) / 0.8

    elif sampleType == ConfigInfo.downsamplerType["threeSizeSample"]:
        # 3X3 sample
        filter1 = torch.FloatTensor([[[[0, 0.5, 0], [0.5, 0.01, 0.5], [0, 0.5, 0]]]]).to(img.device)
        filter1 = filter1.repeat(c, 1, 1, 1)

        filter2 = torch.FloatTensor([[[[0.5, 0, 0.5], [0, 0.01, 0], [0.5, 0, 0.5]]]]).to(img.device)
        filter2 = filter2.repeat(c, 1, 1, 1)

        output1 = F.conv2d(img, filter1, stride=3, groups=c) / 2.01
        output2 = F.conv2d(img, filter2, stride=3, groups=c) / 2.01

    elif sampleType == ConfigInfo.downsamplerType["stripe"]:
        # 2X2 条纹
        filter1 = torch.FloatTensor([[[[0, 0.5], [0, 0.5]]]]).to(img.device)
        filter1 = filter1.repeat(c, 1, 1, 1)

        filter2 = torch.FloatTensor([[[[0.5, 0], [0.5, 0]]]]).to(img.device)
        filter2 = filter2.repeat(c, 1, 1, 1)

        output1 = F.conv2d(img, filter1, stride=2, groups=c)
        output2 = F.conv2d(img, filter2, stride=2, groups=c)

    elif sampleType == ConfigInfo.downsamplerType["maxAndAvgPool"]:
        # 池化
        output1 = F.max_pool2d(img, kernel_size=2, stride=2)
        output2 = F.avg_pool2d(img, kernel_size=2, stride=2)

    elif sampleType == ConfigInfo.downsamplerType["avgPoolAndRandomPool"]:
        # 随机池化
        output1 = F.avg_pool2d(img, kernel_size=2, stride=2)
        randomPooler = StochasticPool2DLayer(maxpool=False, training=True)
        output2 = randomPooler(img)

    elif sampleType == ConfigInfo.downsamplerType["maxPoolAndRandomPool"]:
        # 随机池化
        output1 = F.max_pool2d(img, kernel_size=2, stride=2)
        randomPooler = StochasticPool2DLayer(maxpool=False, training=True)
        output2 = randomPooler(img)

    elif sampleType == ConfigInfo.downsamplerType["RRandomPoolAndRandomPool"]:
        # 随机池化
        randomPooler1 = StochasticPool2DLayer(maxpool=False, training=True)
        output1 = randomPooler1(img)
        randomPooler2 = StochasticPool2DLayer(maxpool=False, training=True)
        output2 = randomPooler2(img)

    elif sampleType == ConfigInfo.downsamplerType["stripeReverse"]:
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

def pair_downsampler(img, sampleType=0, upSample=False,mini_epoch = 0):
    if ConfigInfo.use_step_filter :
        if  mini_epoch < 1000 :
            if mini_epoch != None and mini_epoch % 2 == 0:
                sampleType = ConfigInfo.downsamplerType["stripe"] # todo change

            if mini_epoch != None and mini_epoch % 3 == 0:
                sampleType = ConfigInfo.downsamplerType["stripeReverse"] # todo change

            if mini_epoch != None and mini_epoch % 5 == 0:
                sampleType = ConfigInfo.downsamplerType["remoteWeight"] # todo change

            if mini_epoch != None and mini_epoch % 10 == 0:
                sampleType = ConfigInfo.downsamplerType["remoteNegativeWeight"] # todo change threeSizeSamplePlus

            if mini_epoch != None and mini_epoch % 20 == 0:
                sampleType = ConfigInfo.downsamplerType["threeSizeSample"] # todo change remoteNegativeWeight


        if 500<= mini_epoch:
            return pair_downsampler_tools(img,ConfigInfo.downsamplerType["stripeReverse"], upSample)

    else:
        pass

    return pair_downsampler_tools(img,sampleType, upSample)

def mse(gt: torch.Tensor, pred:torch.Tensor)-> torch.Tensor:
    loss = torch.nn.MSELoss()
    return loss(gt,pred)

def loss_func(model,noisy_img,sampleType,upsampleFlag):
    noisy1, noisy2 = pair_downsampler(noisy_img,sampleType=sampleType,upSample=upsampleFlag,mini_epoch=ConfigInfo.now_epoch)
    pred1 =  noisy1 - model(noisy1)
    pred2 =  noisy2 - model(noisy2)


    loss_res = 1/2*(mse(noisy1,pred2)+mse(noisy2,pred1))

    noisy_denoised =  noisy_img - model(noisy_img)
    denoised1, denoised2 = pair_downsampler(noisy_denoised,sampleType=sampleType,upSample=upsampleFlag,mini_epoch=ConfigInfo.now_epoch)

    loss_cons=1/2*(mse(pred1,denoised1) + mse(pred2,denoised2))

    loss = loss_res + loss_cons

    return loss

def denoise(model, noisy_img):

    with torch.no_grad():
        pred = torch.clamp( noisy_img - model(noisy_img),0,1)

    return pred


#================================train and eval================================
def train( model,optimizer, noisy_img,sampleType,unsampleFlag):
  loss = loss_func(model,noisy_img,sampleType,unsampleFlag)
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
#===============================================================================



# N2N模型训练
def N2NTrain(dataPath,noise_level,noise_type,sampleType,unsampleFlag)->float:
    clean_img = loadSingleImgByPath(dataPath)/255.0
    noisy_img = add_noise(clean_img,noise_level, noise_type)
    model = network(ConfigInfo.channel).to(ConfigInfo.device)

    # # 下采样分析
    # img1, img2 = pair_downsampler(noisy_img)
    # img0 = noisy_img.cpu().squeeze(0).permute(1, 2, 0)
    # img1 = img1.cpu().squeeze(0).permute(1, 2, 0)
    # img2 = img2.cpu().squeeze(0).permute(1, 2, 0)

    optimizer = optim.Adam(model.parameters(), lr=ConfigInfo.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=ConfigInfo.step_size, gamma=ConfigInfo.gamma)

    # start
    PSNRList = []
    for epoch in range(ConfigInfo.max_epoch):
        model.train()
        # to device
        noisy_img = noisy_img.to(ConfigInfo.device)
        clean_img = clean_img.to(ConfigInfo.device)
        ConfigInfo.now_epoch = epoch
        loss_val = train(model,optimizer, noisy_img,sampleType,unsampleFlag)
        scheduler.step()
        if epoch % 100 == 0:
            model.eval()
            PSNR = test(model, noisy_img, clean_img)
            PSNRList.append(PSNR)
            # print(f"epoch:{epoch}: loss:{loss_val} PSNR:{PSNR}")

    finalPSNR = PSNRList[-1]
    return finalPSNR

# config
class ConfigInfo:
    datasetRoot = "./dataset"
    datasetUse = ['Kodak24','McMaster']
    logPath = "./log.txt"
    device = "cuda"
    channel = 3
    input_size = (256,256)
    max_epoch = 3000  # training epochs
    now_epoch = 0
    lr = 0.001  # learning rate 0.001
    step_size = 1000  # number of epochs at which learning rate decays
    gamma = 0.5  # factor by which learning rate decays
    use_step_filter = True # 阶段性采样

    downsamplerType = {
        "orignal": 0,
        "remoteWeight": 1,
        "remoteNegativeWeight": 2,
        "threeSizeSample": 3,
        "stripe": 4,
        "maxAndAvgPool": 5,
        "avgPoolAndRandomPool": 6,
        "maxPoolAndRandomPool": 7,
        "RRandomPoolAndRandomPool":8,
        "stripeReverse":9
    }

if __name__ == "__main__":
    noise_levels = [10,25,50]
    noise_types =  ['gauss','poiss']
    typeNames = ["avgPoolAndRandomPool"] #["orignal","avgPoolAndRandomPool","threeSizeSample","stripe"]

    unsampleFlag = False
    log = open(ConfigInfo.logPath,'a',encoding='utf-8')

    try:
        for noise_level in noise_levels:
            for noise_type in noise_types:
                for typeName in typeNames:
                    sampleType = ConfigInfo.downsamplerType[typeName]

                    for use_datasetPath in ConfigInfo.datasetUse:
                        log.writelines(f"noise_type:{noise_type} noise_level:{noise_level} dataset:{use_datasetPath} \n")
                        print(f"start train {use_datasetPath}...")

                        data_root = os.path.join(ConfigInfo.datasetRoot, use_datasetPath)
                        filePathList =  os.listdir(data_root)
                        PSNR_SUM = 0
                        for filePathIdx  in trange(len(filePathList)) :
                            finalPSNR = N2NTrain(os.path.join(data_root,filePathList[filePathIdx]),noise_level,noise_type,sampleType,unsampleFlag)
                            PSNR_SUM += finalPSNR
                            print(f"{use_datasetPath}---{filePathIdx}---finalPSNR:{finalPSNR}")

                        PSNR_AVG = PSNR_SUM/len(filePathList)
                        print("PSNR_AVG",PSNR_AVG)

                        log.writelines(f"sampleType:{typeName} --serial {sampleType} PSNR:{PSNR_AVG} \n")
                    log.writelines("\n\n")
        log.close()

    except KeyboardInterrupt:
        log.close()
