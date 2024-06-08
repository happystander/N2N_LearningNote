import torch
import torch.nn as nn
import torch.nn.functional as F

class PartialPooling2d(nn.Module):
    def __init__(self, kernel_size=2, stride=2,delta = 0):
        super(PartialPooling2d, self).__init__()
        self.kernel_size = kernel_size
        self.delta = delta
        self.stride = stride

    def forward(self, x):
        # 计算输出的尺寸
        h, w = x.shape[-2:]
        new_h = (h - self.kernel_size) // self.stride + 1
        new_w = (w - self.kernel_size) // self.stride + 1

        # 初始化输出张量
        output = torch.zeros_like(x)

        # 应用部分池化
        for i in range(new_h):
            for j in range(new_w):
                # 计算当前池化区域的索引
                start_h, end_h = i * self.stride, i * self.stride + self.kernel_size
                start_w, end_w = j * self.stride, j * self.stride + self.kernel_size

                # 选择当前区域
                pool_region = x[:, :, start_h:end_h, start_w:end_w]

                # 应用最大池化和最小池化
                max_pool = F.max_pool2d(pool_region, kernel_size=self.kernel_size, stride=self.stride)
                min_pool = F.avg_pool2d(pool_region, kernel_size=self.kernel_size, stride=self.stride)
                
                # 选择使用最大池化或最小池化的结果
                # 这里我们简单地交替使用最大池化和最小池化
                if (i + j + self.delta) % 2 == 0:
                    output[:, :, i, j] = max_pool[:, :, 0, 0]
                else:
                    output[:, :, i, j] = min_pool[:, :, 0, 0]

        return output

class StochasticPool2DLayer(nn.Module):
    def __init__(self, pool_size=2, maxpool=True, training=False, grid_size=None, **kwargs):
        super(StochasticPool2DLayer, self).__init__(**kwargs)
        self.rng = torch.cuda.manual_seed_all(123) # this changed in Pytorch for working
        self.pool_size = pool_size
        self.maxpool_flag = maxpool
        self.training = training
        if grid_size:
            self.grid_size = grid_size
        else:
            self.grid_size = pool_size
 
        self.Maxpool = torch.nn.MaxPool2d(kernel_size=self.pool_size, stride=2)
        self.Avgpool = torch.nn.AvgPool2d(kernel_size=self.pool_size,
                                          stride=self.pool_size,
                                          padding=self.pool_size//2,)
        self.padding = nn.ConstantPad2d((0,1,0,1),0)
 
    def forward(self, x, training=False, **kwargs):
        if self.maxpool_flag:
            x = self.Maxpool(x)
            x = self.padding(x)
        if not self.training:
            x = self.Avgpool(x)
            return x
    
        else:
            w, h = x.data.shape[2:]
            n_w, n_h = w//self.grid_size, h//self.grid_size
            n_sample_per_grid = self.grid_size//self.pool_size

            idx_w = []
            idx_h = []
            if w>2 and h>2:
                for i in range(n_w):
                    offset = self.grid_size * i
                    if i < n_w - 1:
                        this_n = self.grid_size
                    else:
                        this_n = x.data.shape[2] - offset
                    
                    this_idx, _ = torch.sort(torch.randperm(this_n)[:n_sample_per_grid])
                    idx_w.append(offset + this_idx)
                for i in range(n_h):
                    offset = self.grid_size * i
                    if i < n_h - 1:
                        this_n = self.grid_size
                    else:
                        this_n = x.data.shape[3] - offset
                    this_idx, _ = torch.sort(torch.randperm(this_n)[:n_sample_per_grid])
 
                    idx_h.append(offset + this_idx)
                idx_w = torch.cat(idx_w, dim=0)
                idx_h = torch.cat(idx_h, dim=0)
            else:
                idx_w = torch.LongTensor([0])
                idx_h = torch.LongTensor([0])
 
            output = x[:, :, idx_w.cuda()][:, :, :, idx_h.cuda()]
            return output



def salt_and_pepper(input,prob):
    noise_tensor=torch.rand_like(input)
    salt=torch.max(input)
    pepper=torch.min(input)
    input[noise_tensor<prob/2]=salt
    input[noise_tensor>1-prob/2]=pepper
    return input

