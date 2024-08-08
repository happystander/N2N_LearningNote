# 实验结果

## 1. PSNR：30.18

```python
if mini_epoch != None and mini_epoch % 2 == 0:
    sampleType = downsamplerType["remoteNegativeWeight"] # todo change

if mini_epoch != None and mini_epoch % 3 == 0:
    sampleType = downsamplerType["threeSizeSample"] # todo change

if mini_epoch != None and mini_epoch % 5 == 0:
    sampleType = downsamplerType["avgPoolAndRandomPool"] # todo change

if mini_epoch != None and mini_epoch % 10 == 0:
    sampleType = downsamplerType["stripe"] # todo change
```

## 



## 对比

| 噪声类型  | 噪声水平 |    数据集    |         样本类型          |          PSNR          |
| :-------: | :------: | :----------: | :-----------------------: | :--------------------: |
| **gauss** |  **10**  | **Kodak24**  |       **original**        | **31.765016103083884** |
| **gauss** |  **10**  | **McMaster** |       **original**        | **32.60741110854958**  |
|   gauss   |    10    |   Kodak24    |   avgPoolAndRandomPool    |   29.062866779997666   |
|   gauss   |    10    |   McMaster   |   avgPoolAndRandomPool    |   29.921417647989887   |
|   gauss   |    10    |   Kodak24    |      threeSizeSample      |   31.023153498242987   |
|   gauss   |    10    |   McMaster   |      threeSizeSample      |   31.381767704059364   |
|   gauss   |    10    |   Kodak24    |          stripe           |   29.39043638651313    |
|   gauss   |    10    |   McMaster   |          stripe           |   29.935700034499753   |
| **poiss** |  **10**  | **Kodak24**  |       **original**        | **24.63998252012159**  |
| **poiss** |  **10**  | **McMaster** |       **original**        |  **24.737993055765**   |
|   poiss   |    10    |   Kodak24    |   avgPoolAndRandomPool    |   23.888725505988486   |
|   poiss   |    10    |   McMaster   |   avgPoolAndRandomPool    |   24.383550687148443   |
|   poiss   |    10    |   Kodak24    |      threeSizeSample      |   23.37860894241406    |
|   poiss   |    10    |   McMaster   |      threeSizeSample      |   23.209275436613442   |
|   poiss   |    10    |   Kodak24    |          stripe           |   24.380419204099425   |
|   poiss   |    10    |   McMaster   |          stripe           |   24.526338380007147   |
| **gauss** |  **25**  | **Kodak24**  |       **original**        | **28.201306032300106** |
| **gauss** |  **25**  | **McMaster** |       **original**        | **27.71969076892901**  |
|   gauss   |    25    |   Kodak24    |   avgPoolAndRandomPool    |   26.61060509887395    |
|   gauss   |    25    |   McMaster   |   avgPoolAndRandomPool    |   26.646907153163863   |
|   gauss   |    25    |   Kodak24    |      threeSizeSample      |   27.060378680348276   |
|   gauss   |    25    |   McMaster   |      threeSizeSample      |   26.503910908316286   |
|   gauss   |    25    |   Kodak24    |          stripe           |   27.137841081186462   |
|   gauss   |    25    |   McMaster   |          stripe           |   26.895122695039102   |
| **poiss** |  **25**  | **Kodak24**  |       **original**        | **27.11037150793318**  |
| **poiss** |  **25**  | **McMaster** |       **original**        | **27.282469249270306** |
|   poiss   |    25    |   Kodak24    |   avgPoolAndRandomPool    |   25.906800015478606   |
|   poiss   |    25    |   McMaster   |   avgPoolAndRandomPool    |   26.47041069752629    |
|   poiss   |    25    |   Kodak24    |      threeSizeSample      |   25.821359365576324   |
|   poiss   |    25    |   McMaster   |      threeSizeSample      |   25.837963428284862   |
|   poiss   |    25    |   Kodak24    |          stripe           |   26.354139115186218   |
|   poiss   |    25    |   McMaster   |          stripe           |   26.72410542495548    |
| g**auss** |  **50**  | **Kodak24**  |       **original**        | **24.265330529078735** |
| **gauss** |  **50**  | **McMaster** |       **original**        | **23.226792004393147** |
|   gauss   |    50    |   Kodak24    |   avgPoolAndRandomPool    |   23.548712049104854   |
|   gauss   |    50    |   McMaster   |   avgPoolAndRandomPool    |    22.8614852152167    |
|   gauss   |    50    |   Kodak24    |      threeSizeSample      |   23.300704240651587   |
|   gauss   |    50    |   McMaster   |      threeSizeSample      |   22.33870254637075    |
|   gauss   |    50    |   Kodak24    |          stripe           |   23.95362615967436    |
|   gauss   |    50    |   McMaster   |          stripe           |   22.99623858221802    |
| **poiss** |  **50**  | **Kodak24**  |       **original**        | **28.780003063288405** |
| **poiss** |  **50**  | **McMaster** |       **original**        | **29.136332781031577** |
|   poiss   |    50    |   Kodak24    |   avgPoolAndRandomPool    |   27.02767205335897    |
|   poiss   |    50    |   McMaster   |   avgPoolAndRandomPool    |   27.80759388584945    |
|   poiss   |    50    |   Kodak24    |      threeSizeSample      |   27.612206425625413   |
|   poiss   |    50    |   McMaster   |      threeSizeSample      |   27.769896943846746   |
|   poiss   |    50    |   Kodak24    |          stripe           |   27.573180706058096   |
|   poiss   |    50    |   McMaster   |          stripe           |   28.067646150254042   |
|   gauss   |    10    |   Kodak24    | avgPoolAndRandomPool_step |   29.017261190400706   |
|   gauss   |    10    |   McMaster   | avgPoolAndRandomPool_step |   30.420419767049097   |
|   poiss   |    10    |   Kodak24    | avgPoolAndRandomPool_step |   24.51742658752416    |
|   poiss   |    10    |   McMaster   | avgPoolAndRandomPool_step |   24.73299586295389    |
|   gauss   |    25    |   Kodak24    | avgPoolAndRandomPool_step |   27.084612059945158   |
|   gauss   |    25    |   McMaster   | avgPoolAndRandomPool_step |   27.09919699566835    |
|   poiss   |    25    |   Kodak24    | avgPoolAndRandomPool_step |   26.400508687404947   |
|   poiss   |    25    |   McMaster   | avgPoolAndRandomPool_step |   26.89296667022657    |
|   gauss   |    50    |   Kodak24    | avgPoolAndRandomPool_step |   23.997400063797617   |
|   gauss   |    50    |   McMaster   | avgPoolAndRandomPool_step |   23.078842209909798   |
|   poiss   |    50    |   Kodak24    | avgPoolAndRandomPool_step |   27.55859254637116    |
|   poiss   |    50    |   McMaster   | avgPoolAndRandomPool_step |   28.375802548241307   |

| **gauss** | **10** | **Kodak24**  | **original** | **31.765016103083884** |
| :-------: | :----: | :----------: | :----------: | :--------------------: |
| **gauss** | **10** | **McMaster** | **original** | **32.60741110854958**  |

| **poiss** | **10** | **Kodak24**  | **original** | **24.63998252012159** |
| :-------: | :----: | :----------: | :----------: | :-------------------: |
| **poiss** | **10** | **McMaster** | **original** |  **24.737993055765**  |

| **gauss** | **25** | **Kodak24**  | **original** | **28.201306032300106** |
| :-------: | :----: | :----------: | :----------: | :--------------------: |
| **gauss** | **25** | **McMaster** | **original** | **27.71969076892901**  |

| **poiss** | **25** | **Kodak24**  | **original** | **27.11037150793318**  |
| :-------: | :----: | :----------: | :----------: | :--------------------: |
| **poiss** | **25** | **McMaster** | **original** | **27.282469249270306** |

| g**auss** | **50** | **Kodak24**  | **original** | **24.265330529078735** |
| :-------: | :----: | :----------: | :----------: | :--------------------: |
| **gauss** | **50** | **McMaster** | **original** | **23.226792004393147** |

| **poiss** | **50** | **Kodak24**  | **original** | **28.780003063288405** |
| :-------: | :----: | :----------: | :----------: | :--------------------: |
| **poiss** | **50** | **McMaster** | **original** | **29.136332781031577** |

# 噪声水平与PSNR值对比表

## Gauss噪声类型

### 噪声水平 10
| **数据集**   | **样本类型**              | **PSNR值（dB）** |
| ------------ | ------------------------- | ---------------- |
| **Kodak24**  | **original**              | **31.765**       |
| **McMaster** | **original**              | **32.607**       |
| Kodak24      | avgPoolAndRandomPool      | 29.063           |
| McMaster     | avgPoolAndRandomPool      | 29.921           |
| Kodak24      | threeSizeSample           | 31.023           |
| McMaster     | threeSizeSample           | 31.382           |
| Kodak24      | stripe                    | 29.390           |
| McMaster     | stripe                    | 29.936           |
| Kodak24      | avgPoolAndRandomPool_step | 29.017           |
| McMaster     | avgPoolAndRandomPool_step | 30.420           |

### 噪声水平 25
| **数据集**   | **样本类型**              | **PSNR值（dB）** |
| ------------ | ------------------------- | ---------------- |
| **Kodak24**  | **original**              | **28.201**       |
| **McMaster** | **original**              | **27.720**       |
| Kodak24      | avgPoolAndRandomPool      | 26.611           |
| McMaster     | avgPoolAndRandomPool      | 26.647           |
| Kodak24      | threeSizeSample           | 27.060           |
| McMaster     | threeSizeSample           | 26.504           |
| Kodak24      | stripe                    | 27.138           |
| McMaster     | stripe                    | 26.895           |
| Kodak24      | avgPoolAndRandomPool_step | 27.085           |
| McMaster     | avgPoolAndRandomPool_step | 27.099           |

### 噪声水平 50
| 数据集       | 样本类型                  | PSNR值（dB） |
| ------------ | ------------------------- | ------------ |
| **Kodak24**  | **original**              | **24.265**   |
| **McMaster** | **original**              | **23.227**   |
| Kodak24      | avgPoolAndRandomPool      | 23.549       |
| McMaster     | avgPoolAndRandomPool      | 22.861       |
| Kodak24      | threeSizeSample           | 23.301       |
| McMaster     | threeSizeSample           | 22.339       |
| Kodak24      | stripe                    | 23.954       |
| McMaster     | stripe                    | 22.996       |
| Kodak24      | avgPoolAndRandomPool_step | 23.997       |
| McMaster     | avgPoolAndRandomPool_step | 23.079       |

## Poiss噪声类型

### 噪声水平 10
| 数据集       | 样本类型                  | PSNR值（dB） |
| ------------ | ------------------------- | ------------ |
| **Kodak24**  | **original**              | **24.640**   |
| **McMaster** | **original**              | **24.738**   |
| Kodak24      | avgPoolAndRandomPool      | 23.889       |
| McMaster     | avgPoolAndRandomPool      | 24.384       |
| Kodak24      | threeSizeSample           | 23.379       |
| McMaster     | threeSizeSample           | 23.209       |
| Kodak24      | stripe                    | 24.380       |
| McMaster     | stripe                    | 24.526       |
| Kodak24      | avgPoolAndRandomPool_step | 24.517       |
| McMaster     | avgPoolAndRandomPool_step | 24.733       |

### 噪声水平 25
| 数据集       | 样本类型                  | PSNR值（dB） |
| ------------ | ------------------------- | ------------ |
| **Kodak24**  | **original**              | **27.110**   |
| **McMaster** | **original**              | **27.282**   |
| Kodak24      | avgPoolAndRandomPool      | 25.907       |
| McMaster     | avgPoolAndRandomPool      | 26.470       |
| Kodak24      | threeSizeSample           | 25.821       |
| McMaster     | threeSizeSample           | 25.838       |
| Kodak24      | stripe                    | 26.354       |
| McMaster     | stripe                    | 26.724       |
| Kodak24      | avgPoolAndRandomPool_step | 26.401       |
| McMaster     | avgPoolAndRandomPool_step | 26.893       |

### 噪声水平 50
| 数据集       | 样本类型                  | PSNR值（dB） |
| ------------ | ------------------------- | ------------ |
| **Kodak24**  | **original**              | **28.780**   |
| **McMaster** | **original**              | **29.136**   |
| Kodak24      | avgPoolAndRandomPool      | 27.028       |
| McMaster     | avgPoolAndRandomPool      | 27.808       |
| Kodak24      | threeSizeSample           | 27.612       |
| McMaster     | threeSizeSample           | 27.770       |
| Kodak24      | stripe                    | 27.573       |
| McMaster     | stripe                    | 28.068       |
| Kodak24      | avgPoolAndRandomPool_step | 27.559       |
| McMaster     | avgPoolAndRandomPool_step | 28.376       |