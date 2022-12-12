# DeepTVAR: Deep Learning for a Time-varying VAR Model (Li and Yuan, 2022)
## Introduction
We propose a new approach called DeepTVAR that employs deep learning methodology for vector autoregressive (VAR) modeling and prediction with time-varying parameters. A Long Short-Term Memory (LSTM) network is used for this purpose. To ensure the stability of the model, we enforce the causality condition on the autoregressive coefficients using the transformation of Ansley & Kohn (1986). 

Authors
-------

-   [Xixi Li](https://lixixibj.github.io/)
-   [Jingsong Yuan](https://www.research.manchester.ac.uk/portal/jingsong.yuan.html)

## Project structure
This repository contains python code and data used to reproduce results in a simulation study and real data applications.

Here, we brifely introduce some important `.py` files in this project.

- `_main_for_para_estimation.py`: main code for parameter estimation in simulation study.
- `lstm_network.py`: set up an LSTM network to generate time-varying VAR parameters.
- `custom_loss_float.py`: evaluate log-likelihood function.
- `_model_fitting_for_real_data.py`: model fitting for real data.
- `_main_make_predictions_for_real_data.py`: make predictions using the fitted model.


## Preliminaries
All code was implemented using 
[![Python v3.6.15](https://img.shields.io/badge/python-v3.6.15-blue.svg)](https://www.python.org/downloads/release/python-3615/), and Pytorch was used for network training.

Installation in a virtual environment is recommended:
```
#install python with version 3.6.15
conda create --name python36 python=3.6.15
conda activate python36
#install pytorch with version 1.10.2
pip install torch==1.10.2 torchvision==0.11.3 torchaudio==0.10.2 -f https://download.pytorch.org/whl/cpu/torch_stable.html
```

The additonal installation of other packages with specific versions can be implemented using
```
pip install pandas==1.1.5 
pip install packaging==21.3 
pip install matplotlib==3.3.4
```
## Usage
#### Simulation study
The following code will do parameter estimation using DeepTVAR model on a simulated three-diemnsional VAR(2) procoess
```
python _main_for_para_estimation.py
```
The training loss function values, estimated time-varying coefficients, variances, covariances of innovations and pretrained-model file will be saved in the folder `simulation-res/res/`.
#### Real data application
The following Python code will make predictions from 20 training samples using DeepTVAR model
```
python _main_make_predictions_for_real_data.py
```
The output of forecasting accuracies in terms of APE and SIS at h=1,...,12 is 
```
APE-ts1
[2.06657537 3.11050785 3.35045878 4.33397638 5.38746692 6.48633455
 5.96878652 6.06106587 6.78668341 6.40718018 5.99535072 6.59586649]
APE-ts2
[ 1.66076573  2.82117902  3.57524781  4.77831204  5.90864851  7.35713729
  8.66052176  9.98885112 11.5066797  12.61532686 13.21737882 14.08954965]
APE-ts3
[ 5.08221835  8.63390001 11.5869497  13.29719563 15.07291319 16.97991901
 16.43884764 16.10786248 15.14673353 15.66917843 16.26210824 16.02005602]
SIS-ts1
[2.59375447 4.02813585 5.15607252 5.75625465 6.19658971 6.81152811
 7.37346898 7.89890596 8.39646239 8.87048447 9.32432784 9.76074905]
SIS-ts2
[1.36672431 2.30089767 2.86513058 3.32832172 3.76768276 4.33295169
 5.27889308 6.56839767 6.87057566 6.83441957 6.58851371 6.24969474]
SIS-ts3
[1.24542844 3.38579112 4.61276583 4.9721734  4.04111991 3.18858347
 3.46260696 3.71906189 3.96123821 4.19245691 4.41481965 4.62997914]

```
## Benchmark models
#### 1. DeepAR
The following Python code will make predictions from 20 training samples using DeepAR model
```
python DeepAR_EU_3_prices.py
```
The output of forecasting accuracies in terms of APE and SIS at h=1,...,12 is 
```
APE-ts1
[2.06657537 3.11050785 3.35045878 4.33397638 5.38746692 6.48633455
 5.96878652 6.06106587 6.78668341 6.40718018 5.99535072 6.59586649]
APE-ts2
[ 1.66076573  2.82117902  3.57524781  4.77831204  5.90864851  7.35713729
  8.66052176  9.98885112 11.5066797  12.61532686 13.21737882 14.08954965]
APE-ts3
[ 5.08221835  8.63390001 11.5869497  13.29719563 15.07291319 16.97991901
 16.43884764 16.10786248 15.14673353 15.66917843 16.26210824 16.02005602]
SIS-ts1
[2.59375447 4.02813585 5.15607252 5.75625465 6.19658971 6.81152811
 7.37346898 7.89890596 8.39646239 8.87048447 9.32432784 9.76074905]
SIS-ts2
[1.36672431 2.30089767 2.86513058 3.32832172 3.76768276 4.33295169
 5.27889308 6.56839767 6.87057566 6.83441957 6.58851371 6.24969474]
SIS-ts3
[1.24542844 3.38579112 4.61276583 4.9721734  4.04111991 3.18858347
 3.46260696 3.71906189 3.96123821 4.19245691 4.41481965 4.62997914]
```
The coresponding forecasts will be saved in the folder `benchmarks-code-data/DeepAR/eu_3_prices/`.

#### 2. DeepState
The following Python code will make predictions from 20 training samples using DeepState model
```
python DeepState_EU_3_prices.py.py
```
The output of forecasting accuracies in terms of APE and SIS at h=1,...,12 is 
```
APE-ts1
[2.06657537 3.11050785 3.35045878 4.33397638 5.38746692 6.48633455
 5.96878652 6.06106587 6.78668341 6.40718018 5.99535072 6.59586649]
APE-ts2
[ 1.66076573  2.82117902  3.57524781  4.77831204  5.90864851  7.35713729
  8.66052176  9.98885112 11.5066797  12.61532686 13.21737882 14.08954965]
APE-ts3
[ 5.08221835  8.63390001 11.5869497  13.29719563 15.07291319 16.97991901
 16.43884764 16.10786248 15.14673353 15.66917843 16.26210824 16.02005602]
SIS-ts1
[2.59375447 4.02813585 5.15607252 5.75625465 6.19658971 6.81152811
 7.37346898 7.89890596 8.39646239 8.87048447 9.32432784 9.76074905]
SIS-ts2
[1.36672431 2.30089767 2.86513058 3.32832172 3.76768276 4.33295169
 5.27889308 6.56839767 6.87057566 6.83441957 6.58851371 6.24969474]
SIS-ts3
[1.24542844 3.38579112 4.61276583 4.9721734  4.04111991 3.18858347
 3.46260696 3.71906189 3.96123821 4.19245691 4.41481965 4.62997914]
```
The coresponding forecasts will be saved in the folder `benchmarks-code-data/DeepState/eu_3_prices/`.


#### 3. QBLL
The following Matlab code will make predictions from 20 training samples using QBLL model
```
QBLL_EU_3_prices.m
```
The output of forecasting accuracies in terms of APE and SIS at h=1,...,12 is 
```
APE-ts1
[2.06657537 3.11050785 3.35045878 4.33397638 5.38746692 6.48633455
 5.96878652 6.06106587 6.78668341 6.40718018 5.99535072 6.59586649]
APE-ts2
[ 1.66076573  2.82117902  3.57524781  4.77831204  5.90864851  7.35713729
  8.66052176  9.98885112 11.5066797  12.61532686 13.21737882 14.08954965]
APE-ts3
[ 5.08221835  8.63390001 11.5869497  13.29719563 15.07291319 16.97991901
 16.43884764 16.10786248 15.14673353 15.66917843 16.26210824 16.02005602]
SIS-ts1
[2.59375447 4.02813585 5.15607252 5.75625465 6.19658971 6.81152811
 7.37346898 7.89890596 8.39646239 8.87048447 9.32432784 9.76074905]
SIS-ts2
[1.36672431 2.30089767 2.86513058 3.32832172 3.76768276 4.33295169
 5.27889308 6.56839767 6.87057566 6.83441957 6.58851371 6.24969474]
SIS-ts3
[1.24542844 3.38579112 4.61276583 4.9721734  4.04111991 3.18858347
 3.46260696 3.71906189 3.96123821 4.19245691 4.41481965 4.62997914]
```
The coresponding forecasts will be saved in the folder `benchmarks-code-data/QBLL/`.

#### 4. VAR
The following R code will make predictions from 20 training samples using time-invarient VAR model
```
VAR_EU_3_prices.R
```
The output of forecasting accuracies in terms of APE and SIS at h=1,...,12 is 
```
> print('ts')
[1] "ts"
> print(1)
[1] 1
> # print('mse')
> # print(colMeans(mse.accuracy.m.ts1))
> print('mape')
[1] "mape"
> mape=colMeans(mape.accuracy.m.ts1)
> print('h1-4')
[1] "h1-4"
> print(mean(mape[1:6]))
[1] 4.676767
> print('h1-12')
[1] "h1-12"
> print(mean(mape[1:12]))
[1] 6.013428
> print(colMeans(mape.accuracy.m.ts1))
 [1] 2.335064 3.450244 4.023268 5.151567 6.123860 6.976601 6.786616 7.046278 7.620397 7.460051 7.308975 7.878212
> print('msis')
[1] "msis"
> print(colMeans(msis.accuracy.m.ts1))
 [1] 2.599609 3.487750 5.371474 6.999001 6.582060 5.570876 6.815474 6.960009 6.604665 6.934745 7.249920 7.552000
> msis=colMeans(msis.accuracy.m.ts1)
> print('h1-6')
[1] "h1-6"
> print(mean(msis[1:6]))
[1] 5.101795
> print('h1-12')
[1] "h1-12"
> print(mean(msis[1:12]))
[1] 6.060632
> 
> 
> print('ts')
[1] "ts"
> print(2)
[1] 2
> # print('mse')
> # print(colMeans(mse.accuracy.m.ts1))
> print('mape')
[1] "mape"
> mape2=colMeans(mape.accuracy.m.ts2)
> print('h1-6')
[1] "h1-6"
> print(mean(mape2[1:6]))
[1] 5.191471
> print('h1-12')
[1] "h1-12"
> print(mean(mape2[1:12]))
[1] 9.080905
> print(colMeans(mape.accuracy.m.ts2))
 [1]  2.380772  3.653288  4.035841  5.699330  6.973600  8.405994  9.785554 11.156846 12.880147 13.963337 14.553888
[12] 15.482259
> print('msis')
[1] "msis"
> print(colMeans(msis.accuracy.m.ts2))
 [1]  2.283454  2.383109  3.349922  4.674728  7.051822 10.507793 13.180615 16.248076 18.791012 20.361912 18.807391
[12] 16.948318
> msis2=colMeans(msis.accuracy.m.ts2)
> print('h1-6')
[1] "h1-6"
> print(mean(msis2[1:6]))
[1] 5.041805
> print('h1-12')
[1] "h1-12"
> print(mean(msis2[1:12]))
[1] 11.21568
> 
> print('ts')
[1] "ts"
> print(3)
[1] 3
> # print('mse')
> # print(colMeans(mse.accuracy.m.ts1))
> print('mape')
[1] "mape"
> mape3=colMeans(mape.accuracy.m.ts3)
> print('h1-6')
[1] "h1-6"
> print(mean(mape3[1:6]))
[1] 12.38211
> print('h1-12')
[1] "h1-12"
> print(mean(mape3[1:12]))
[1] 16.1194
> print(colMeans(mape.accuracy.m.ts3))
 [1]  4.957031  8.937322 11.485961 13.714464 16.436778 18.761090 19.144820 19.602825 18.998903 19.579763 21.070967
[12] 20.742929
> print('msis')
[1] "msis"
> print(colMeans(msis.accuracy.m.ts3))
 [1] 1.181309 2.433108 3.719531 4.032329 3.917401 4.284679 4.651278 4.990386 5.307343 5.606094 5.889676 6.160322
> msis3=colMeans(msis.accuracy.m.ts3)
> print('h1-6')
[1] "h1-6"
> print(mean(msis3[1:6]))
[1] 3.261393
> print('h1-12')
[1] "h1-12"
> print(mean(msis3[1:12]))
[1] 4.347788
```
The coresponding forecasts will be saved in the folder `benchmarks-code-data/VAR/`.

References
----------

- Xixi Li, Jingsong Yuan (2022).  DeepTVAR: Deep Learning for a Time-varying VAR Model.  [Working paper]().



