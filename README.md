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
The following code will do parameter estimation on a simulated three-diemnsional VAR(2) procoess
```
python _main_for_para_estimation.py
```
The training loss function values, estimated time-varying coefficients, variances, covariances of innovations and pretrained-model file will be saved in the folder `simulation-res/res/`.
#### Real data application
The following code will make predictions from 20 training samples
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

References
----------

- Xixi Li, Jingsong Yuan (2022).  DeepTVAR: Deep Learning for a Time-varying VAR Model.  [Working paper]().



