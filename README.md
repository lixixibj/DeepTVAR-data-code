# DeepTVAR: Deep Learning for a Time-Varying VAR Model with Extension to Integrated VAR (Li and Yuan, 2023)
## Introduction
We propose a new approach called DeepTVAR that employs deep learning methodology for vector autoregressive (VAR) modeling and prediction with time-varying parameters. A Long Short-Term Memory (LSTM) network is used for this purpose. To ensure the stability of the model, we enforce the causality condition on the autoregressive coefficients using the transformation of Ansley & Kohn (1986).

Authors
-------

-   [Xixi Li](https://lixixibj.github.io/)
-   [Jingsong Yuan](https://www.research.manchester.ac.uk/portal/jingsong.yuan.html)
- For any questions about this project, please feel free to contact Xixi Li via email at xixi.li@manchester.ac.uk.

## Project structure
This repository contains code and data used to reproduce results in the simulation study and real data applications. The structure of this project is as follows:
```
  ├── benchmarks-code-data 
    ├── DeepAR                   # DeepAR model
    ├── DeepState                    #Deep State space model
    ├── QBLL                     # Kernel based time-varying VAR model                  
    └── VAR               # Standard time-invariant VAR model
  ├── real-data-forecast-res #Forecasting results from the DeepTVAR model
  ├── simulation-res                   # Simulation results from DeepTVAR model
  ├── _main_for_para_estimation.py                  # Main code for parameter estimation in a simulation study
  ├── quick_plot_simu_res.py    # Code for quickly plotting simulation results based on 100 simulation runs
  ├── lstm_network.py                     # Set up an LSTM network to generate time-varying VAR parameters                  
  ├──custom_loss_float.py               #Evaluate log-likelihood function.
  ├── _model_fitting_for_real_data.py                     #  Model fitting for real data                  
  └── _main_make_predictions_for_real_data.py               #  Make predictions using the fitted model

  
 ```


## Preliminaries
All the computations were executed on an Intel Core i9 2.3 GHz processor with eight cores using a Macbook Pro.

All Python code was implemented using 
[![Python v3.6.15](https://img.shields.io/badge/python-v3.6.15-blue.svg)](https://www.python.org/downloads/release/python-3615/), and Pytorch was used for network training.

To reproduce all results, installation in a virtual environment as follows is essential:
```
#install python with version 3.6.15
conda create --name python36 python=3.6.15
conda activate python36
pip install --upgrade typing-extensions
#install pytorch with version 1.10.2
pip install torch==1.10.2 torchvision==0.11.3 torchaudio==0.10.2 -f https://download.pytorch.org/whl/cpu/torch_stable.html
```

Ps: Using Python 3.7 or 3.8 should work fine; however, having the correct version of PyTorch is essential.

The additional installation of other packages with specific versions can be implemented using
```
pip install pandas==1.1.5 
pip install packaging==21.3 
pip install matplotlib==3.3.4
```
## DeepTVAR
#### Simulation study
The following code will quickly plot simulation results based on pre-trained models from 100 simulation runs
```
python quick_plot_simu_res.py
```
The plots for estimated time-varying coefficients, variances, and covariances of innovations will be saved in the folder `simulation-res/estimated-A-mean/` and `simulation-res/estimated-var-cov-mean/` respectively.

The following code will do parameter estimation from scratch using the DeepTVAR model on 100 simulated two-dimensional time-varying VAR(2) processes.
```
python _main_for_para_estimation.py
```
The training loss function values, estimated time-varying coefficients, variances, covariances of innovations, and the trained-model file for each simulation will be saved in the folder `simulation-res/res/`.
#### Real data application
The following Python code will make predictions from 20 training samples using the DeepTVAR model
```
python _main_make_predictions_for_real_data.py
```
The output of forecasting accuracies in terms of MSE, MAPE, MIS and MSIS at h=1,...,12 is 
```
APE-ts1
ts
1
mse:h1-12
[0.00900153 0.02237251 0.02959229 0.04480644 0.06332411 0.07745956
 0.07877966 0.08027546 0.08192823 0.0792874  0.07560444 0.08290458]
average over h=1-6
0.04109274043278943
average over h=1-12
0.060444684399348005
mape:h1-12
[2.06657537 3.11050785 3.35045878 4.33397638 5.38746692 6.48633455
 5.96878652 6.06106587 6.78668341 6.40718018 5.99535072 6.59586649]
average over h=1-6
4.122553310574613
average over h=1-12
5.212521087216844
mis:h1-12
[0.43595915 0.67471566 0.8589389  0.96103086 1.0386559  1.14187448
 1.2362146  1.32437186 1.40783185 1.48732936 1.56342886 1.63659755]
average over h=1-6
0.851862494505816
average over h=1-12
1.147245753825519
msis:h1-12
[2.59375447 4.02813585 5.15607252 5.75625465 6.19658971 6.81152811
 7.37346898 7.89890596 8.39646239 8.87048447 9.32432784 9.76074905]
average over h=1-6
5.090389217681467
average over h=1-12
6.84722783291874
ts
2
mse:h1-12
[0.00418722 0.01097218 0.01769698 0.03080672 0.04692746 0.06883468
 0.09326944 0.12134203 0.14969652 0.17266855 0.18209303 0.19217234]
average over h=1-6
0.029904208933105508
average over h=1-12
0.09088893027383053
mape:h1-12
[ 1.66076573  2.82117902  3.57524781  4.77831204  5.90864851  7.35713729
  8.66052176  9.98885112 11.5066797  12.61532686 13.21737882 14.08954965]
average over h=1-6
4.350215067869802
average over h=1-12
8.014966525794717
mis:h1-12
[0.30544463 0.51220125 0.63789574 0.74046336 0.8375356  0.9598472
 1.16269595 1.43834501 1.50542347 1.50204886 1.45393051 1.38573437]
average over h=1-6
0.6655646305373984
average over h=1-12
1.0367971628476906
msis:h1-12
[1.36672431 2.30089767 2.86513058 3.32832172 3.76768276 4.33295169
 5.27889308 6.56839767 6.87057566 6.83441957 6.58851371 6.24969474]
average over h=1-6
2.9936181203078895
average over h=1-12
4.696016929255358
ts
3
mse:h1-12
[0.00134301 0.00501425 0.00863531 0.01111455 0.01265431 0.01264197
 0.0119476  0.0111955  0.01082306 0.0111929  0.01175113 0.01279572]
average over h=1-6
0.008567233050153283
average over h=1-12
0.010092442495417343
mape:h1-12
[ 5.08221835  8.63390001 11.5869497  13.29719563 15.07291319 16.97991901
 16.43884764 16.10786248 15.14673353 15.66917843 16.26210824 16.02005602]
average over h=1-6
11.7755159791605
average over h=1-12
13.858156851746394
mis:h1-12
[0.18068875 0.47874828 0.64945775 0.70072931 0.58256078 0.47204087
 0.51307794 0.55149308 0.58779378 0.62249327 0.65590844 0.68828347]
average over h=1-6
0.5107042917038167
average over h=1-12
0.5569396435164106
msis:h1-12
[1.24542844 3.38579112 4.61276583 4.9721734  4.04111991 3.18858347
 3.46260696 3.71906189 3.96123821 4.19245691 4.41481965 4.62997914]
average over h=1-6
3.574310361730394
average over h=1-12
3.818835410692971

```
## Benchmark models
All the code and data for the implementations of benchmark models are in the folder `benchmarks-code-data/`. The structure is as follows:
```
  ├── benchmarks-code-data 
    ├── DeepAR                   # DeepAR model
    ├── DeepState                    #Deep State space model
    ├── QBLL                     # Kernel based time-varying VAR model                  
    └── VAR               # Standard time-invariant VAR model
 ```
#### 1. DeepAR
The following Python code will make predictions from 20 training samples using the DeepAR model
```
python DeepAR_EU_3_prices.py
```
The output of forecasting accuracies in terms of MSE, MAPE, MIS and MSIS at h=1,...,12 is 
```
ts
0
mse:h1-12
[0.01478207 0.02933676 0.03860885 0.04742568 0.06454721 0.09066734
 0.09552431 0.09500929 0.11458311 0.11248295 0.11108641 0.13006667]
h=1-6
0.047561319123137345
h=1-12
0.078676721502744
mape:h1-12
[2.59683493 3.7049264  4.0878875  4.80345847 5.75433894 6.24130485
 6.49677532 7.00583356 7.39178079 7.13352009 7.19312851 7.9911179 ]
h=1-6
4.531458514367196
h=1-12
5.866742272253213
mis:h1-12
[0.44618609 1.19016231 1.96627403 1.75084929 2.35236194 3.87239212
 3.98051215 3.50484487 3.7109312  3.85000085 3.45536034 3.78765183]
h1-6
1.9297042976071659
h1-12
2.822293919199019
msis:h1-12
[ 2.67477823  6.90327552 11.51545188 10.27973822 13.60109446 22.11619044
 22.89372946 19.97981933 21.05324902 21.62059582 19.3958144  21.17582428]
h1-6
11.181754792440385
h1-12
16.100796754498816
ts
1
mse:h1-12
[0.01258709 0.03640446 0.0663249  0.09901041 0.13628341 0.19764291
 0.24935267 0.28362867 0.34683041 0.37896206 0.3670643  0.37815194]
h=1-6
0.09137553070275682
h=1-12
0.21268693615963488
mape:h1-12
[ 3.11986307  5.1201962   6.81423685  8.41293115 10.15513334 12.02047688
 13.52476171 15.12485622 16.51720612 17.63502927 18.09300226 18.75361701]
h=1-6
7.607139580629099
h=1-12
12.107609173497492
mis:h1-12
[ 0.45574415  1.52798628  2.97089729  3.90500839  5.30134963  6.81570463
  7.98395776  9.28214829 10.95204756 11.57359701 12.29135348 12.8566989 ]
h1-6
3.4961150619048547
h1-12
7.15970778019913
msis:h1-12
[ 2.02981765  6.44486962 12.45481784 16.3158941  22.28153597 28.9526657
 33.99097799 39.93432194 47.38832358 50.46420253 53.86072967 56.48912483]
h1-6
14.746600146282887
h1-12
30.883940118610028
ts
2
mse:h1-12
[0.00270914 0.00792339 0.01093926 0.01210866 0.01574792 0.01810605
 0.01560543 0.01389181 0.01677901 0.01926638 0.01673407 0.01598474]
h=1-6
0.011255738820933862
h=1-12
0.013816321790877122
mape:h1-12
[ 7.55682361 12.2820395  14.25616815 14.40943932 16.52570441 18.82152136
 18.49189512 17.34852144 17.94187478 19.45182029 17.64657278 17.32034043]
h=1-6
13.975282724775433
h=1-12
16.004393431856005
mis:h1-12
[0.67676839 1.43970992 1.89362682 1.96378076 2.42654824 2.7280781
 2.37157923 2.2127926  2.69022371 2.88654641 2.65131108 2.48907554]
h1-6
1.8547520383745189
h1-12
2.202503399399744
msis:h1-12
[ 4.68743074 10.00210144 13.12830125 13.28721734 16.34607269 18.45022855
 15.82288878 14.96309774 18.10097846 19.21800015 17.80894152 16.72681759]
h1-6
12.650225334807617
h1-12
14.878506354506285
```
The corresponding forecasts will be saved in the folder `benchmarks-code-data/DeepAR/eu_3_prices/`.

#### 2. DeepState
The following Python code will make predictions from 20 training samples using the DeepState model
```
python DeepState_EU_3_prices.py.py
```
The output of forecasting accuracies in terms of MSE, MAPE, MIS and MSIS at h=1,...,12 is 
```
 ts
0
mse:h1-12
[0.05232268 0.05618714 0.04556269 0.06142202 0.06437436 0.06974694
 0.07239647 0.10288358 0.09484858 0.09998459 0.0675105  0.07495493]
h=1-6
0.058269306357523885
h=1-12
0.07184954007711654
mape:h1-12
[4.70622011 4.94439529 4.52749536 5.02032833 5.27090887 5.19975284
 5.56985693 7.02182275 7.16723393 7.26012648 5.83754784 6.29197332]
h=1-6
4.944850133743054
h=1-12
5.734805171164638
mis:h1-12
[2.01753837 2.93453021 2.74386782 3.35740175 2.97838152 3.16479007
 3.15564669 4.6174627  4.13369775 4.46388193 2.44143625 2.77162468]
h1-6
2.866084957808923
h1-12
3.2316883128513383
msis:h1-12
[11.95509049 17.70251604 16.83221608 20.50414192 17.84084586 18.90962771
 19.31965466 27.05327515 25.068537   27.25947328 14.29660966 16.72956774]
h1-6
17.290739683620174
h1-12
19.455962966342636
ts
1
mse:h1-12
[0.02725959 0.03404635 0.03235979 0.03703907 0.04238568 0.06284687
 0.07843651 0.12239128 0.15177959 0.1783047  0.16328959 0.14118344]
h=1-6
0.03932288967609554
h=1-12
0.08927687051252353
mape:h1-12
[ 4.21587707  5.23407306  4.93721024  4.78922417  5.46906707  7.10414786
  7.59430681 10.48442849 11.03692439 11.82926113 11.0386565  10.49031092]
h=1-6
5.2915999121234085
h=1-12
7.851957309414345
mis:h1-12
[1.44619745 2.19607307 1.91123338 2.53596657 2.66253038 3.70266073
 4.82941429 6.53122786 7.3011127  7.84166296 7.31719207 5.63760726]
h1-6
2.4091102641066726
h1-12
4.4927398935217715
msis:h1-12
[ 6.46386548  9.70208039  8.84539938 11.40705803 11.92651679 17.02481596
 22.40723283 29.76634796 34.04456766 36.95245358 33.80310415 26.05508257]
h1-6
10.894956005126085
h1-12
20.699877065662207
ts
2
mse:h1-12
[0.00338778 0.00793941 0.01069544 0.01245135 0.01236888 0.01203381
 0.01057973 0.01000394 0.00974984 0.01032889 0.01100701 0.00985043]
h=1-6
0.00981277850883277
h=1-12
0.010033042738822153
mape:h1-12
[ 7.87440008 11.1940157  13.79979436 14.88243571 14.71544408 14.19620922
 14.32685562 14.21206631 15.09373101 15.63630589 15.82288123 15.41665674]
h=1-6
12.777049859057797
h=1-12
13.930899662355765
mis:h1-12
[0.73795946 1.23369299 1.94305231 2.33797714 2.29969982 2.24878678
 2.16054329 2.0780627  2.02289343 2.23032208 1.99077027 1.80691467]
h1-6
1.8001947522343376
h1-12
1.924222912399209
msis:h1-12
[ 5.06785509  8.65592997 13.89485355 16.65479011 16.2383781  15.66246568
 14.8389786  14.03246016 13.53045935 14.94402934 13.06122754 11.78518762]
h1-6
12.695712084669111
h1-12
13.197217925330635
```
The corresponding forecasts will be saved in the folder `benchmarks-code-data/DeepState/eu_3_prices/`.


#### 3. QBLL
The following Matlab code from [Katerina's personal website](https://sites.google.com/site/katerinapetrovawebpage/research) will make predictions from 20 training samples using the QBLL model
```
QBLL_EU_3_prices.m
```
The output of forecasting accuracies in terms of MSE, MAPE, MIS and MSIS at h=1,...,12 is 
```
ts =

     1

se:h1-12

ans =

    0.0124    0.0209    0.0371    0.0544    0.1027    0.1616    0.1673    0.1276    0.1131    0.1083    0.1435    0.0846

ape:h1-12

ans =

    2.2117    2.8239    3.7744    4.8446    6.4409    8.5891    8.8321    6.7177    7.0750    6.8519    8.1766    6.3923

is:h1-12

ans =

    0.6615    1.2544    1.6697    1.9109    2.2547    2.4887    2.7778    3.2152    3.0861    3.3796    3.5277    3.7493

sis:h1-12

ans =

    3.9880    7.5568   10.1276   11.5375   13.6160   15.0187   16.7573   19.4776   18.4936   20.2217   21.1139   22.4504

ts

ts =

     2

se:h1-12

ans =

    0.0070    0.0212    0.0257    0.0472    0.0700    0.1174    0.1547    0.1567    0.1978    0.2086    0.2689    0.2110

ape:h1-12

ans =

    2.3284    3.8711    4.3964    6.2087    7.7082    9.6190   11.3972   11.0239   13.1596   13.6910   15.9470   14.2495

is:h1-12

ans =

    0.5046    1.0479    1.1280    1.3461    1.6676    1.8140    2.3449    2.5745    2.7104    2.5338    3.2091    2.7490

sis:h1-12

ans =

    2.2843    4.7043    5.1222    6.1240    7.5797    8.1997   10.7071   11.6194   12.3610   11.5253   14.7109   12.5571

ts

ts =

     3

se:h1-12

ans =

    0.0041    0.0084    0.0086    0.0117    0.0174    0.0217    0.0206    0.0192    0.0239    0.0241    0.0187    0.0239

ape:h1-12

ans =

    9.1634   12.7505   12.3338   15.3860   18.9394   21.4130   20.1462   20.4748   21.9487   23.8209   19.6872   22.2406

is:h1-12

ans =

    0.4458    0.6618    0.7851    0.9255    1.0553    1.3502    1.2701    1.4501    1.4717    1.5182    1.4987    1.5468

sis:h1-12

ans =

    3.0675    4.5197    5.4001    6.3604    7.2506    9.2505    8.6996    9.9286   10.0631   10.3650   10.2415   10.5976

>> 
```
The corresponding forecasts will be saved in the folder `benchmarks-code-data/QBLL/`.

#### 4. VAR
The following R code will make predictions from 20 training samples using a time-invariant VAR model
```
VAR_EU_3_prices.R
```
The output of forecasting accuracies in terms of MSE, MAPE, MIS and MSIS at h=1,...,12 is 
```
> 
> print('ts')
[1] "ts"
> print(1)
[1] 1
> print('mse')
[1] "mse"
> mse1=colMeans(mse.accuracy.m.ts1)
> print(colMeans(mse.accuracy.m.ts1))
 [1] 0.01188507 0.02476166 0.03784571 0.05755078 0.07707349 0.09246908 0.09361060 0.09680412
 [9] 0.10049559 0.09871302 0.09887104 0.11357693
> print('h1-6')
[1] "h1-6"
> print(mean(mse1[1:6]))
[1] 0.0502643
> print('h1-12')
[1] "h1-12"
> print(mean(mse1[1:12]))
[1] 0.07530476
> 
> print('mape')
[1] "mape"
> mape=colMeans(mape.accuracy.m.ts1)
> print('h1-6')
[1] "h1-6"
> print(mean(mape[1:6]))
[1] 4.676767
> print('h1-12')
[1] "h1-12"
> print(mean(mape[1:12]))
[1] 6.013428
> print(colMeans(mape.accuracy.m.ts1))
 [1] 2.335064 3.450244 4.023268 5.151567 6.123860 6.976601 6.786616 7.046278 7.620397 7.460051
[11] 7.308975 7.878212
> 
> print('msis')
[1] "msis"
> print(colMeans(msis.accuracy.m.ts1))
 [1] 2.599609 3.487750 5.371474 6.999001 6.582060 5.570876 6.815474 6.960009 6.604665 6.934745
[11] 7.249920 7.552000
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
> print('mis')
[1] "mis"
> print(colMeans(mis.accuracy.m.ts1))
 [1] 0.4369075 0.5865419 0.8899457 1.1541930 1.0958534 0.9379158 1.1428956 1.1685877 1.1124237
[10] 1.1680171 1.2211019 1.2719817
> mis1=colMeans(mis.accuracy.m.ts1)
> print('h1-6')
[1] "h1-6"
> print(mean(mis1[1:6]))
[1] 0.8502262
> print('h1-12')
[1] "h1-12"
> print(mean(mis1[1:12]))
[1] 1.01553
> 
> 
> 
> print('ts')
[1] "ts"
> print(2)
[1] 2
> print('mse')
[1] "mse"
> mse2=colMeans(mse.accuracy.m.ts2)
> print(colMeans(mse.accuracy.m.ts2))
 [1] 0.008700075 0.016415219 0.022462267 0.042160659 0.063698522 0.089182473 0.118664171
 [8] 0.152652720 0.184450797 0.210992731 0.222530212 0.236135318
> print('h1-6')
[1] "h1-6"
> print(mean(mse2[1:6]))
[1] 0.04043654
> print('h1-12')
[1] "h1-12"
> print(mean(mse2[1:12]))
[1] 0.1140038
> 
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
 [1]  2.380772  3.653288  4.035841  5.699330  6.973600  8.405994  9.785554 11.156846 12.880147
[10] 13.963337 14.553888 15.482259
> 
> print('msis')
[1] "msis"
> print(colMeans(msis.accuracy.m.ts2))
 [1]  2.283454  2.383109  3.349922  4.674728  7.051822 10.507793 13.180615 16.248076 18.791012
[10] 20.361912 18.807391 16.948318
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
> 
> print('mis')
[1] "mis"
> print(colMeans(mis.accuracy.m.ts2))
 [1] 0.5200200 0.5345973 0.7370087 1.0227053 1.5247733 2.2462955 2.8113842 3.4590339 3.9967360
[10] 4.3302429 4.0185606 3.6300224
> mis2=colMeans(mis.accuracy.m.ts2)
> print('h1-6')
[1] "h1-6"
> print(mean(mis2[1:6]))
[1] 1.097567
> print('h1-12')
[1] "h1-12"
> print(mean(mis2[1:12]))
[1] 2.402615
> 
> print('ts')
[1] "ts"
> print(3)
[1] 3
> print('mse')
[1] "mse"
> mse3=colMeans(mse.accuracy.m.ts3)
> print(colMeans(mse.accuracy.m.ts3))
 [1] 0.001372272 0.005291837 0.009146024 0.011936061 0.014256716 0.015641630 0.016314175
 [8] 0.016597048 0.016965454 0.017940767 0.019055213 0.020065467
> print('h1-6')
[1] "h1-6"
> print(mean(mse3[1:6]))
[1] 0.009607423
> print('h1-12')
[1] "h1-12"
> print(mean(mse3[1:12]))
[1] 0.01371522
> 
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
 [1]  4.957031  8.937322 11.485961 13.714464 16.436778 18.761090 19.144820 19.602825 18.998903
[10] 19.579763 21.070967 20.742929
> 
> print('msis')
[1] "msis"
> print(colMeans(msis.accuracy.m.ts3))
 [1] 1.181309 2.433108 3.719531 4.032329 3.917401 4.284679 4.651278 4.990386 5.307343 5.606094
[11] 5.889676 6.160322
> msis3=colMeans(msis.accuracy.m.ts3)
> print('h1-6')
[1] "h1-6"
> print(mean(msis3[1:6]))
[1] 3.261393
> print('h1-12')
[1] "h1-12"
> print(mean(msis3[1:12]))
[1] 4.347788
> 
> print('mis')
[1] "mis"
> print(colMeans(mis.accuracy.m.ts3))
 [1] 0.1743438 0.3550837 0.5377414 0.5893892 0.5814903 0.6369823 0.6917594 0.7423679 0.7896538
[10] 0.8342223 0.8765270 0.9168990
> mis3=colMeans(mis.accuracy.m.ts3)
> print('h1-6')
[1] "h1-6"
> print(mean(mis3[1:6]))
[1] 0.4791718
> print('h1-12')
[1] "h1-12"
> print(mean(mis3[1:12]))
[1] 0.6438717
```
The corresponding forecasts will be saved in the folder `benchmarks-code-data/VAR/`.

References
----------

- Xixi Li, Jingsong Yuan (2023).  DeepTVAR: Deep Learning for a Time-Varying VAR Model with Extension to Integrated VAR.  [Working paper]().



