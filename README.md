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

Ps: If the installation of python 3.6 fails, using Python 3.7 or 3.8 should work fine; however, having the correct version of PyTorch is essential.

The additional installation of other packages with specific versions can be implemented using
```
pip install pandas==1.1.5 
pip install packaging==21.3 
pip install matplotlib==3.3.4
#for deepar and deepstate
pip install mxnet==1.7.0.post2 gluonts==0.8.1 numpy==1.19.5 pandas
```
## DeepTVAR
#### Simulation study
The following code will quickly plot simulation results based on pre-trained models from 100 simulation runs
```
python quick_plot_simu_res.py
```
The plots for estimated time-varying coefficients, variances, and covariances of innovations will be saved in the folder `simulation-res/estimated-A-mean/` and `simulation-res/estimated-var-cov-mean/` respectively.

The following code will do parameter estimation from scratch using the DeepTVAR model on 100 simulated two-dimensional time-varying VAR(2) processes. The running time for each set of simulation studies is approximately 10 minutes.
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
ts
1
mse:h1-12
[0.00868982 0.02210492 0.03052253 0.04534604 0.06313274 0.07779477
 0.07928851 0.08109754 0.08267712 0.07963193 0.07632775 0.08333542]
average over h=1-6
0.04126513846170169
average over h=1-12
0.06082909191330655
mape:h1-12
[2.02572073 3.08722468 3.36289759 4.33282201 5.36286772 6.47490984
 5.96670786 6.11772115 6.84065188 6.40654087 6.05647572 6.64271551]
average over h=1-6
4.1077404274623825
average over h=1-12
5.223104629340352
mis:h1-12
[0.43603576 0.67692232 0.84761771 0.93096959 1.04636412 1.15134452
 1.24818659 1.3395149  1.42684322 1.51219972 1.59815709 1.69208542]
average over h=1-6
0.8482090049555934
average over h=1-12
1.1588534133500878
msis:h1-12
[ 2.59459426  4.0431114   5.08125744  5.55619079  6.24499365  6.87053699
  7.44724089  7.99130441  8.51144717  9.01941796  9.53010304 10.08578705]
average over h=1-6
5.06511408865818
average over h=1-12
6.914665420035662
ts
2
mse:h1-12
[0.00432966 0.01088011 0.01797433 0.03118608 0.04707456 0.06911232
 0.09350108 0.12158063 0.14990688 0.17286257 0.18232163 0.19216958]
average over h=1-6
0.030092842760687966
average over h=1-12
0.09107495197043801
mape:h1-12
[ 1.71261197  2.78492128  3.6220975   4.80991217  5.90515973  7.40223587
  8.69456044 10.02985794 11.51796612 12.61818353 13.22880527 14.09560836]
average over h=1-6
4.3728230867118505
average over h=1-12
8.035160015227019
mis:h1-12
[0.30470452 0.51376285 0.64055225 0.74460879 0.84292738 0.96630297
 1.17054933 1.44789814 1.51701144 1.51652739 1.47289511 1.41347482]
average over h=1-6
0.6688097935083089
average over h=1-12
1.045934582144443
msis:h1-12
[1.36372896 2.30928098 2.87879229 3.34868549 3.79382487 4.36401103
 5.31619841 6.61322849 6.92431703 6.9006025  6.67375041 6.37175625]
average over h=1-6
3.0097206018280587
average over h=1-12
4.738181391379723
ts
3
mse:h1-12
[0.00134098 0.0050121  0.00862198 0.01113217 0.01266408 0.01269096
 0.01194261 0.01117963 0.01086541 0.01119534 0.01177022 0.01284458]
average over h=1-6
0.008577044799880786
average over h=1-12
0.010105004422184238
mape:h1-12
[ 5.02240932  8.64768161 11.53959233 13.33379759 15.12088647 17.02665594
 16.43943896 16.11136538 15.19645909 15.67436832 16.26760548 16.10080151]
average over h=1-6
11.781837210813181
average over h=1-12
13.8734218342163
mis:h1-12
[0.17986783 0.4779145  0.64918933 0.70105648 0.58308162 0.47267167
 0.51393347 0.55258866 0.58915002 0.62426898 0.65841494 0.69280291]
average over h=1-6
0.5106302394692267
average over h=1-12
0.5579117006508106
msis:h1-12
[1.23982052 3.38044741 4.61143105 4.97485115 4.04507706 3.1932578
 3.46876354 3.7268145  3.97072954 4.20472836 4.43190673 4.66025413]
average over h=1-6
3.5741474991282627
average over h=1-12
3.8256734832658545

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
The additional installation of other packages with specific versions is needed using
```
pip install gluonts==0.8.1 
pip install mxnet --ignore-installed certifi
```
Ps: Using Python 3.6 is essential for DeepAR and DeepState.
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

- Xixi Li, Jingsong Yuan (2023).  DeepTVAR: Deep Learning for a Time-Varying VAR Model with Extension to Integrated VAR.  [International Journal of Forecasting]().



