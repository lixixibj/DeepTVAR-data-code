1  #!/usr/bin/env python35
2  # -*- coding: utf-8 -*-



import numpy as np
import pandas as pd
import numpy as np
import torch
from custom_loss_float import *
import random
import os
from forecasting_accuracy import *
from seed import *
import random
from lstm_network import DeepTVAR_lstm
from train_model_for_real_data_eu_3_prices_many_times import *


def get_t_function_values_for_train_and_test_data(len_of_train_and_test,horizon):
    r"""
        Get t function values.
        Parameters
        ----------
        len_of_train_and_test
           description: the length of training and test part of time series
           type: int
        horizon
           description: forecast horizon.
           type: int


        Returns
        -------
        time_functions_array
           description: array of time function values
           type: arrary
           shape: (2,len_of_train_and_test_data)
     """

    len_of_train_ts=len_of_train_and_test-horizon
    # print(ts_entry)
    print('len_of_train_ts')
    print(len_of_train_ts)
    time_functions_array = np.zeros(shape=(2, len_of_train_and_test))
    t = (np.arange(len_of_train_and_test) + 1) / len_of_train_ts
    time_functions_array[0, :] = t
    time_functions_array[1, :] = t * t

    return time_functions_array



def get_data_and_time_function_values_for_prediction(train_and_test_original_data,difference_list,horizon):
    r"""
        Prepare t function values and data for predictions
        Parameters
        ----------
        train_and_test_original_data
           description: the original data (not difference processing)
           type: dataframe

        difference_list
           description: a list contains difference infor for each series
           type: list

        horizon
           description: forecast horizon.
           type: int

        Returns
        -------
        data_and_t_functions
           description: the differenced observations of multivariate ts and time function values
           type: dict
    """

    data = train_and_test_original_data.iloc[:,1:]
    # difference processing
    for j in range(data.shape[1]):
        ts = data.iloc[:, j]
        if difference_list[j]:
            # shape (seq_len,)
            ts_differenced = ts.diff()
        else:
            ts_differenced = ts
        data.iloc[:, j] = ts_differenced
    diff_data=data.dropna()
    data_and_t_functions = {}
    seq_len_of_train_and_test=diff_data.shape[0]
    #time_feature_array: shape(2,seq_len)
    time_feature_array=get_t_function_values_for_train_and_test_data(seq_len_of_train_and_test,horizon)
    #observations: shape(m*seq_len)
    time_feature_temp=time_feature_array
    time_feature_array1=time_feature_temp.transpose().tolist()
    time_features=[]
    time_features.append(time_feature_array1)
    t_func_array= np.array(time_features)
    # the shape of time functions array: (batch=1,seq=T,input_size=2)
    data_and_t_functions['t_functions'] = torch.from_numpy(t_func_array)
    #observations_array: shape(seq_len*m)
    data_and_t_functions['multi_target'] = torch.from_numpy(np.array(diff_data))
    return data_and_t_functions





def change_data_shape(original_data):
    r"""
        Change shape of data.
        Parameters
        ----------
        original_data
           description: the original data
           type: tensor
           shape: (batch,seq,input_size)
        Returns
        -------
        transformed data 
           description: transformed data
           type: tensor
           shape: (seq,batch,input_size)
    """
    original_data=original_data.numpy()
    new_data=[]
    for seq_temp in range(original_data.shape[1]):
        new_data.append(original_data[:,seq_temp,:].tolist())
    #change to tensor
    new_data=torch.from_numpy(np.array(new_data))
    return new_data







def forecast_based_on_pretrained_model(original_train_test_data,m,order,difference_list,num_layers,hidden_dim,pretrained_model,horizon,seasonality):
    r"""
        Make predictions based on pretrained model
        Parameters
        ----------
        origina_train_test_data
           description: the original data
           type: dataframe
           shape: (T+h,m+1)
        m
           description: the dimension of multivariate ts
           type: int
        order
           description: the order of VAR
           type: int

        difference_list
           description: a list contains difference infor for each series
           type: list

        num_layers
           description: the number of layers in an lstm network
           type: int

        hidden_dim
           description: the number of hidden dimenisons in an lstm network
           type: int

        pretrained_model
           description: the pretrained model


        horizon
           description: forecast horizon.
           type: int


        seasonality
           description: the seasonality of time series
           type: int

        Returns
        -------
    """

    data_and_t_function_values=get_data_and_time_function_values_for_prediction(original_train_test_data,difference_list,horizon)
    #x:shape(batch=1,seq=(len_of_original_series-1),input=2)
    x = data_and_t_function_values['t_functions']
    y = data_and_t_function_values['multi_target']
    lstm_model = pretrained_model
    seqence_len_all = x.shape[1]
    seqence_len_train_of_diff_data=seqence_len_all-horizon
    print('se_len')
    print(seqence_len_all)
    origial_data=original_train_test_data.iloc[:,1:]
    original_observations=np.array(origial_data)
    original_observations_train=original_observations[0:(seqence_len_train_of_diff_data+1),:]
    train_of_diff_data=y[0:seqence_len_train_of_diff_data,:].detach().numpy()
    train_and_test_of_differenced_observations = y.detach().numpy()
    x_input = change_data_shape(x)
    print('x-input')
    print(x_input.shape)
    print(x_input)
    A_coeffs_of_VAR_p, lower_traig_parms, initial_var_cov_params_of_initial_obs = lstm_model(x_input.float())
    print('A_coeffs_of_VAR_p')
    print(A_coeffs_of_VAR_p)
    print('lower_traig_parms')
    print(lower_traig_parms)
    

   ######################begin to forecast###################################
    mp = m * order
    #calulate point forecasting
    #get last p obs: Y_T=[y_T,y_T-1,...,y_T-p+1]
    Y_T=torch.zeros([mp,1])
    for i in range(order):
        obs = train_and_test_of_differenced_observations[(seqence_len_train_of_diff_data - 1-i), :].reshape(m, 1)
        b=i*m
        e=b+m
        Y_T[b:e,:]=torch.from_numpy(obs).float()

    identy_m=torch.eye(m)
    zeros_cols = torch.zeros([m, (mp - m)])
    #J: shape(m,mp)
    J= torch.cat((identy_m, zeros_cols), 1)

    point_forecasts_list=[]
    upper_forecast_list=[]
    lower_forecast_list=[]
    de_diff_var_list=[]

#get point forecasts for differenced series
    for l in range(1,horizon+1):
        ##first put VAR(p) into VAR(1) form
        #get A coeffis of testing part
        #################################point forecast#############################
        A_multiply_tmp = torch.eye(mp, mp)
        #note: range(b,e+1)=[b,...,e]
        for i in range(1,l+1):
            print('A_t')
            print(A_coeffs_of_VAR_p[(seqence_len_train_of_diff_data+l-i),0,:])
            A_t_temp=A_coeffs_of_VAR_p[(seqence_len_train_of_diff_data+l-i),0,:]
            print('A_t_temp')
            print(A_t_temp)
            # return all_stationary_F.shape:d*d*p
            var_cov_innovations_varp=make_var_cov_matrix_for_innovation_of_varp(lower_traig_parms[(seqence_len_train_of_diff_data+l-i),0, :], m, order)
            print('lower_traig_parms_t')
            print(lower_traig_parms[(seqence_len_train_of_diff_data+l-i),0, :])
            print('var_cov_innovations_varp')
            print(var_cov_innovations_varp)
            A_coeffs=A_coeffs_for_causal_VAR(A_t_temp,order,m,var_cov_innovations_varp)
            print('A_coeffs')
            print(A_coeffs)
            #put VAR(p) into VAR(1) and get corresponding big A coffs matrix for VAR(1)
            FF = get_A_coeff_m_for_VAR_1(A_coeffs, m, order)
            A_multiply_tmp=torch.mm(A_multiply_tmp,FF)

        point_pred_for_VAR_1=torch.mm(A_multiply_tmp,Y_T)
        point_pre_for_VAR_p=torch.mm(J,point_pred_for_VAR_1)
        point_forecasts_list.append(point_pre_for_VAR_p)
        #################################interval forecast#############################              
        #1.get variance-covariance of h_step_ahead prediction error of each series (differenced series)
        square_root_list=cal_var_cov_of_prediction_error_of_differenced_series(A_coeffs_of_VAR_p,lower_traig_parms,l,seqence_len_train_of_diff_data,order,m)
        #2 make interval prediction (differenced series)
        upper_forecast=point_pre_for_VAR_p+torch.from_numpy(np.array(square_root_list).reshape(m,1)*1.96)
        lower_forecast =point_pre_for_VAR_p -torch.from_numpy(np.array(square_root_list).reshape(m, 1) * 1.96)
        upper_forecast_list.append(upper_forecast)
        lower_forecast_list.append(lower_forecast)
        #we need to calculate var-cov of orginal series (original series)
        square_root_list_for_dedifferenced_series=cal_var_cov_of_prediction_error_of_original_series(A_coeffs_of_VAR_p, lower_traig_parms, l, seqence_len_train_of_diff_data, order, m)
        de_diff_var_list.append(square_root_list_for_dedifferenced_series)

    print('A_coeffs')
    print(A_coeffs)
    print('A_multiply_tmp')
    print(A_multiply_tmp)


    forecasts_ar=torch.cat(point_forecasts_list, dim=1)
        # print(forecasts_ar.shape)
    upper_forecast_ar=torch.cat(upper_forecast_list, dim=1)
    lower_forecast_ar = torch.cat(lower_forecast_list, dim=1)


        #invert forecasts to original forecasts (difference)

        #final_forecast:shape(m,horizon)
    point_forecast_array=forecasts_ar.detach().numpy()
    upper_forecast_array = upper_forecast_ar.detach().numpy()
    lower_forecast_array = lower_forecast_ar.detach().numpy()
    # de_diff_var:shape(horizon*num_of_ts)
    de_diff_var = np.array(de_diff_var_list).transpose()

    #process difference (not all)
    for r in range(m):
        if difference_list[r]:
            point_forecast_array[r,:] = np.cumsum(point_forecast_array[r,:]) +original_observations_train[-1,r]
            upper_forecast_array[r,:]=point_forecast_array[r,:]+1.96*de_diff_var[r,:]
            lower_forecast_array[r, :] = point_forecast_array[r, :] - 1.96 * de_diff_var[r, :]

    print('original_observations')
    print(original_observations.shape)
    print(seqence_len_train_of_diff_data)
    acutal_observations_for_test=original_observations[(seqence_len_train_of_diff_data+1):,:]
    print('acutal_observations_shape')
    print(acutal_observations_for_test.shape)
    mse_list=[]
    mape_list=[]
    msis_list=[]

    for i in range(m):
        print(i)
        print('mse')
        print(mse_cal(acutal_observations_for_test[:,i],point_forecast_array[i,:]))
        print('mean-mse')
        print(np.mean(mse_cal(acutal_observations_for_test[:,i],point_forecast_array[i,:])))
        mse_list.append(list(mse_cal(acutal_observations_for_test[:,i],point_forecast_array[i,:])))


        print('mape')
        print(mape_cal(acutal_observations_for_test[:,i], point_forecast_array[i,:]))
        print('mean-mape')
        print(np.mean(mape_cal(acutal_observations_for_test[:,i], point_forecast_array[i,:])))
        mape_list.append(mape_cal(acutal_observations_for_test[:,i], point_forecast_array[i,:]))

        print('msis')
        print(msis(original_observations[0:(seqence_len_train_of_diff_data+1),i], acutal_observations_for_test[:,i],
                       upper_forecast_array[i,:], lower_forecast_array[i,:], 0.05, seasonality, horizon))
        print('mean-msis')
        print(np.mean(msis(original_observations[0:(seqence_len_train_of_diff_data+1),i], acutal_observations_for_test[:,i],
                       upper_forecast_array[i,:], lower_forecast_array[i,:], 0.05, seasonality, horizon)))
            #cal msis
        msis_list.append(msis(original_observations[0:(seqence_len_train_of_diff_data+1),i], acutal_observations_for_test[:,i],
                       upper_forecast_array[i,:], lower_forecast_array[i,:], 0.05, seasonality, horizon))


    return mse_list,mape_list,msis_list,point_forecast_array,lower_forecast_array,upper_forecast_array






def cal_var_cov_of_prediction_error_of_differenced_series(A_coeffs_of_VAR_p,lower_traig_parms,horizon,seqence_len_train,order,m):
    r"""
        Calculate variance-covariance matrix of prediction error for differenced series.
        Parameters
        ----------
        A_coeffs_of_VAR_p
           description: the initial A coffes generated from LSTM.
           type: tensor, 3d

        lower_traig_parms
           description: the elements of lower triangular matrix for generating variance-covariance matrix of innovations.
           type: tensor

        horizon
           description: forecast horizon.
           type: int

        seqence_len_train
           description: len of train data (differenced data).
           type: int

        m
           description: the dimension of multivariate ts
           type: int
        order
           description: the order of VAR
           type: int

        Returns
        -------
        var_list
           description: a list contains the variance of prediction error of each component series.
        type: list
    """

    mp=m*order
    identy_m=torch.eye(m)
    zeros_cols = torch.zeros([m, (mp - m)])
    #J: shape(m,mp)
    J= torch.cat((identy_m, zeros_cols), 1)
    var_cov=torch.zeros(m,m)
    for i in range(horizon):
        #U_covariance:shape(mp*mp)
        U_covariance = make_var_cov_of_innovations_var1(lower_traig_parms[(seqence_len_train + horizon-1-i), 0, :], m, order)
        if i==0:
            A_temp=torch.mm(torch.mm(J,U_covariance),J.t())
            var_cov=A_temp+var_cov
        if i>0:
            A_multiply_tmp = torch.eye(mp, mp)
            for j in range(1,(i+1)):
                A_t_temp = A_coeffs_of_VAR_p[(seqence_len_train + horizon-1-j+1), 0, :]
                var_cov_innovations_varp=make_var_cov_matrix_for_innovation_of_varp(lower_traig_parms[(seqence_len_train + horizon-1-j+1),0, :], m, order)
                A_coeffs=A_coeffs_for_causal_VAR(A_t_temp,order,m,var_cov_innovations_varp)
                #A_coeffs = (staionary_coefs(F_out_network=F_t_temp, p=lag_order, d=m))
                AA = get_A_coeff_m_for_VAR_1(A_coeffs, m, order)
                A_multiply_tmp=torch.mm(A_multiply_tmp,AA)
            J_AA=torch.mm(J,A_multiply_tmp)
            var_cov=var_cov+torch.mm(torch.mm(J_AA,U_covariance),J_AA.t())
    #cal h_step forecast variance of each time series
    var_list=[]
    import math
    for n in range(m):
        var_list.append(math.sqrt(var_cov[n,n]))
    return var_list



def cal_var_cov_of_prediction_error_of_original_series(A_coeffs_of_VAR_p,lower_traig_parms,horizon,seqence_len_train,order,m):
    r"""
        Calculate variance-covariance matrix of prediction error for original (integrated) series.
        Parameters
        ----------
        A_coeffs_of_VAR_p
           description: the initial A coffes generated from LSTM.
           type: tensor, 3d

        lower_traig_parms
           description: the elements of lower triangular matrix for generating variance-covariance matrix of innovations.
           type: tensor

        horizon
           description: forecast horizon.
           type: int

        seqence_len_train
           description: len of train data (differenced data).
           type: int

        m
           description: the dimension of multivariate ts
           type: int
        order
           description: the order of VAR
           type: int

        Returns
        -------
        var_list
           description: a list contains the variance of prediction error of each component series.
        type: list
    """
    mp=m*order
    identy_m=torch.eye(m)
    zeros_cols = torch.zeros([m, (mp - m)])
    #J: shape(k,kp)
    J= torch.cat((identy_m, zeros_cols), 1)
    var_cov=torch.zeros(mp,mp)
    for k in range(1,horizon+1):
        print('k-size')
        print(k)
        #U_covariance:shape(mp*mp)
        U_covariance = make_var_cov_of_innovations_var1(lower_traig_parms[(seqence_len_train +k-1), 0, :],m, order)
        #make
        print('U-shape')
        print(U_covariance.shape)
        summing_m=cal_summing_term(k,mp,A_coeffs_of_VAR_p,seqence_len_train,m,lower_traig_parms,order,horizons)
        var_cov=var_cov+torch.mm(torch.mm(summing_m,U_covariance),summing_m.t())
    var_cov_of_oginal_series=torch.mm(torch.mm(J,var_cov),J.t())
    var_list=[]
    import math
    for n in range(m):
        var_list.append(math.sqrt(var_cov_of_oginal_series[n,n]))
    return var_list






def cal_summing_term(k,mp,A_coeffs_of_VAR_p,seqence_len_train,m,lower_traig_parms,order,horizon):
    r"""
        Calculate a summing term.
        Parameters
        ----------
        k
           description: the forecast horizon
           type: int
        mp
           description: m*p
           type: int

        A_coeffs_of_VAR_p
           description: the initial A coffes generated from LSTM.
           type: tensor, 3d

        lower_traig_parms
           description: the elements of lower triangular matrix for generating variance-covariance matrix of innovations.
           type: tensor

        horizon
           description: forecast horizon.
           type: int

        seqence_len_train
           description: len of train data (differenced data).
           type: int

        m
           description: the dimension of multivariate ts
           type: int
        order
           description: the order of VAR
           type: int

        Returns
        -------
        var_list
           description: a list contains the variance of prediction error of each component series.
        type: list
    """
    summing_m = torch.zeros(mp, mp)
    for l in range(k, horizon + 1):
        if l-k==0:
            summing_m=summing_m + torch.eye(mp, mp)
        if l-k>0:
            A_multiply_tmp = torch.eye(mp, mp)
            for j in range(1,(l-k+1)):
                A_t_temp = A_coeffs_of_VAR_p[(seqence_len_train + l - j), 0, :]
                var_cov_innovations_varp=make_var_cov_matrix_for_innovation_of_varp(lower_traig_parms[(seqence_len_train + l - j),0, :], m, order)
                A_coeffs=A_coeffs_for_causal_VAR(A_t_temp,order,m,var_cov_innovations_varp)
                #A_coeffs = (staionary_coefs(F_out_network=F_t_temp, p=lag_order, d=dim))
                AA = get_A_coeff_m_for_VAR_1(A_coeffs, m, order)
                A_multiply_tmp=torch.mm(A_multiply_tmp, AA)
            summing_m=summing_m+A_multiply_tmp
    return summing_m












name_of_dataset='/Users/xixili/Dropbox/DeepTVAR-code/benchmarks-all/eu-3-prices-logged.csv'
train_data=pd.read_csv(name_of_dataset)
difference_list=[True,True,True]
m=3
order=2
num_layers=1
seasonality=12
len_of_data=train_data.shape[0]
hidden_dim=15
num_layers=1
num_of_forecast=20
horizons=12
test_len=horizons+num_of_forecast-1
train_len=len_of_data-test_len
threshould=5e-6
seed_value=6000
iterations=500
saving_path='./real-data-forecasting-res-seed6000-iters500/'
#seed values for reproducting forecasting results


all_mape_ts1=np.zeros((num_of_forecast,horizons))
all_msis_ts1=np.zeros((num_of_forecast,horizons))
all_mape_ts2=np.zeros((num_of_forecast,horizons))
all_msis_ts2=np.zeros((num_of_forecast,horizons))
all_mape_ts3=np.zeros((num_of_forecast,horizons))
all_msis_ts3=np.zeros((num_of_forecast,horizons))
for f in range(num_of_forecast):
    set_global_seed(seed_value)
    print('ts-section:')
    print(f)
    b=f
    e=b+train_len
    training_data=train_data.iloc[b:e,:]
    original_train_test_data=train_data.iloc[b:(e+horizons),:]
    
    res_saving_path=saving_path+'section_'+str(b)+'/'
#model fitting
    lstm_model=train_network(training_data,difference_list,m, order, num_layers, iterations, hidden_dim,res_saving_path,threshould)
#make predictions
    mse_list,mape_list,msis_list,point_forecast_array,lower_forecast_array,upper_forecast_array=forecast_based_on_pretrained_model(original_train_test_data,m,order,difference_list,num_layers,hidden_dim,lstm_model,horizons,seasonality)
    print('point_forecast_array')
    print(point_forecast_array)
    print('lower_forecast_array')
    print(lower_forecast_array)
    print('upper_forecast_array')
    print(upper_forecast_array)
    all_mape_ts1[f,:]=mape_list[0]
    all_msis_ts1[f,:]=msis_list[0]
    all_mape_ts2[f,:]=mape_list[1]
    all_msis_ts2[f,:]=msis_list[1]
    all_mape_ts3[f,:]=mape_list[2]
    all_msis_ts3[f,:]=msis_list[2]
    #save forecasts
    pd.DataFrame(point_forecast_array).to_csv(res_saving_path+'point_forecasts.csv')
    pd.DataFrame(lower_forecast_array).to_csv(res_saving_path+'lower_forecasts.csv')
    pd.DataFrame(upper_forecast_array).to_csv(res_saving_path+'upper_forecasts.csv')

#calculate averaged accuracy
print('APE-ts1')
print(np.mean(all_mape_ts1,axis=0))
print('APE-ts2')
print(np.mean(all_mape_ts2,axis=0))
print('APE-ts3')
print(np.mean(all_mape_ts3,axis=0))

print('SIS-ts1')
print(np.mean(all_msis_ts1,axis=0))
print('SIS-ts2')
print(np.mean(all_msis_ts2,axis=0))
print('SIS-ts3')
print(np.mean(all_msis_ts3,axis=0))
















