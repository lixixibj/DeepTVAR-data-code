1  # !/usr/bin/env python36
2  # -*- coding: utf-8 -*-


# -----------------main code for simulation study------------------------


import numpy as np
import pandas as pd
import random
from custom_loss_float import *
from seed import *
import random
from lstm_network import*
import torch
import time
import pandas as pd
from lstm_network import DeepTVAR_lstm
from custom_loss_float import A_coeffs_for_causal_VAR


def get_t_function_values_(series_len):
    r"""
        Get values of time functions.
        Parameters
        ----------
        sample_size
           description: the length of time series
           type: int
           shape: T
        Returns
        -------
        time_functions_array
           description: array of time function values
           type: array
           shape: (3,T)
     """

    time_functions_array = np.zeros(shape=(6, series_len))
    t = (np.arange(series_len) + 1) / series_len
    time_functions_array[0, :] = t
    time_functions_array[1, :] = t * t
    time_functions_array[2, :] = t * t * t
    inverse_t = 1 / (np.arange(series_len) + 1)
    time_functions_array[3, :] = inverse_t
    time_functions_array[4, :] = inverse_t * inverse_t
    time_functions_array[5, :] = inverse_t * inverse_t * inverse_t


    return time_functions_array


def get_time_function_values(seq_len):
    r"""
        Prepare time function values and data for neural network training.
        Parameters
        ----------
        path_of_dataset
           description: data storage path
           type: str
        Returns
        -------
        data_and_t_function_values
           description: the observations of time series and values of time functions
           type: dict
    """
    #data_name = path_of_dataset + '_train.csv'
   
    train_data = {}
    # time_feature_array: shape(6*seq_len)
    time_functions_array = get_t_function_values_(seq_len)
    time_functions_array1 = time_functions_array.transpose().tolist()
    time_functions = []
    time_functions.append(time_functions_array1)
    t_func_array = np.array(time_functions)
    # original_shape: num.of.train.data*seq_t*num.of.features (batch,seq,input)
    # here we need to change the shape of the all_train_data_feature to(sep,batch,input)



    return torch.from_numpy(t_func_array)



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
    # change to numpy from tensor
    original_data = original_data.numpy()
    new_data = []
    for seq_temp in range(original_data.shape[1]):
        new_data.append(original_data[:, seq_temp, :].tolist())
    # change to tensor
    new_data = torch.from_numpy(np.array(new_data))
    return new_data


def get_estimated_tv_params(len_of_seq,order,m,A_coeffs_of_VAR_p,lower_traig_parms):
    r"""
        Print estimated AR parameters
        Parameters
        ----------
        len_of_seq
           description: length of time series
           type: int 
        m
           description: number of time series
           type: int      
         order
           description: order of VAR model
           type: int  
        A_coeffs_of_VAR_p
           description: VAR coefficients generated from an LSTM network
           type: tensor
        lower_traig_parms
           description: residual parameters generated from an LSTM network
           type: tensor      
              
        Returns
        -------
    """

    all_A_coeffs_list = []
    all_var_cov_list=[]
    for t in range(len_of_seq):
        var_cov_innovations_varp = make_var_cov_matrix_for_innovation_of_varp(lower_traig_parms[t, 0, :], m,order)
        A_coeffs = A_coeffs_for_causal_VAR(A_coeffs_of_VAR_p[t, 0, :], order, m, var_cov_innovations_varp)
        all_A_coeffs_list.append(A_coeffs)
        all_var_cov_list.append(var_cov_innovations_varp)
        # generate array for saving time varying parmeters (num*T)
        all_A_etsimated_coeffs = np.ones((m * m * order, len_of_seq))
    for t1 in range(len_of_seq):
            # A_t: p*(m*m)
        A_t = all_A_coeffs_list[t1]
        num = 0
        for lag in range(order):
            one_A = A_t[:, :, lag]
            for r in range(m):
                for c in range(m):
                    all_A_etsimated_coeffs[num, t1] = one_A[r, c]
                    num = num + 1
   
    all_estimated_var_cov = np.ones((m*m, len_of_seq))
    for t in range(len_of_seq):
        var_cov_t=all_var_cov_list[t]
        count_num=0
        for r in range(m):
            for c in range(m):
                all_estimated_var_cov[count_num,t]=var_cov_t[r,c]
                count_num=count_num+1

#np.array(estimated_A_df.iloc[:,order:]): shape (num,T-order)
    return all_A_etsimated_coeffs[:,order:],all_estimated_var_cov[:,order:]




def get_estimated_params_from_trained_model(sequence_length, num_layers, hidden_dim, m, order,path_of_pretrained_net):
    r"""
        Train neural network
        Parameters
        ----------
        sequence_length
           description: sample size
           type: str
        num_layers
           description: number of LSTM network layer
           type: int

        hidden_dim
           description: the number of dimenison of hidden state in lstm
           type: int
        m
           description: the dimension of multivariate ts
           type: int
        order
           description: the order of VAR
           type: int
        path_of_pretrained_net
           description: the path of trained network
           type: str  

        Returns
        -------
    """


    x = get_time_function_values(sequence_length)
    lstm_model = DeepTVAR_lstm(input_size=x.shape[2],
                          hidden_dim=hidden_dim,
                          num_layers=num_layers,
                          seqence_len=sequence_length,
                          m=m,
                          order=order)
    #lstm_model = lstm_model.float()
    lstm_model = lstm_model.float()
    #load initilized model
    lstm_model.load_state_dict(torch.load(path_of_pretrained_net))
    x_input = change_data_shape(x)
    A_coeffs_of_VAR_p, lower_traig_parms, initial_var_cov_params_of_initial_obs = lstm_model(x_input.float())

    #save loss, tv-params and pretrained-model
    all_estimated_A,all_estimated_var_cov=get_estimated_tv_params(sequence_length,order,m,A_coeffs_of_VAR_p,lower_traig_parms)

    return all_estimated_A,all_estimated_var_cov





def plot_estimates_over_100(simulated_A_path,simulated_var_cov_path,mean_estimated_A,mean_estimated_var_cov,lower_A,upper_A,lower_var_cov,upper_var_cov,saving_path):
    A_path =saving_path+'/estimated-A-mean/'
    import os
    A_folder = os.path.exists(A_path)
    if not A_folder: 
        os.makedirs(A_path)
    simulated_A_df = pd.read_csv(simulated_A_path)
    estimated_A_df=pd.DataFrame(mean_estimated_A)
    estimated_var_cov_df=pd.DataFrame(mean_estimated_var_cov)
    lower_A_df=pd.DataFrame(lower_A)
    upper_A_df=pd.DataFrame(upper_A)
    lower_var_cov_df=pd.DataFrame(lower_var_cov)
    upper_var_cov_df=pd.DataFrame(upper_var_cov)
    from matplotlib.pylab import plt

    k = simulated_A_df.shape[0]
    x_value = np.array(range(simulated_A_df.shape[1])[(1+order):])
    for i in range(k):
                # plt.figure(figsize=(20, 10))
        s = simulated_A_df.iloc[i, (order+1):]
                # plt.plot(s)
        e = estimated_A_df.iloc[i,:]

        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(x_value, list(s), '-b', label='True value')
        ax.plot(x_value, list(e), '-r', label='Mean')
        ax.plot(x_value, list(lower_A_df.iloc[i,:]), '-g', label='Lower')
        ax.plot(x_value, list(upper_A_df.iloc[i,:]), '-m', label='Upper')
        ax.legend(loc='upper left', frameon=False)

        name = 'the_'+str(i + 1) + 'th_param_100_average_A_coeffs.png'
                # plt.savefig('./image/poly_trend_sine_s50/A/' + name)
        plt.savefig(A_path+ name)
        plt.close()

    simulated_var_cov_df = pd.read_csv(simulated_var_cov_path)
            # estimated_lower_tri_df = df
            # visulize
    # print(simulated_elems_df.head())
    # print(simulated_elems_df.shape)
    var_cov_path = saving_path+ '/estimated-var-cov-mean/' 
    var_cov_folder = os.path.exists(var_cov_path)
    if not var_cov_folder: 
        os.makedirs(var_cov_path)
    
    k = simulated_var_cov_df.shape[1]-1
    for i in range(k):
        s = simulated_var_cov_df.iloc[order:,(i+1)]
                # plt.plot(s)
        e = estimated_var_cov_df.iloc[i,:]
                # plt.plot(e)
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(x_value, list(s), '-b', label='True value')
        ax.plot(x_value, list(e), '-r', label='Mean')
        ax.plot(x_value, list(lower_var_cov_df.iloc[i,:]), '-g', label='Lower')
        ax.plot(x_value, list(upper_var_cov_df.iloc[i,:]), '-m', label='Upper')
                # ax.axis('equal')
                # leg = ax.legend();
        ax.legend(loc='upper left', frameon=False)
        ax.set_title('$a_{1t}(1,1)$ ')
        name = 'the_'+str(i + 1) + 'th_param_100_average_var_covs.png'

        plt.savefig(var_cov_path+ name)
        plt.close()





#####################################################main######################################################################################
m = 2
order = 2
hidden_dim=18
num_layers = 1
simulated_A_path='./simulation-res/simulated-params-data/A_coeffs_VAR_m2_p2_T500_causality_more_complex_1.csv'
simulated_var_cov_path='./simulation-res/simulated-params-data/fitted_cov_by_order_T500_from_macr_more_complex_1.csv'
saving_path='./simulation-res/res/'
threshould=1e-6

seed_value=511
len_of_seq=500
n=100
set_global_seed(seed_value)

#100 estimations
all_A_coffs_all_t=np.zeros((n,m * m * order, len_of_seq-order))
all_var_cov_elets_all_t = np.zeros((n,m*m, len_of_seq-order))
#100 estimation
for i in range(n):
    print('i')
    print(i+1)
    path_of_pretrained_net=saving_path+str(i)+'/pretrained_model/449_net_params.pkl'
    all_estimated_A,all_estimated_var_cov=get_estimated_params_from_trained_model(len_of_seq, num_layers, hidden_dim, m, order,path_of_pretrained_net)

    #all_estimated_A,all_estimated_var_cov=train_network(path_of_dataset, num_layers, iterations, hidden_dim, m, order, path_of_initialized_params,simulated_A_path, simulated_var_cov_path,saving_path,threshould,i)
    all_A_coffs_all_t[i,:,:]=all_estimated_A
    all_var_cov_elets_all_t[i,:,:]=all_estimated_var_cov
#plot mean

#mean_A:num*(T-order)

mean_A=np.mean(all_A_coffs_all_t, axis=0)
mean_var_cov=np.mean(all_var_cov_elets_all_t,axis=0)
#save results
var_A=np.var(all_A_coffs_all_t, axis=0)
var_var_cov=np.var(all_var_cov_elets_all_t,axis=0)
lower_A=mean_A-1.96*(np.sqrt(var_A)/np.sqrt(n))
upper_A=mean_A+1.96*(np.sqrt(var_A)/np.sqrt(n))
lower_var_cov=mean_var_cov-1.96*(np.sqrt(var_var_cov)/np.sqrt(n))
upper_var_cov=mean_var_cov+1.96*(np.sqrt(var_var_cov)/np.sqrt(n))

print(var_A.shape)
print(var_var_cov.shape)


plot_estimates_over_100(simulated_A_path,simulated_var_cov_path,mean_A,mean_var_cov,lower_A,upper_A,lower_var_cov,upper_var_cov,'./simulation-res/')




