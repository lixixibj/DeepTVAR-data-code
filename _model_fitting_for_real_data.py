1  #!/usr/bin/env python36
2  # -*- coding: utf-8 -*-



import numpy as np
import pandas as pd
from seed import *

import numpy as np
import torch
from custom_loss_float import *
import random
from lstm_network import DeepTVAR_lstm




def get_t_function_values_for_train_data(series_len):
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
           shape: (2,T)
     """
    time_functions_array = np.zeros(shape=(2, series_len))
    t = (np.arange(series_len) + 1) / series_len
    time_functions_array[0, :] = t
    time_functions_array[1, :] = t * t
    return time_functions_array




def get_data_and_time_function_values(train_data,difference_list):
    r"""
        Prepare time function values and data for neural network training.
        Parameters
        ----------
        train_data
           description: training data
           type: dataframe
           shape: (T,m+1)
        difference_list
           description: a list contains difference infor for each series
           type: list
           shape: m

        Returns
        -------
        data_and_t_function_values
           description: the observations of multivariate time series and values of time functions
           type: dict
    """

    data = train_data.iloc[:,1:]
    #difference processing
    for j in range(data.shape[1]):
        ts = data.iloc[:, j]
        if difference_list[j]:
            # shape (T,), the first obs becomes NA
            ts_differenced = ts.diff()
        else:
            ts_differenced = ts
        data.iloc[:, j] = ts_differenced

    data=data.dropna()
    seq_len_of_train_data = data.shape[0]
    #get time function values
    data_and_t_function_values={}
    # time_feature_array: shape(2,T)
    time_functions_array = get_t_function_values_for_train_data(seq_len_of_train_data)
    time_functions_temp = time_functions_array
    time_functions_array1 = time_functions_temp.transpose().tolist()
    time_functions = []
    time_functions.append(time_functions_array1)
    t_func_array = np.array(time_functions)
    # the shape of time functions array: (batch=1,seq=T,input_size=2)
    data_and_t_function_values['t_functions'] = torch.from_numpy(t_func_array)
    # observations shape (T,m)
    observations_array = np.array(data)
    # observations_array: shape(seq_len*m)
    data_and_t_function_values['multi_target'] = torch.from_numpy(observations_array)

    return data_and_t_function_values








def change_data_shape(original_data):
    r"""
        Change shape of data.
        Parameters
        ----------
        original_data
           description: the original data
           type: tensor
           shape: (batch,seq,input)
        Returns
        -------
        transformed data 
           description: transformed data
           type: tensor
           shape: (seq,batch,input)
    """
    original_data=original_data.numpy()
    new_data=[]
    for seq_temp in range(original_data.shape[1]):
        new_data.append(original_data[:,seq_temp,:].tolist())
    #change to tensor
    new_data=torch.from_numpy(np.array(new_data))
    return new_data


def get_stop_num(threshould,likelihood_list):
    trigger=0
    for num in range(2,len(likelihood_list)):
        current_likelihood=-likelihood_list[num]
        past1_likelihood=-likelihood_list[num-1]
        abs_relative_error1=abs((current_likelihood-past1_likelihood)/past1_likelihood)
        print('abs_relative_error1')
        print(abs_relative_error1)
        if abs_relative_error1<threshould:
            trigger=trigger+1
            past1_likelihood=-likelihood_list[num-1]
            past2_likelihood=-likelihood_list[num-2]
            abs_relative_error2=abs((past1_likelihood-past2_likelihood)/past2_likelihood)
            print('abs_relative_error2')
            print(abs_relative_error2)
            if abs_relative_error2<threshould:
                print('trigger')
                trigger=trigger+1
                print('num-index')
                print(num)
                break
    return num



def train_network(data,difference_list,m,order,num_layers,iterations,hidden_dim,res_saving_path,threshould):
    
    r"""
        Network training
        Parameters
        ----------
        data
           description: the original data (no difference)
           type: dataframe
           shape: (T,m+1)

        difference_list
           description: a list contains difference infor for each series
           type: list

        m
           description: the dimension of multivariate ts
           type: int
        order
           description: the order of VAR
           type: int

        iterations
           description: the number of iterations
           type: int

        hidden_dim
           description: the number of dimenison of hidden state in lstm
           type: int

        res_saving_path
           description: the path for saving fitting results
           type: str

        Returns
        -------
        lstm_model
            description: pretrained lstm network

    """

    data_and_t_function_values = get_data_and_time_function_values(data,difference_list)
    x = data_and_t_function_values['t_functions']
    y = data_and_t_function_values['multi_target']
    lstm_model = DeepTVAR_lstm(input_size=x.shape[2],
                          hidden_dim=hidden_dim,
                          num_layers=num_layers,
                          seqence_len=x.shape[1],
                          m=m,
                          order=order)
    lstm_model = lstm_model.float()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
    x_input = change_data_shape(x)
    count_temp=1
    #create folder for saving pretrained-models
    import os
    pretrained_model_file_path = res_saving_path+ 'pretrained_model/'
    model_folder = os.path.exists(pretrained_model_file_path)
    if not model_folder:  # 
        os.makedirs(pretrained_model_file_path)

    #create folder for saving estimated coeffs   
    coeffs_path = res_saving_path+ 'A/'
    coeffs_folder = os.path.exists(coeffs_path)
    if not coeffs_folder:  # 
        os.makedirs(coeffs_path)

    #create folder for saving estimated var-cov   
    var_cov_path = res_saving_path+ 'var_cov/'
    var_cov_folder = os.path.exists(var_cov_path)
    if not var_cov_folder:  # 
        os.makedirs(var_cov_path)


    loss_list=[]
    for i in range(iterations):
        count_temp = 1 + count_temp
        A_coeffs_of_VAR_p, lower_traig_parms, initial_var_cov_params_of_initial_obs = lstm_model(x_input.float())
        loss = compute_log_likelihood(target=y.float(),
            A_coeffs_from_lstm=A_coeffs_of_VAR_p.float(),
            lower_triang_params_form_lstm=lower_traig_parms,
            m=m,
            order=order,
            var_cov_params_for_initial_obs=initial_var_cov_params_of_initial_obs)
        loss_list.append(loss.detach().numpy()[0,0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('iterations' + str(i + 1))
        print('Train Loss: ' + str(loss))
        #saving model
        if i>2:
            current_likelihood=-loss_list[i]
            past1_likelihood=-loss_list[i-1]
            abs_relative_error1=abs((current_likelihood-past1_likelihood)/past1_likelihood)
            print('abs_relative_error1')
            print(abs_relative_error1)
            if abs_relative_error1<threshould:
                past1_likelihood=-loss_list[i-1]
                past2_likelihood=-loss_list[i-2]
                abs_relative_error2=abs((past1_likelihood-past2_likelihood)/past2_likelihood)
                print('abs_relative_error2')
                print(abs_relative_error2)
                if abs_relative_error2<threshould:
                    break


    model_name = pretrained_model_file_path + str(i) + '_' + 'net_params.pkl'
    torch.save(lstm_model.state_dict(), model_name)
    #saving loss values
    loss_pd= pd.DataFrame({'loss':loss_list})
    loss_pd.to_csv(res_saving_path+'loss.csv')

    return lstm_model










