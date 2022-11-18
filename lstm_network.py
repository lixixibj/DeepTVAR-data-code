1  #!/usr/bin/env python36
2  # -*- coding: utf-8 -*-
3  # @File  : lstm_network.py
4  # @Author: Xixi Li
5  # @Date  : 2019-12-02
6  # @Desc  :



import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable






class DeepTVAR_lstm(torch.nn.Module):
    def __init__(self,input_size,hidden_dim,num_layers,seqence_len,m,order):
        r"""
        Initilize neural network.
        Parameters
        ----------
        input_size
           description: the dimension of t functions
           type: int
        hidden_dim
           description: the dimension of ht of lstm
           type: int
        num_layers
           description: the number of layers of lstm
           type: int
        seqence_len
           description: the lenght of time series
           type: int

        var
           description: the setup of a VAR model
           type: class

        Returns
        -------
        initilized NN
        """
        super(DeepTVAR_lstm,self).__init__()
        self.input_size=input_size
        self.hidden_dim=hidden_dim
        self.num_layers=num_layers
        self.seqence_len=seqence_len
        self.m = m
        self.order=order

        self.lstm_for_var_params = nn.LSTM(self.input_size, self.hidden_dim, self.num_layers)
        #self.lstm_for_var_params = nn.LSTM(self.input_size, 5, self.num_layers)
        #number of parameters of var-cov of innovations
        num_var_cov_parmas_of_innovs = (self.m * (self.m + 1)) / 2
        #number of coefficients A
        num_A=self.order*self.m*self.m
        #affine transformations for A and var-cov
        #self.linear1=nn.Linear(self.hidden_dim,hidden_dim)
        #self.linear1=nn.Linear(5,hidden_dim)
        self.linear_affine_transf=nn.Linear(self.hidden_dim,
                                 (int(num_A)+int(num_var_cov_parmas_of_innovs)))
        mp=self.m*self.order
        # num_of_params=(mp*(mp+1))/2
        # print('self.seqence_len')
        # print(self.seqence_len)
        num_of_initial_var_cov_params_of_initial_obs = (mp * (mp + 1)) / 2
        self.initial_var_cov_params_of_initial_obs = torch.nn.Parameter(torch.randn(int(num_of_initial_var_cov_params_of_initial_obs)))
        #
        # self.h0=torch.nn.Parameter(torch.randn(self.num_layers, 1, self.hidden_dim))
        # self.c0 = torch.nn.Parameter(torch.randn(self.num_layers, 1, self.hidden_dim))



    def forward(self,inputs):

        #output_h, (hn1, cn1) = self.lstm_for_var_params(inputs,(self.h0,self.c0))
        output_h, (hn1, cn1) = self.lstm_for_var_params(inputs)
        #output_h=self.linear1(output_h)
        # print('hn1')
        # print(hn1.shape)
        # print('cn1')
        # print(cn1.shape)

        num_var_cov_parmas_of_innovs = (self.m * (self.m + 1)) / 2
        num_A=self.order*self.m*self.m
        #curve:shape:(#seq_len,batch,num_of_A_and_var)
        curve = self.linear_affine_transf(output_h)
        A_coeffs_of_VAR_p=curve[:,:,0:int(num_A)]
        lower_traig_parms=curve[:,:,int(num_A):int(num_A+num_var_cov_parmas_of_innovs)]
        seq_len=inputs.shape[0]
        #inputs_2=torch.zeros(1, 1)
        #v_covariance_coeffs=self.initial_variance_model(inputs_2)
        initial_var_cov_params_of_initial_obs = self.initial_var_cov_params_of_initial_obs
        # h0=self.h0
        # c0=self.c0
        return A_coeffs_of_VAR_p,lower_traig_parms,initial_var_cov_params_of_initial_obs



#DeepTVAR_lstm original res
class DeepTVAR_lstm_copy(torch.nn.Module):
    def __init__(self,input_size,hidden_dim,num_layers,seqence_len,m,order):
        r"""
        Initilize neural network.
        Parameters
        ----------
        input_size
           description: the dimension of t functions
           type: int
        hidden_dim
           description: the dimension of ht of lstm
           type: int
        num_layers
           description: the number of layers of lstm
           type: int
        seqence_len
           description: the lenght of time series
           type: int

        var
           description: the setup of a VAR model
           type: class

        Returns
        -------
        initilized NN
        """
        super(DeepTVAR_lstm_copy,self).__init__()
        self.input_size=input_size
        self.hidden_dim=hidden_dim
        self.num_layers=num_layers
        self.seqence_len=seqence_len
        self.m = m
        self.order=order

        #self.lstm_for_var_params = nn.LSTM(self.input_size, self.hidden_dim, self.num_layers)
        self.lstm_for_var_params = nn.LSTM(self.input_size, 5, self.num_layers)
        #number of parameters of var-cov of innovations
        num_var_cov_parmas_of_innovs = (self.m * (self.m + 1)) / 2
        #number of coefficients A
        num_A=self.order*self.m*self.m
        #affine transformations for A and var-cov
        #self.linear1=nn.Linear(self.hidden_dim,hidden_dim)
        self.linear1=nn.Linear(5,hidden_dim)
        self.linear_affine_transf=nn.Linear(self.hidden_dim,
                                 (int(num_A)+int(num_var_cov_parmas_of_innovs)))
        mp=self.m*self.order
        # num_of_params=(mp*(mp+1))/2
        # print('self.seqence_len')
        # print(self.seqence_len)
        num_of_initial_var_cov_params_of_initial_obs = (mp * (mp + 1)) / 2
        self.initial_var_cov_params_of_initial_obs = torch.nn.Parameter(torch.randn(int(num_of_initial_var_cov_params_of_initial_obs)))
        #
        # self.h0=torch.nn.Parameter(torch.randn(self.num_layers, 1, self.hidden_dim))
        # self.c0 = torch.nn.Parameter(torch.randn(self.num_layers, 1, self.hidden_dim))



    def forward(self,inputs):

        #output_h, (hn1, cn1) = self.lstm_for_var_params(inputs,(self.h0,self.c0))
        output_h, (hn1, cn1) = self.lstm_for_var_params(inputs)
        output_h=self.linear1(output_h)
        # print('hn1')
        # print(hn1.shape)
        # print('cn1')
        # print(cn1.shape)

        num_var_cov_parmas_of_innovs = (self.m * (self.m + 1)) / 2
        num_A=self.order*self.m*self.m
        #curve:shape:(#seq_len,batch,num_of_A_and_var)
        curve = self.linear_affine_transf(output_h)
        A_coeffs_of_VAR_p=curve[:,:,0:int(num_A)]
        lower_traig_parms=curve[:,:,int(num_A):int(num_A+num_var_cov_parmas_of_innovs)]
        seq_len=inputs.shape[0]
        #inputs_2=torch.zeros(1, 1)
        #v_covariance_coeffs=self.initial_variance_model(inputs_2)
        initial_var_cov_params_of_initial_obs = self.initial_var_cov_params_of_initial_obs
        # h0=self.h0
        # c0=self.c0
        return A_coeffs_of_VAR_p,lower_traig_parms,initial_var_cov_params_of_initial_obs


#DeepTVAR_lstm original res
class DeepTVAR_lstm_copy_mlp(torch.nn.Module):
    def __init__(self,input_size,hidden_dim,num_layers,seqence_len,m,order):
        r"""
        Initilize neural network.
        Parameters
        ----------
        input_size
           description: the dimension of t functions
           type: int
        hidden_dim
           description: the dimension of ht of lstm
           type: int
        num_layers
           description: the number of layers of lstm
           type: int
        seqence_len
           description: the lenght of time series
           type: int

        var
           description: the setup of a VAR model
           type: class

        Returns
        -------
        initilized NN
        """
        super(DeepTVAR_lstm_copy_mlp,self).__init__()
        self.input_size=input_size
        self.hidden_dim=hidden_dim
        self.num_layers=num_layers
        self.seqence_len=seqence_len
        self.m = m
        self.order=order

        #self.lstm_for_var_params = nn.LSTM(self.input_size, self.hidden_dim, self.num_layers)
        self.lstm_for_var_params = nn.LSTM(self.input_size, 5, self.num_layers)
        #number of parameters of var-cov of innovations
        num_var_cov_parmas_of_innovs = (self.m * (self.m + 1)) / 2
        #number of coefficients A
        num_A=self.order*self.m*self.m
        #affine transformations for A and var-cov
        #self.linear1=nn.Linear(self.hidden_dim,hidden_dim)
        self.linear1=nn.Linear(5,hidden_dim)
        self.linear_affine_transf=nn.Linear(self.hidden_dim,
                                 (int(num_A)))
        self.linear_var=nn.Linear(self.input_size,int(num_var_cov_parmas_of_innovs))

        mp=self.m*self.order
        # num_of_params=(mp*(mp+1))/2
        # print('self.seqence_len')
        # print(self.seqence_len)
        num_of_initial_var_cov_params_of_initial_obs = (mp * (mp + 1)) / 2
        self.initial_var_cov_params_of_initial_obs = torch.nn.Parameter(torch.randn(int(num_of_initial_var_cov_params_of_initial_obs)))
        #
        # self.h0=torch.nn.Parameter(torch.randn(self.num_layers, 1, self.hidden_dim))
        # self.c0 = torch.nn.Parameter(torch.randn(self.num_layers, 1, self.hidden_dim))



    def forward(self,inputs):

        #output_h, (hn1, cn1) = self.lstm_for_var_params(inputs,(self.h0,self.c0))
        output_h, (hn1, cn1) = self.lstm_for_var_params(inputs)
        output_h=self.linear1(output_h)
        # print('hn1')
        # print(hn1.shape)
        # print('cn1')
        # print(cn1.shape)

        num_var_cov_parmas_of_innovs = (self.m * (self.m + 1)) / 2
        num_A=self.order*self.m*self.m
        #curve:shape:(#seq_len,batch,num_of_A_and_var)
        A_coeffs_of_VAR_p = self.linear_affine_transf(output_h)
        #A_coeffs_of_VAR_p=curve[:,:,0:int(num_A)]
        #lower_traig_parms=curve[:,:,int(num_A):int(num_A+num_var_cov_parmas_of_innovs)]
        lower_traig_parms=self.linear_var(inputs)
        seq_len=inputs.shape[0]
        #inputs_2=torch.zeros(1, 1)
        #v_covariance_coeffs=self.initial_variance_model(inputs_2)
        initial_var_cov_params_of_initial_obs = self.initial_var_cov_params_of_initial_obs
        # h0=self.h0
        # c0=self.c0
        return A_coeffs_of_VAR_p,lower_traig_parms,initial_var_cov_params_of_initial_obs



#DeepTVAR_lstm original res
class DeepTVAR_lstm_lstm_copy_mlp(torch.nn.Module):
    def __init__(self,input_size,hidden_dim,num_layers,seqence_len,m,order):
        r"""
        Initilize neural network.
        Parameters
        ----------
        input_size
           description: the dimension of t functions
           type: int
        hidden_dim
           description: the dimension of ht of lstm
           type: int
        num_layers
           description: the number of layers of lstm
           type: int
        seqence_len
           description: the lenght of time series
           type: int

        var
           description: the setup of a VAR model
           type: class

        Returns
        -------
        initilized NN
        """
        super(DeepTVAR_lstm_lstm_copy_mlp,self).__init__()
        self.input_size=input_size
        self.hidden_dim=hidden_dim
        self.num_layers=num_layers
        self.seqence_len=seqence_len
        self.m = m
        self.order=order

        #self.lstm_for_var_params = nn.LSTM(self.input_size, self.hidden_dim, self.num_layers)
        self.lstm_for_var_params = nn.LSTM(self.input_size, 5, self.num_layers)
        #number of parameters of var-cov of innovations
        num_var_cov_parmas_of_innovs = (self.m * (self.m + 1)) / 2
        #number of coefficients A
        num_A=self.order*self.m*self.m
        #affine transformations for A and var-cov
        #self.linear1=nn.Linear(self.hidden_dim,hidden_dim)
        self.linear1=nn.Linear(5,hidden_dim)
        self.linear_affine_transf=nn.Linear(self.hidden_dim,
                                 (int(num_A)))
        self.linear2=nn.Linear(5,hidden_dim-20)
        self.linear_var=nn.Linear((self.hidden_dim-20),int(num_var_cov_parmas_of_innovs))

        mp=self.m*self.order
        # num_of_params=(mp*(mp+1))/2
        # print('self.seqence_len')
        # print(self.seqence_len)
        num_of_initial_var_cov_params_of_initial_obs = (mp * (mp + 1)) / 2
        self.initial_var_cov_params_of_initial_obs = torch.nn.Parameter(torch.randn(int(num_of_initial_var_cov_params_of_initial_obs)))
        #
        # self.h0=torch.nn.Parameter(torch.randn(self.num_layers, 1, self.hidden_dim))
        # self.c0 = torch.nn.Parameter(torch.randn(self.num_layers, 1, self.hidden_dim))



    def forward(self,inputs):

        #output_h, (hn1, cn1) = self.lstm_for_var_params(inputs,(self.h0,self.c0))
        output_h, (hn1, cn1) = self.lstm_for_var_params(inputs)
        output_h1=self.linear1(output_h)
        output_h2=self.linear2(output_h)
        # print('hn1')
        # print(hn1.shape)
        # print('cn1')
        # print(cn1.shape)

        num_var_cov_parmas_of_innovs = (self.m * (self.m + 1)) / 2
        num_A=self.order*self.m*self.m
        #curve:shape:(#seq_len,batch,num_of_A_and_var)
        A_coeffs_of_VAR_p = self.linear_affine_transf(output_h1)
        #A_coeffs_of_VAR_p=curve[:,:,0:int(num_A)]
        #lower_traig_parms=curve[:,:,int(num_A):int(num_A+num_var_cov_parmas_of_innovs)]
        lower_traig_parms=self.linear_var(output_h2)
        seq_len=inputs.shape[0]
        #inputs_2=torch.zeros(1, 1)
        #v_covariance_coeffs=self.initial_variance_model(inputs_2)
        initial_var_cov_params_of_initial_obs = self.initial_var_cov_params_of_initial_obs
        # h0=self.h0
        # c0=self.c0
        return A_coeffs_of_VAR_p,lower_traig_parms,initial_var_cov_params_of_initial_obs



#DeepTVAR_lstm original res
class DeepTVAR_lstm_seperate_hidden_state(torch.nn.Module):
    def __init__(self,input_size,hidden_dim,num_layers,seqence_len,m,order):
        r"""
        Initilize neural network.
        Parameters
        ----------
        input_size
           description: the dimension of t functions
           type: int
        hidden_dim
           description: the dimension of ht of lstm
           type: int
        num_layers
           description: the number of layers of lstm
           type: int
        seqence_len
           description: the lenght of time series
           type: int

        var
           description: the setup of a VAR model
           type: class

        Returns
        -------
        initilized NN
        """
        super(DeepTVAR_lstm_seperate_hidden_state,self).__init__()
        self.input_size=input_size
        self.hidden_dim=hidden_dim
        self.num_layers=num_layers
        self.seqence_len=seqence_len
        self.m = m
        self.order=order

        #self.lstm_for_var_params = nn.LSTM(self.input_size, self.hidden_dim, self.num_layers)
        self.lstm_for_var_params = nn.LSTM(self.input_size, 5*2, self.num_layers)
        #number of parameters of var-cov of innovations
        num_var_cov_parmas_of_innovs = (self.m * (self.m + 1)) / 2
        #number of coefficients A
        num_A=self.order*self.m*self.m
        #affine transformations for A and var-cov
        #self.linear1=nn.Linear(self.hidden_dim,hidden_dim)
        self.linear1=nn.Linear(5,hidden_dim)
        self.linear_affine_transf=nn.Linear(self.hidden_dim,
                                 (int(num_A)))
        self.linear2=nn.Linear(5,8)
        self.linear_var=nn.Linear((8),int(num_var_cov_parmas_of_innovs))

        mp=self.m*self.order
        # num_of_params=(mp*(mp+1))/2
        # print('self.seqence_len')
        # print(self.seqence_len)
        num_of_initial_var_cov_params_of_initial_obs = (mp * (mp + 1)) / 2
        self.initial_var_cov_params_of_initial_obs = torch.nn.Parameter(torch.randn(int(num_of_initial_var_cov_params_of_initial_obs)))
        #
        # self.h0=torch.nn.Parameter(torch.randn(self.num_layers, 1, self.hidden_dim))
        # self.c0 = torch.nn.Parameter(torch.randn(self.num_layers, 1, self.hidden_dim))



    def forward(self,inputs):

        #output_h, (hn1, cn1) = self.lstm_for_var_params(inputs,(self.h0,self.c0))
        output_h, (hn1, cn1) = self.lstm_for_var_params(inputs)
        output_h1=self.linear1(output_h[:,:,0:5])
        output_h2=self.linear2(output_h[:,:,5:10])
        # print('hn1')
        # print(hn1.shape)
        # print('cn1')
        # print(cn1.shape)

        num_var_cov_parmas_of_innovs = (self.m * (self.m + 1)) / 2
        num_A=self.order*self.m*self.m
        #curve:shape:(#seq_len,batch,num_of_A_and_var)
        A_coeffs_of_VAR_p = self.linear_affine_transf(output_h1)
        #A_coeffs_of_VAR_p=curve[:,:,0:int(num_A)]
        #lower_traig_parms=curve[:,:,int(num_A):int(num_A+num_var_cov_parmas_of_innovs)]
        lower_traig_parms=self.linear_var(output_h2)
        seq_len=inputs.shape[0]
        #inputs_2=torch.zeros(1, 1)
        #v_covariance_coeffs=self.initial_variance_model(inputs_2)
        initial_var_cov_params_of_initial_obs = self.initial_var_cov_params_of_initial_obs
        # h0=self.h0
        # c0=self.c0
        return A_coeffs_of_VAR_p,lower_traig_parms,initial_var_cov_params_of_initial_obs

#DeepTVAR_lstm original res
class DeepTVAR_lstm_lstm_seperate(torch.nn.Module):
    def __init__(self,input_size,hidden_dim,num_layers,seqence_len,m,order):
        r"""
        Initilize neural network.
        Parameters
        ----------
        input_size
           description: the dimension of t functions
           type: int
        hidden_dim
           description: the dimension of ht of lstm
           type: int
        num_layers
           description: the number of layers of lstm
           type: int
        seqence_len
           description: the lenght of time series
           type: int

        var
           description: the setup of a VAR model
           type: class

        Returns
        -------
        initilized NN
        """
        super(DeepTVAR_lstm_lstm_seperate,self).__init__()
        self.input_size=input_size
        self.hidden_dim=hidden_dim
        self.num_layers=num_layers
        self.seqence_len=seqence_len
        self.m = m
        self.order=order

        #self.lstm_for_var_params = nn.LSTM(self.input_size, self.hidden_dim, self.num_layers)
        self.lstm_for_var_params = nn.LSTM(self.input_size, 5, self.num_layers)
        self.lstm_for_var_cov = nn.LSTM(self.input_size, 5, self.num_layers)
        #number of parameters of var-cov of innovations
        num_var_cov_parmas_of_innovs = (self.m * (self.m + 1)) / 2
        #number of coefficients A
        num_A=self.order*self.m*self.m
        #affine transformations for A and var-cov
        #self.linear1=nn.Linear(self.hidden_dim,hidden_dim)
        self.linear1=nn.Linear(5,hidden_dim)
        self.linear_affine_transf=nn.Linear(self.hidden_dim,
                                 (int(num_A)))
        self.linear2=nn.Linear(5,hidden_dim-20)
        self.linear_var=nn.Linear((self.hidden_dim-20),int(num_var_cov_parmas_of_innovs))

        mp=self.m*self.order
        # num_of_params=(mp*(mp+1))/2
        # print('self.seqence_len')
        # print(self.seqence_len)
        num_of_initial_var_cov_params_of_initial_obs = (mp * (mp + 1)) / 2
        self.initial_var_cov_params_of_initial_obs = torch.nn.Parameter(torch.randn(int(num_of_initial_var_cov_params_of_initial_obs)))
        #
        # self.h0=torch.nn.Parameter(torch.randn(self.num_layers, 1, self.hidden_dim))
        # self.c0 = torch.nn.Parameter(torch.randn(self.num_layers, 1, self.hidden_dim))



    def forward(self,inputs):

        #output_h, (hn1, cn1) = self.lstm_for_var_params(inputs,(self.h0,self.c0))
        output_h1, (hn1, cn1) = self.lstm_for_var_params(inputs)
        output_h2, (hn2, cn2) = self.lstm_for_var_cov(inputs)
        output_h1=self.linear1(output_h1)
        output_h2=self.linear2(output_h2)
        # print('hn1')
        # print(hn1.shape)
        # print('cn1')
        # print(cn1.shape)

        num_var_cov_parmas_of_innovs = (self.m * (self.m + 1)) / 2
        num_A=self.order*self.m*self.m
        #curve:shape:(#seq_len,batch,num_of_A_and_var)
        A_coeffs_of_VAR_p = self.linear_affine_transf(output_h1)
        #A_coeffs_of_VAR_p=curve[:,:,0:int(num_A)]
        #lower_traig_parms=curve[:,:,int(num_A):int(num_A+num_var_cov_parmas_of_innovs)]
        lower_traig_parms=self.linear_var(output_h2)
        seq_len=inputs.shape[0]
        #inputs_2=torch.zeros(1, 1)
        #v_covariance_coeffs=self.initial_variance_model(inputs_2)
        initial_var_cov_params_of_initial_obs = self.initial_var_cov_params_of_initial_obs
        # h0=self.h0
        # c0=self.c0
        return A_coeffs_of_VAR_p,lower_traig_parms,initial_var_cov_params_of_initial_obs

class DeepTVAR_lstm_mlp(torch.nn.Module):
    def __init__(self,input_size,hidden_dim,num_layers,seqence_len,m,order):
        r"""
        Initilize neural network.
        Parameters
        ----------
        input_size
           description: the dimension of t functions
           type: int
        hidden_dim
           description: the dimension of ht of lstm
           type: int
        num_layers
           description: the number of layers of lstm
           type: int
        seqence_len
           description: the lenght of time series
           type: int

        var
           description: the setup of a VAR model
           type: class

        Returns
        -------
        initilized NN
        """
        super(DeepTVAR_lstm_mlp,self).__init__()
        self.input_size=input_size
        self.hidden_dim=hidden_dim
        self.num_layers=num_layers
        self.seqence_len=seqence_len
        self.m = m
        self.order=order

        self.lstm_for_var_params = nn.LSTM(self.input_size, self.hidden_dim, self.num_layers)
        #self.lstm_for_var_params = nn.LSTM(self.input_size, 5, self.num_layers)
        #number of parameters of var-cov of innovations
        num_var_cov_parmas_of_innovs = (self.m * (self.m + 1)) / 2
        #number of coefficients A
        num_A=self.order*self.m*self.m
        #affine transformations for A and var-cov
        #self.linear1=nn.Linear(self.hidden_dim,hidden_dim)
        self.linear1=nn.Linear(self.input_size,int(num_var_cov_parmas_of_innovs))
        self.linear_affine_transf=nn.Linear(self.hidden_dim,
                                 (int(num_A)))
        mp=self.m*self.order
        # num_of_params=(mp*(mp+1))/2
        # print('self.seqence_len')
        # print(self.seqence_len)
        num_of_initial_var_cov_params_of_initial_obs = (mp * (mp + 1)) / 2
        self.initial_var_cov_params_of_initial_obs = torch.nn.Parameter(torch.randn(int(num_of_initial_var_cov_params_of_initial_obs)))
        #
        # self.h0=torch.nn.Parameter(torch.randn(self.num_layers, 1, self.hidden_dim))
        # self.c0 = torch.nn.Parameter(torch.randn(self.num_layers, 1, self.hidden_dim))



    def forward(self,inputs):

        #output_h, (hn1, cn1) = self.lstm_for_var_params(inputs,(self.h0,self.c0))
        output_h, (hn1, cn1) = self.lstm_for_var_params(inputs)
        lower_traig_parms=self.linear1(inputs)
        # print('hn1')
        # print(hn1.shape)
        # print('cn1')
        # print(cn1.shape)

        num_var_cov_parmas_of_innovs = (self.m * (self.m + 1)) / 2
        num_A=self.order*self.m*self.m
        #curve:shape:(#seq_len,batch,num_of_A_and_var)
        A_coeffs_of_VAR_p = self.linear_affine_transf(output_h)
        #A_coeffs_of_VAR_p=curve[:,:,0:int(num_A)]
        #lower_traig_parms=curve[:,:,int(num_A):int(num_A+num_var_cov_parmas_of_innovs)]
        seq_len=inputs.shape[0]
        #inputs_2=torch.zeros(1, 1)
        #v_covariance_coeffs=self.initial_variance_model(inputs_2)
        initial_var_cov_params_of_initial_obs = self.initial_var_cov_params_of_initial_obs
        # h0=self.h0
        # c0=self.c0
        return A_coeffs_of_VAR_p,lower_traig_parms,initial_var_cov_params_of_initial_obs


class DeepTVAR_mlp_onelayer(torch.nn.Module):
    def __init__(self,input_size,hidden_dim,num_layers,seqence_len,m,order):
        r"""
        Initilize neural network.
        Parameters
        ----------
        input_size
           description: the dimension of t functions
           type: int
        hidden_dim
           description: the dimension of ht of lstm
           type: int
        num_layers
           description: the number of layers of lstm
           type: int
        seqence_len
           description: the lenght of time series
           type: int

        var
           description: the setup of a VAR model
           type: class

        Returns
        -------
        initilized NN
        """
        super(DeepTVAR_mlp_onelayer,self).__init__()
        self.input_size=input_size
        self.hidden_dim=hidden_dim
        self.num_layers=num_layers
        self.seqence_len=seqence_len
        self.m = m
        self.order=order

        #self.lstm_for_var_params = nn.LSTM(self.input_size, self.hidden_dim, self.num_layers)
        #number of parameters of var-cov of innovations
        num_var_cov_parmas_of_innovs = (self.m * (self.m + 1)) / 2
        #number of coefficients A
        num_A=self.order*self.m*self.m
        #affine transformations for A and var-cov
        # self.linear1=nn.Linear(self.input_size,self.hidden_dim*2)
        # #self.linear2=nn.Linear(self.hidden_dim*3,self.hidden_dim*2)
        # self.linear2=nn.Linear(self.hidden_dim*2,self.hidden_dim)
        self.linear_affine_transf=nn.Linear(self.input_size,
                                 (int(num_A)+int(num_var_cov_parmas_of_innovs)))
        mp=self.m*self.order
        # num_of_params=(mp*(mp+1))/2
        # print('self.seqence_len')
        # print(self.seqence_len)
        num_of_initial_var_cov_params_of_initial_obs = (mp * (mp + 1)) / 2
        self.initial_var_cov_params_of_initial_obs = torch.nn.Parameter(torch.randn(int(num_of_initial_var_cov_params_of_initial_obs)))
        #self.Eulayer=nn.ELU()
        self.relu=nn.ReLU()


        #
        # self.h0=torch.nn.Parameter(torch.randn(self.num_layers, 1, self.hidden_dim))
        # self.c0 = torch.nn.Parameter(torch.randn(self.num_layers, 1, self.hidden_dim))



    def forward(self,inputs):

        #output_h, (hn1, cn1) = self.lstm_for_var_params(inputs,(self.h0,self.c0))
        #output_h, (hn1, cn1) = self.lstm_for_var_params(inputs)
        # output_h = self.linear1(inputs)
        # # print('output_h1')
        # # print(output_h.shape)
        # #output_h= F.sigmoid(output_h)
        # #output_h=self.Eulayer(output_h)
        # output_h=self.relu(output_h)
        # # print('output_h2')
        # # print(output_h.shape)
        # output_h = self.linear2(output_h)
        # output_h=self.relu(output_h)
        # output_h = self.linear3(output_h)
        # output_h=self.relu(output_h)
        # print('output_h3')
        # print(output_h.shape)
        #output_h=self.Eulayer(output_h)
        # print('output_h4')
        # print(output_h.shape)
        #output_h= F.tanh(output_h)
        #output_h= F.relu(output_h)
        # print('hn1')
        # print(hn1.shape)
        # print('cn1')
        # print(cn1.shape)

        num_var_cov_parmas_of_innovs = (self.m * (self.m + 1)) / 2
        num_A=self.order*self.m*self.m
        #curve:shape:(#seq_len,batch,num_of_A_and_var)
        #curve = self.linear_affine_transf(output_h)
        curve = self.linear_affine_transf(inputs)
        A_coeffs_of_VAR_p=curve[:,:,0:int(num_A)]
        lower_traig_parms=curve[:,:,int(num_A):int(num_A+num_var_cov_parmas_of_innovs)]
        seq_len=inputs.shape[0]
        #inputs_2=torch.zeros(1, 1)
        #v_covariance_coeffs=self.initial_variance_model(inputs_2)
        initial_var_cov_params_of_initial_obs = self.initial_var_cov_params_of_initial_obs
        # h0=self.h0
        # c0=self.c0
        return A_coeffs_of_VAR_p,lower_traig_parms,initial_var_cov_params_of_initial_obs


class DeepTVAR_mlp_many(torch.nn.Module):
    def __init__(self,input_size,hidden_dim,num_layers,seqence_len,m,order):
        r"""
        Initilize neural network.
        Parameters
        ----------
        input_size
           description: the dimension of t functions
           type: int
        hidden_dim
           description: the dimension of ht of lstm
           type: int
        num_layers
           description: the number of layers of lstm
           type: int
        seqence_len
           description: the lenght of time series
           type: int

        var
           description: the setup of a VAR model
           type: class

        Returns
        -------
        initilized NN
        """
        super(DeepTVAR_mlp_many,self).__init__()
        self.input_size=input_size
        self.hidden_dim=hidden_dim
        self.num_layers=num_layers
        self.seqence_len=seqence_len
        self.m = m
        self.order=order

        #self.lstm_for_var_params = nn.LSTM(self.input_size, self.hidden_dim, self.num_layers)
        #number of parameters of var-cov of innovations
        num_var_cov_parmas_of_innovs = (self.m * (self.m + 1)) / 2
        #number of coefficients A
        num_A=self.order*self.m*self.m
        #affine transformations for A and var-cov
        self.fc1 = nn.Linear(self.input_size, self.hidden_dim)
        self.fc2 = nn.Linear( self.hidden_dim,  self.hidden_dim)
        self.fc3 = nn.Linear( self.hidden_dim,  self.hidden_dim)
        #self.fc4 = nn.Linear(units, units)
        self.linear_affine_transf=nn.Linear( self.hidden_dim,
                                 (int(num_A)+int(num_var_cov_parmas_of_innovs)))
        mp=self.m*self.order
        # num_of_params=(mp*(mp+1))/2
        # print('self.seqence_len')
        # print(self.seqence_len)
        num_of_initial_var_cov_params_of_initial_obs = (mp * (mp + 1)) / 2
        self.initial_var_cov_params_of_initial_obs = torch.nn.Parameter(torch.randn(int(num_of_initial_var_cov_params_of_initial_obs)))
        #self.Eulayer=nn.ELU()
        #self.relu=nn.ReLU()



        #
        # self.h0=torch.nn.Parameter(torch.randn(self.num_layers, 1, self.hidden_dim))
        # self.c0 = torch.nn.Parameter(torch.randn(self.num_layers, 1, self.hidden_dim))



    def forward(self,inputs):

        #output_h, (hn1, cn1) = self.lstm_for_var_params(inputs,(self.h0,self.c0))
        #output_h, (hn1, cn1) = self.lstm_for_var_params(inputs)
        # output_h = self.linear1(inputs)
        # # print('output_h1')
        # # print(output_h.shape)
        # #output_h= F.sigmoid(output_h)
        # #output_h=self.Eulayer(output_h)
        # output_h=self.relu(output_h)
        # # print('output_h2')
        # # print(output_h.shape)
        # output_h = self.linear2(output_h)
        # output_h=self.relu(output_h)
        # output_h = self.linear3(output_h)
        # output_h=self.relu(output_h)
        # print('output_h3')
        # print(output_h.shape)
        #output_h=self.Eulayer(output_h)
        # print('output_h4')
        # print(output_h.shape)
        #output_h= F.tanh(output_h)
        #output_h= F.relu(output_h)
        # print('hn1')
        # print(hn1.shape)
        # print('cn1')
        # print(cn1.shape)
        m = nn.Tanh()

        num_var_cov_parmas_of_innovs = (self.m * (self.m + 1)) / 2
        num_A=self.order*self.m*self.m
        #curve:shape:(#seq_len,batch,num_of_A_and_var)
        #curve = self.linear_affine_transf(output_h)
        #x = F.relu(self.fc1(inputs))
        x = m(self.fc1(inputs))
        x = m(self.fc2(x))
        x = m(self.fc3(x))
        #x = F.relu(self.fc4(x))
        curve = self.linear_affine_transf(x)
        A_coeffs_of_VAR_p=curve[:,:,0:int(num_A)]
        lower_traig_parms=curve[:,:,int(num_A):int(num_A+num_var_cov_parmas_of_innovs)]
        seq_len=inputs.shape[0]
        #inputs_2=torch.zeros(1, 1)
        #v_covariance_coeffs=self.initial_variance_model(inputs_2)
        initial_var_cov_params_of_initial_obs = self.initial_var_cov_params_of_initial_obs
        # h0=self.h0
        # c0=self.c0
        return A_coeffs_of_VAR_p,lower_traig_parms,initial_var_cov_params_of_initial_obs



class DeepTVAR_mlp(torch.nn.Module):
    def __init__(self,input_size,hidden_dim,num_layers,seqence_len,m,order):
        r"""
        Initilize neural network.
        Parameters
        ----------
        input_size
           description: the dimension of t functions
           type: int
        hidden_dim
           description: the dimension of ht of lstm
           type: int
        num_layers
           description: the number of layers of lstm
           type: int
        seqence_len
           description: the lenght of time series
           type: int

        var
           description: the setup of a VAR model
           type: class

        Returns
        -------
        initilized NN
        """
        super(DeepTVAR_mlp,self).__init__()
        self.input_size=input_size
        self.hidden_dim=hidden_dim
        self.num_layers=num_layers
        self.seqence_len=seqence_len
        self.m = m
        self.order=order

        #self.lstm_for_var_params = nn.LSTM(self.input_size, self.hidden_dim, self.num_layers)
        #number of parameters of var-cov of innovations
        num_var_cov_parmas_of_innovs = (self.m * (self.m + 1)) / 2
        #number of coefficients A
        num_A=self.order*self.m*self.m
        #affine transformations for A and var-cov
        self.linear1=nn.Linear(self.input_size,self.hidden_dim*2)
        #self.linear2=nn.Linear(self.hidden_dim*3,self.hidden_dim*2)
        self.linear2=nn.Linear(self.hidden_dim*2,self.hidden_dim)
        self.linear_affine_transf=nn.Linear(self.hidden_dim,
                                 (int(num_A)+int(num_var_cov_parmas_of_innovs)))
        mp=self.m*self.order
        # num_of_params=(mp*(mp+1))/2
        # print('self.seqence_len')
        # print(self.seqence_len)
        num_of_initial_var_cov_params_of_initial_obs = (mp * (mp + 1)) / 2
        self.initial_var_cov_params_of_initial_obs = torch.nn.Parameter(torch.randn(int(num_of_initial_var_cov_params_of_initial_obs)))
        #self.Eulayer=nn.ELU()
        self.relu=nn.ReLU()


        #
        # self.h0=torch.nn.Parameter(torch.randn(self.num_layers, 1, self.hidden_dim))
        # self.c0 = torch.nn.Parameter(torch.randn(self.num_layers, 1, self.hidden_dim))



    def forward(self,inputs):

        #output_h, (hn1, cn1) = self.lstm_for_var_params(inputs,(self.h0,self.c0))
        #output_h, (hn1, cn1) = self.lstm_for_var_params(inputs)
        output_h = self.linear1(inputs)
        # print('output_h1')
        # print(output_h.shape)
        #output_h= F.sigmoid(output_h)
        #output_h=self.Eulayer(output_h)
        output_h=self.relu(output_h)
        # print('output_h2')
        # print(output_h.shape)
        output_h = self.linear2(output_h)
        output_h=self.relu(output_h)
        # output_h = self.linear3(output_h)
        # output_h=self.relu(output_h)
        # print('output_h3')
        # print(output_h.shape)
        #output_h=self.Eulayer(output_h)
        # print('output_h4')
        # print(output_h.shape)
        #output_h= F.tanh(output_h)
        #output_h= F.relu(output_h)
        # print('hn1')
        # print(hn1.shape)
        # print('cn1')
        # print(cn1.shape)

        num_var_cov_parmas_of_innovs = (self.m * (self.m + 1)) / 2
        num_A=self.order*self.m*self.m
        #curve:shape:(#seq_len,batch,num_of_A_and_var)
        curve = self.linear_affine_transf(output_h)
        A_coeffs_of_VAR_p=curve[:,:,0:int(num_A)]
        lower_traig_parms=curve[:,:,int(num_A):int(num_A+num_var_cov_parmas_of_innovs)]
        seq_len=inputs.shape[0]
        #inputs_2=torch.zeros(1, 1)
        #v_covariance_coeffs=self.initial_variance_model(inputs_2)
        initial_var_cov_params_of_initial_obs = self.initial_var_cov_params_of_initial_obs
        # h0=self.h0
        # c0=self.c0
        return A_coeffs_of_VAR_p,lower_traig_parms,initial_var_cov_params_of_initial_obs




class DeepTVAR_2layerslstm(torch.nn.Module):
    def __init__(self,input_size,hidden_dim,num_layers,seqence_len,m,order):
        r"""
        Initilize neural network.
        Parameters
        ----------
        input_size
           description: the dimension of t functions
           type: int
        hidden_dim
           description: the dimension of ht of lstm
           type: int
        num_layers
           description: the number of layers of lstm
           type: int
        seqence_len
           description: the lenght of time series
           type: int

        var
           description: the setup of a VAR model
           type: class

        Returns
        -------
        initilized NN
        """
        super(DeepTVAR_2layerslstm,self).__init__()
        self.input_size=input_size
        self.hidden_dim=hidden_dim
        self.num_layers=num_layers
        self.seqence_len=seqence_len
        self.m = m
        self.order=order

        self.lstm_for_var_params1 = nn.LSTM(self.input_size, self.hidden_dim, self.num_layers)
        self.lstm_for_var_params2 = nn.LSTM(self.hidden_dim, self.hidden_dim, self.num_layers)
        #number of parameters of var-cov of innovations
        num_var_cov_parmas_of_innovs = (self.m * (self.m + 1)) / 2
        #number of coefficients A
        num_A=self.order*self.m*self.m
        #affine transformations for A and var-cov
        self.linear_affine_transf=nn.Linear(self.hidden_dim,
                                 (int(num_A)+int(num_var_cov_parmas_of_innovs)))
        mp=self.m*self.order
        # num_of_params=(mp*(mp+1))/2
        # print('self.seqence_len')
        # print(self.seqence_len)
        num_of_initial_var_cov_params_of_initial_obs = (mp * (mp + 1)) / 2
        self.initial_var_cov_params_of_initial_obs = torch.nn.Parameter(torch.randn(int(num_of_initial_var_cov_params_of_initial_obs)))
        #
        # self.h0=torch.nn.Parameter(torch.randn(self.num_layers, 1, self.hidden_dim))
        # self.c0 = torch.nn.Parameter(torch.randn(self.num_layers, 1, self.hidden_dim))



    def forward(self,inputs):

        #output_h, (hn1, cn1) = self.lstm_for_var_params(inputs,(self.h0,self.c0))
        output_h1, (hn1, cn1) = self.lstm_for_var_params1(inputs)
        output_h2, (hn2, cn2) = self.lstm_for_var_params2(output_h1)
        # print('hn1')
        # print(hn1.shape)
        # print('cn1')
        # print(cn1.shape)

        num_var_cov_parmas_of_innovs = (self.m * (self.m + 1)) / 2
        num_A=self.order*self.m*self.m
        #curve:shape:(#seq_len,batch,num_of_A_and_var)
        curve = self.linear_affine_transf(output_h2)
        A_coeffs_of_VAR_p=curve[:,:,0:int(num_A)]
        lower_traig_parms=curve[:,:,int(num_A):int(num_A+num_var_cov_parmas_of_innovs)]
        seq_len=inputs.shape[0]
        #inputs_2=torch.zeros(1, 1)
        #v_covariance_coeffs=self.initial_variance_model(inputs_2)
        initial_var_cov_params_of_initial_obs = self.initial_var_cov_params_of_initial_obs
        # h0=self.h0
        # c0=self.c0
        return A_coeffs_of_VAR_p,lower_traig_parms,initial_var_cov_params_of_initial_obs



class DeepTVAR_bilstm(torch.nn.Module):
    def __init__(self,input_size,hidden_dim,num_layers,seqence_len,m,order):
        r"""
        Initilize neural network.
        Parameters
        ----------
        input_size
           description: the dimension of t functions
           type: int
        hidden_dim
           description: the dimension of ht of lstm
           type: int
        num_layers
           description: the number of layers of lstm
           type: int
        seqence_len
           description: the lenght of time series
           type: int

        var
           description: the setup of a VAR model
           type: class

        Returns
        -------
        initilized NN
        """
        super(DeepTVAR_bilstm,self).__init__()
        self.input_size=input_size
        self.hidden_dim=hidden_dim
        self.num_layers=num_layers
        self.seqence_len=seqence_len
        self.m = m
        self.order=order

        self.lstm_for_var_params = nn.LSTM(self.input_size, self.hidden_dim, self.num_layers, bidirectional=True)
        #number of parameters of var-cov of innovations
        num_var_cov_parmas_of_innovs = (self.m * (self.m + 1)) / 2
        #number of coefficients A
        num_A=self.order*self.m*self.m
        #affine transformations for A and var-cov
        #self.linear1=nn.Linear(self.hidden_dim*2,self.hidden_dim)
        # self.linear_affine_transf=nn.Linear(self.hidden_dim*2,
        #                          (int(num_A)+int(num_var_cov_parmas_of_innovs)))
        self.linear_affine_transf=nn.Linear(self.hidden_dim*2,
                                 (int(num_A)+int(num_var_cov_parmas_of_innovs)))
        mp=self.m*self.order
        # num_of_params=(mp*(mp+1))/2
        # print('self.seqence_len')
        # print(self.seqence_len)
        num_of_initial_var_cov_params_of_initial_obs = (mp * (mp + 1)) / 2
        self.initial_var_cov_params_of_initial_obs = torch.nn.Parameter(torch.randn(int(num_of_initial_var_cov_params_of_initial_obs)))
        #self.dropout = nn.Dropout(0.1)
        #
        # self.h0=torch.nn.Parameter(torch.randn(self.num_layers, 1, self.hidden_dim))
        # self.c0 = torch.nn.Parameter(torch.randn(self.num_layers, 1, self.hidden_dim))



    def forward(self,inputs):

        #output_h, (hn1, cn1) = self.lstm_for_var_params(inputs,(self.h0,self.c0))
        output_h, (hn1, cn1) = self.lstm_for_var_params(inputs)
        # print('hn1')
        # print(hn1.shape)
        # print('cn1')
        # print(cn1.shape)

        num_var_cov_parmas_of_innovs = (self.m * (self.m + 1)) / 2
        num_A=self.order*self.m*self.m
        #curve:shape:(#seq_len,batch,num_of_A_and_var)
        #output_h = self.dropout(output_h)
        #output_h=self.linear1(output_h)
        curve = self.linear_affine_transf(output_h)

        A_coeffs_of_VAR_p=curve[:,:,0:int(num_A)]
        lower_traig_parms=curve[:,:,int(num_A):int(num_A+num_var_cov_parmas_of_innovs)]
        seq_len=inputs.shape[0]
        #inputs_2=torch.zeros(1, 1)
        #v_covariance_coeffs=self.initial_variance_model(inputs_2)
        initial_var_cov_params_of_initial_obs = self.initial_var_cov_params_of_initial_obs
        # h0=self.h0
        # c0=self.c0
        return A_coeffs_of_VAR_p,lower_traig_parms,initial_var_cov_params_of_initial_obs



class DeepTVAR_bilstm_mlp(torch.nn.Module):
    def __init__(self,input_size,hidden_dim,num_layers,seqence_len,m,order):
        r"""
        Initilize neural network.
        Parameters
        ----------
        input_size
           description: the dimension of t functions
           type: int
        hidden_dim
           description: the dimension of ht of lstm
           type: int
        num_layers
           description: the number of layers of lstm
           type: int
        seqence_len
           description: the lenght of time series
           type: int

        var
           description: the setup of a VAR model
           type: class

        Returns
        -------
        initilized NN
        """
        super(DeepTVAR_bilstm_mlp,self).__init__()
        self.input_size=input_size
        self.hidden_dim=hidden_dim
        self.num_layers=num_layers
        self.seqence_len=seqence_len
        self.m = m
        self.order=order

        self.lstm_for_var_params = nn.LSTM(self.input_size, self.hidden_dim, self.num_layers, bidirectional=True)
        #number of parameters of var-cov of innovations
        num_var_cov_parmas_of_innovs = (self.m * (self.m + 1)) / 2
        #number of coefficients A
        num_A=self.order*self.m*self.m
        #affine transformations for A and var-cov
        #self.linear1=nn.Linear(self.hidden_dim*2,self.hidden_dim)
        # self.linear_affine_transf=nn.Linear(self.hidden_dim*2,
        #                          (int(num_A)+int(num_var_cov_parmas_of_innovs)))
        self.linear_affine_transf=nn.Linear(self.hidden_dim*2,
                                 (int(num_A)))
        self.linear1=nn.Linear(self.input_size,self.input_size*2)
        #self.linear2=nn.Linear(self.input_size*2,int(num_var_cov_parmas_of_innovs))
        mp=self.m*self.order
        # num_of_params=(mp*(mp+1))/2
        # print('self.seqence_len')
        # print(self.seqence_len)
        num_of_initial_var_cov_params_of_initial_obs = (mp * (mp + 1)) / 2
        self.initial_var_cov_params_of_initial_obs = torch.nn.Parameter(torch.randn(int(num_of_initial_var_cov_params_of_initial_obs)))
        #self.dropout = nn.Dropout(0.1)
        #
        # self.h0=torch.nn.Parameter(torch.randn(self.num_layers, 1, self.hidden_dim))
        # self.c0 = torch.nn.Parameter(torch.randn(self.num_layers, 1, self.hidden_dim))



    def forward(self,inputs):

        #output_h, (hn1, cn1) = self.lstm_for_var_params(inputs,(self.h0,self.c0))
        output_h, (hn1, cn1) = self.lstm_for_var_params(inputs)
        #output1=self.linear1(inputs)
        lower_traig_parms=self.linear1(inputs)
        # print('hn1')
        # print(hn1.shape)
        # print('cn1')
        # print(cn1.shape)

        num_var_cov_parmas_of_innovs = (self.m * (self.m + 1)) / 2
        num_A=self.order*self.m*self.m
        #curve:shape:(#seq_len,batch,num_of_A_and_var)
        #output_h = self.dropout(output_h)
        #output_h=self.linear1(output_h)
        curve = self.linear_affine_transf(output_h)

        A_coeffs_of_VAR_p=curve

        #lower_traig_parms=curve[:,:,int(num_A):int(num_A+num_var_cov_parmas_of_innovs)]
        seq_len=inputs.shape[0]
        #inputs_2=torch.zeros(1, 1)
        #v_covariance_coeffs=self.initial_variance_model(inputs_2)
        initial_var_cov_params_of_initial_obs = self.initial_var_cov_params_of_initial_obs
        # h0=self.h0
        # c0=self.c0
        return A_coeffs_of_VAR_p,lower_traig_parms,initial_var_cov_params_of_initial_obs



class DeepTVAR_bilstm_lstm(torch.nn.Module):
    def __init__(self,input_size,hidden_dim,num_layers,seqence_len,m,order):
        r"""
        Initilize neural network.
        Parameters
        ----------
        input_size
           description: the dimension of t functions
           type: int
        hidden_dim
           description: the dimension of ht of lstm
           type: int
        num_layers
           description: the number of layers of lstm
           type: int
        seqence_len
           description: the lenght of time series
           type: int

        var
           description: the setup of a VAR model
           type: class

        Returns
        -------
        initilized NN
        """
        super(DeepTVAR_bilstm_lstm,self).__init__()
        self.input_size=input_size
        self.hidden_dim=hidden_dim
        self.num_layers=num_layers
        self.seqence_len=seqence_len
        self.m = m
        self.order=order

        self.lstm_for_var_params = nn.LSTM(self.input_size, self.hidden_dim, self.num_layers, bidirectional=True)
        #number of parameters of var-cov of innovations
        num_var_cov_parmas_of_innovs = (self.m * (self.m + 1)) / 2
        #number of coefficients A
        num_A=self.order*self.m*self.m
        #affine transformations for A and var-cov
        #self.linear1=nn.Linear(self.hidden_dim*2,self.hidden_dim)
        # self.linear_affine_transf=nn.Linear(self.hidden_dim*2,
        #                          (int(num_A)+int(num_var_cov_parmas_of_innovs)))
        self.linear_affine_transf=nn.Linear(self.hidden_dim*2,
                                 (int(num_A)))
        self.lstm = nn.LSTM(self.input_size, 5, self.num_layers)
        self.linear1=nn.Linear(5,int(num_var_cov_parmas_of_innovs))

        #self.linear2=nn.Linear(self.input_size*2,int(num_var_cov_parmas_of_innovs))
        mp=self.m*self.order
        # num_of_params=(mp*(mp+1))/2
        # print('self.seqence_len')
        # print(self.seqence_len)
        num_of_initial_var_cov_params_of_initial_obs = (mp * (mp + 1)) / 2
        self.initial_var_cov_params_of_initial_obs = torch.nn.Parameter(torch.randn(int(num_of_initial_var_cov_params_of_initial_obs)))
        #self.dropout = nn.Dropout(0.1)
        #
        # self.h0=torch.nn.Parameter(torch.randn(self.num_layers, 1, self.hidden_dim))
        # self.c0 = torch.nn.Parameter(torch.randn(self.num_layers, 1, self.hidden_dim))



    def forward(self,inputs):

        #output_h, (hn1, cn1) = self.lstm_for_var_params(inputs,(self.h0,self.c0))
        output_h, (hn1, cn1) = self.lstm_for_var_params(inputs)
        #output1=self.linear1(inputs)
        output_2, (hn2, cn2) = self.lstm(inputs)
        lower_traig_parms=self.linear1(output_2)
        # print('hn1')
        # print(hn1.shape)
        # print('cn1')
        # print(cn1.shape)

        num_var_cov_parmas_of_innovs = (self.m * (self.m + 1)) / 2
        num_A=self.order*self.m*self.m
        #curve:shape:(#seq_len,batch,num_of_A_and_var)
        #output_h = self.dropout(output_h)
        #output_h=self.linear1(output_h)
        curve = self.linear_affine_transf(output_h)

        A_coeffs_of_VAR_p=curve

        #lower_traig_parms=curve[:,:,int(num_A):int(num_A+num_var_cov_parmas_of_innovs)]
        seq_len=inputs.shape[0]
        #inputs_2=torch.zeros(1, 1)
        #v_covariance_coeffs=self.initial_variance_model(inputs_2)
        initial_var_cov_params_of_initial_obs = self.initial_var_cov_params_of_initial_obs
        # h0=self.h0
        # c0=self.c0
        return A_coeffs_of_VAR_p,lower_traig_parms,initial_var_cov_params_of_initial_obs


class DeepTVAR_bilstm_bilstm(torch.nn.Module):
    def __init__(self,input_size,hidden_dim,num_layers,seqence_len,m,order):
        r"""
        Initilize neural network.
        Parameters
        ----------
        input_size
           description: the dimension of t functions
           type: int
        hidden_dim
           description: the dimension of ht of lstm
           type: int
        num_layers
           description: the number of layers of lstm
           type: int
        seqence_len
           description: the lenght of time series
           type: int

        var
           description: the setup of a VAR model
           type: class

        Returns
        -------
        initilized NN
        """
        super(DeepTVAR_bilstm_bilstm,self).__init__()
        self.input_size=input_size
        self.hidden_dim=hidden_dim
        self.num_layers=num_layers
        self.seqence_len=seqence_len
        self.m = m
        self.order=order

        self.lstm_for_var_params = nn.LSTM(self.input_size, self.hidden_dim, self.num_layers, bidirectional=True)
        #number of parameters of var-cov of innovations
        num_var_cov_parmas_of_innovs = (self.m * (self.m + 1)) / 2
        #number of coefficients A
        num_A=self.order*self.m*self.m
        #affine transformations for A and var-cov
        #self.linear1=nn.Linear(self.hidden_dim*2,self.hidden_dim)
        # self.linear_affine_transf=nn.Linear(self.hidden_dim*2,
        #                          (int(num_A)+int(num_var_cov_parmas_of_innovs)))
        self.linear_affine_transf=nn.Linear(self.hidden_dim*2,
                                 (int(num_A)))
        self.lstm = nn.LSTM(self.input_size, 6, self.num_layers,bidirectional=True)
        self.linear1=nn.Linear(12,int(num_var_cov_parmas_of_innovs))

        #self.linear2=nn.Linear(self.input_size*2,int(num_var_cov_parmas_of_innovs))
        mp=self.m*self.order
        # num_of_params=(mp*(mp+1))/2
        # print('self.seqence_len')
        # print(self.seqence_len)
        num_of_initial_var_cov_params_of_initial_obs = (mp * (mp + 1)) / 2
        self.initial_var_cov_params_of_initial_obs = torch.nn.Parameter(torch.randn(int(num_of_initial_var_cov_params_of_initial_obs)))
        #self.dropout = nn.Dropout(0.1)
        #
        # self.h0=torch.nn.Parameter(torch.randn(self.num_layers, 1, self.hidden_dim))
        # self.c0 = torch.nn.Parameter(torch.randn(self.num_layers, 1, self.hidden_dim))



    def forward(self,inputs):

        #output_h, (hn1, cn1) = self.lstm_for_var_params(inputs,(self.h0,self.c0))
        output_h, (hn1, cn1) = self.lstm_for_var_params(inputs)
        #output1=self.linear1(inputs)
        output_2, (hn2, cn2) = self.lstm(inputs)
        lower_traig_parms=self.linear1(output_2)
        # print('hn1')
        # print(hn1.shape)
        # print('cn1')
        # print(cn1.shape)

        num_var_cov_parmas_of_innovs = (self.m * (self.m + 1)) / 2
        num_A=self.order*self.m*self.m
        #curve:shape:(#seq_len,batch,num_of_A_and_var)
        #output_h = self.dropout(output_h)
        #output_h=self.linear1(output_h)
        curve = self.linear_affine_transf(output_h)

        A_coeffs_of_VAR_p=curve

        #lower_traig_parms=curve[:,:,int(num_A):int(num_A+num_var_cov_parmas_of_innovs)]
        seq_len=inputs.shape[0]
        #inputs_2=torch.zeros(1, 1)
        #v_covariance_coeffs=self.initial_variance_model(inputs_2)
        initial_var_cov_params_of_initial_obs = self.initial_var_cov_params_of_initial_obs
        # h0=self.h0
        # c0=self.c0
        return A_coeffs_of_VAR_p,lower_traig_parms,initial_var_cov_params_of_initial_obs



class DeepTVAR_lstm_lstm(torch.nn.Module):
    def __init__(self,input_size,hidden_dim,num_layers,seqence_len,m,order):
        r"""
        Initilize neural network.
        Parameters
        ----------
        input_size
           description: the dimension of t functions
           type: int
        hidden_dim
           description: the dimension of ht of lstm
           type: int
        num_layers
           description: the number of layers of lstm
           type: int
        seqence_len
           description: the lenght of time series
           type: int

        var
           description: the setup of a VAR model
           type: class

        Returns
        -------
        initilized NN
        """
        super(DeepTVAR_lstm_lstm,self).__init__()
        self.input_size=input_size
        self.hidden_dim=hidden_dim
        self.num_layers=num_layers
        self.seqence_len=seqence_len
        self.m = m
        self.order=order

        self.lstm_for_var_params = nn.LSTM(self.input_size, self.hidden_dim, self.num_layers)
        #number of parameters of var-cov of innovations
        num_var_cov_parmas_of_innovs = (self.m * (self.m + 1)) / 2
        #number of coefficients A
        num_A=self.order*self.m*self.m
        #affine transformations for A and var-cov
        #self.linear1=nn.Linear(self.hidden_dim*2,self.hidden_dim)
        # self.linear_affine_transf=nn.Linear(self.hidden_dim*2,
        #                          (int(num_A)+int(num_var_cov_parmas_of_innovs)))
        self.linear_affine_transf=nn.Linear(self.hidden_dim,
                                 (int(num_A)))
        self.lstm = nn.LSTM(self.input_size, 6, self.num_layers)
        self.linear1=nn.Linear(6,int(num_var_cov_parmas_of_innovs))

        #self.linear2=nn.Linear(self.input_size*2,int(num_var_cov_parmas_of_innovs))
        mp=self.m*self.order
        # num_of_params=(mp*(mp+1))/2
        # print('self.seqence_len')
        # print(self.seqence_len)
        num_of_initial_var_cov_params_of_initial_obs = (mp * (mp + 1)) / 2
        self.initial_var_cov_params_of_initial_obs = torch.nn.Parameter(torch.randn(int(num_of_initial_var_cov_params_of_initial_obs)))
        #self.dropout = nn.Dropout(0.1)
        #
        # self.h0=torch.nn.Parameter(torch.randn(self.num_layers, 1, self.hidden_dim))
        # self.c0 = torch.nn.Parameter(torch.randn(self.num_layers, 1, self.hidden_dim))



    def forward(self,inputs):

        #output_h, (hn1, cn1) = self.lstm_for_var_params(inputs,(self.h0,self.c0))
        output_h, (hn1, cn1) = self.lstm_for_var_params(inputs)
        #output1=self.linear1(inputs)
        output_2, (hn2, cn2) = self.lstm(inputs)
        lower_traig_parms=self.linear1(output_2)
        # print('hn1')
        # print(hn1.shape)
        # print('cn1')
        # print(cn1.shape)

        num_var_cov_parmas_of_innovs = (self.m * (self.m + 1)) / 2
        num_A=self.order*self.m*self.m
        #curve:shape:(#seq_len,batch,num_of_A_and_var)
        #output_h = self.dropout(output_h)
        #output_h=self.linear1(output_h)
        curve = self.linear_affine_transf(output_h)

        A_coeffs_of_VAR_p=curve

        #lower_traig_parms=curve[:,:,int(num_A):int(num_A+num_var_cov_parmas_of_innovs)]
        seq_len=inputs.shape[0]
        #inputs_2=torch.zeros(1, 1)
        #v_covariance_coeffs=self.initial_variance_model(inputs_2)
        initial_var_cov_params_of_initial_obs = self.initial_var_cov_params_of_initial_obs
        # h0=self.h0
        # c0=self.c0
        return A_coeffs_of_VAR_p,lower_traig_parms,initial_var_cov_params_of_initial_obs


class DeepTVAR_bilstm_seperate(torch.nn.Module):
    def __init__(self,input_size,hidden_dim,num_layers,seqence_len,m,order):
        r"""
        Initilize neural network.
        Parameters
        ----------
        input_size
           description: the dimension of t functions
           type: int
        hidden_dim
           description: the dimension of ht of lstm
           type: int
        num_layers
           description: the number of layers of lstm
           type: int
        seqence_len
           description: the lenght of time series
           type: int

        var
           description: the setup of a VAR model
           type: class

        Returns
        -------
        initilized NN
        """
        super(DeepTVAR_bilstm_seperate,self).__init__()
        self.input_size=input_size
        self.hidden_dim=hidden_dim
        self.num_layers=num_layers
        self.seqence_len=seqence_len
        self.m = m
        self.order=order

        self.lstm_for_var_params = nn.LSTM(self.input_size, self.hidden_dim, self.num_layers, bidirectional=True)
        #number of parameters of var-cov of innovations
        num_var_cov_parmas_of_innovs = (self.m * (self.m + 1)) / 2
        #number of coefficients A
        num_A=self.order*self.m*self.m
        #affine transformations for A and var-cov
        self.linear_affine_transf=nn.Linear(self.hidden_dim*2,int(num_A))
        self.linear_affine_transf2=nn.Linear(self.hidden_dim*2,int(num_var_cov_parmas_of_innovs))
        mp=self.m*self.order
        # num_of_params=(mp*(mp+1))/2
        # print('self.seqence_len')
        # print(self.seqence_len)
        num_of_initial_var_cov_params_of_initial_obs = (mp * (mp + 1)) / 2
        self.initial_var_cov_params_of_initial_obs = torch.nn.Parameter(torch.randn(int(num_of_initial_var_cov_params_of_initial_obs)))
        #
        # self.h0=torch.nn.Parameter(torch.randn(self.num_layers, 1, self.hidden_dim))
        # self.c0 = torch.nn.Parameter(torch.randn(self.num_layers, 1, self.hidden_dim))



    def forward(self,inputs):

        #output_h, (hn1, cn1) = self.lstm_for_var_params(inputs,(self.h0,self.c0))
        output_h, (hn1, cn1) = self.lstm_for_var_params(inputs)
        # print('hn1')
        # print(hn1.shape)
        # print('cn1')
        # print(cn1.shape)

        num_var_cov_parmas_of_innovs = (self.m * (self.m + 1)) / 2
        num_A=self.order*self.m*self.m
        #curve:shape:(#seq_len,batch,num_of_A_and_var)
        A_coeffs_of_VAR_p = self.linear_affine_transf(output_h)
        #A_coeffs_of_VAR_p=curve[:,:,0:int(num_A)]
        #lower_traig_parms=curve[:,:,int(num_A):int(num_A+num_var_cov_parmas_of_innovs)]
        lower_traig_parms=self.linear_affine_transf2(output_h)
        seq_len=inputs.shape[0]
        #inputs_2=torch.zeros(1, 1)
        #v_covariance_coeffs=self.initial_variance_model(inputs_2)
        initial_var_cov_params_of_initial_obs = self.initial_var_cov_params_of_initial_obs
        # h0=self.h0
        # c0=self.c0
        return A_coeffs_of_VAR_p,lower_traig_parms,initial_var_cov_params_of_initial_obs





class DeepTVAR_lstm_padding(torch.nn.Module):
    def __init__(self,input_size,hidden_dim,num_layers,seqence_len,m,order):
        r"""
        Initilize neural network.
        Parameters
        ----------
        input_size
           description: the dimension of t functions
           type: int
        hidden_dim
           description: the dimension of ht of lstm
           type: int
        num_layers
           description: the number of layers of lstm
           type: int
        seqence_len
           description: the lenght of time series
           type: int

        var
           description: the setup of a VAR model
           type: class

        Returns
        -------
        initilized NN
        """
        super(DeepTVAR_lstm_padding,self).__init__()
        self.input_size=input_size
        self.hidden_dim=hidden_dim
        self.num_layers=num_layers
        self.seqence_len=seqence_len+10
        self.m = m
        self.order=order

        self.lstm_for_var_params = nn.LSTM(self.input_size, self.hidden_dim, self.num_layers)
        #number of parameters of var-cov of innovations
        num_var_cov_parmas_of_innovs = (self.m * (self.m + 1)) / 2
        #number of coefficients A
        num_A=self.order*self.m*self.m
        #affine transformations for A and var-cov
        self.linear_affine_transf=nn.Linear(self.hidden_dim,
                                 (int(num_A)+int(num_var_cov_parmas_of_innovs)))
        mp=self.m*self.order
        # num_of_params=(mp*(mp+1))/2
        # print('self.seqence_len')
        # print(self.seqence_len)
        num_of_initial_var_cov_params_of_initial_obs = (mp * (mp + 1)) / 2
        self.initial_var_cov_params_of_initial_obs = torch.nn.Parameter(torch.randn(int(num_of_initial_var_cov_params_of_initial_obs)))
        #
        # self.h0=torch.nn.Parameter(torch.randn(self.num_layers, 1, self.hidden_dim))
        # self.c0 = torch.nn.Parameter(torch.randn(self.num_layers, 1, self.hidden_dim))



    def forward(self,inputs):

        #output_h, (hn1, cn1) = self.lstm_for_var_params(inputs,(self.h0,self.c0))
        output_h, (hn1, cn1) = self.lstm_for_var_params(inputs)
        # print('hn1')
        # print(hn1.shape)
        # print('cn1')
        # print(cn1.shape)

        num_var_cov_parmas_of_innovs = (self.m * (self.m + 1)) / 2
        num_A=self.order*self.m*self.m
        #curve:shape:(#seq_len,batch,num_of_A_and_var)
        curve = self.linear_affine_transf(output_h)
        A_coeffs_of_VAR_p=curve[10:,:,0:int(num_A)]
        lower_traig_parms=curve[10:,:,int(num_A):int(num_A+num_var_cov_parmas_of_innovs)]
        seq_len=inputs.shape[0]
        #inputs_2=torch.zeros(1, 1)
        #v_covariance_coeffs=self.initial_variance_model(inputs_2)
        initial_var_cov_params_of_initial_obs = self.initial_var_cov_params_of_initial_obs
        # h0=self.h0
        # c0=self.c0
        return A_coeffs_of_VAR_p,lower_traig_parms,initial_var_cov_params_of_initial_obs





class DeepTVAR_birnn(torch.nn.Module):
    def __init__(self,input_size,hidden_dim,num_layers,seqence_len,m,order):
        r"""
        Initilize neural network.
        Parameters
        ----------
        input_size
           description: the dimension of t functions
           type: int
        hidden_dim
           description: the dimension of ht of lstm
           type: int
        num_layers
           description: the number of layers of lstm
           type: int
        seqence_len
           description: the lenght of time series
           type: int

        var
           description: the setup of a VAR model
           type: class

        Returns
        -------
        initilized NN
        """
        super(DeepTVAR_birnn,self).__init__()
        self.input_size=input_size
        self.hidden_dim=hidden_dim
        self.num_layers=num_layers
        self.seqence_len=seqence_len
        self.m = m
        self.order=order

        self.lstm_for_var_params = nn.RNN(self.input_size, self.hidden_dim, self.num_layers,bidirectional=True)
        #number of parameters of var-cov of innovations
        num_var_cov_parmas_of_innovs = (self.m * (self.m + 1)) / 2
        #number of coefficients A
        num_A=self.order*self.m*self.m
        #affine transformations for A and var-cov
        self.linear_affine_transf=nn.Linear(2*self.hidden_dim,
                                 (int(num_A)+int(num_var_cov_parmas_of_innovs)))
        mp=self.m*self.order
        # num_of_params=(mp*(mp+1))/2
        # print('self.seqence_len')
        # print(self.seqence_len)
        num_of_initial_var_cov_params_of_initial_obs = (mp * (mp + 1)) / 2
        self.initial_var_cov_params_of_initial_obs = torch.nn.Parameter(torch.randn(int(num_of_initial_var_cov_params_of_initial_obs)))
        #
        # self.h0=torch.nn.Parameter(torch.randn(self.num_layers, 1, self.hidden_dim))
        # self.c0 = torch.nn.Parameter(torch.randn(self.num_layers, 1, self.hidden_dim))



    def forward(self,inputs):

        #output_h, (hn1, cn1) = self.lstm_for_var_params(inputs,(self.h0,self.c0))
        output_h, hn1 = self.lstm_for_var_params(inputs)
        print('output_h')
        print(output_h.shape)
        # print('cn1')
        # print(cn1.shape)

        num_var_cov_parmas_of_innovs = (self.m * (self.m + 1)) / 2
        num_A=self.order*self.m*self.m
        #curve:shape:(#seq_len,batch,num_of_A_and_var)
        curve = self.linear_affine_transf(output_h)
        A_coeffs_of_VAR_p=curve[:,:,0:int(num_A)]
        lower_traig_parms=curve[:,:,int(num_A):int(num_A+num_var_cov_parmas_of_innovs)]
        seq_len=inputs.shape[0]
        #inputs_2=torch.zeros(1, 1)
        #v_covariance_coeffs=self.initial_variance_model(inputs_2)
        initial_var_cov_params_of_initial_obs = self.initial_var_cov_params_of_initial_obs
        # h0=self.h0
        # c0=self.c0
        return A_coeffs_of_VAR_p,lower_traig_parms,initial_var_cov_params_of_initial_obs



class DeepTVAR_rnn(torch.nn.Module):
    def __init__(self,input_size,hidden_dim,num_layers,seqence_len,m,order):
        r"""
        Initilize neural network.
        Parameters
        ----------
        input_size
           description: the dimension of t functions
           type: int
        hidden_dim
           description: the dimension of ht of lstm
           type: int
        num_layers
           description: the number of layers of lstm
           type: int
        seqence_len
           description: the lenght of time series
           type: int

        var
           description: the setup of a VAR model
           type: class

        Returns
        -------
        initilized NN
        """
        super(DeepTVAR_rnn,self).__init__()
        self.input_size=input_size
        self.hidden_dim=hidden_dim
        self.num_layers=num_layers
        self.seqence_len=seqence_len
        self.m = m
        self.order=order

        self.lstm_for_var_params = nn.RNN(self.input_size, self.hidden_dim, self.num_layers)
        #number of parameters of var-cov of innovations
        num_var_cov_parmas_of_innovs = (self.m * (self.m + 1)) / 2
        #number of coefficients A
        num_A=self.order*self.m*self.m
        #affine transformations for A and var-cov
        self.linear_affine_transf=nn.Linear(self.hidden_dim,
                                 (int(num_A)+int(num_var_cov_parmas_of_innovs)))
        mp=self.m*self.order
        # num_of_params=(mp*(mp+1))/2
        # print('self.seqence_len')
        # print(self.seqence_len)
        num_of_initial_var_cov_params_of_initial_obs = (mp * (mp + 1)) / 2
        self.initial_var_cov_params_of_initial_obs = torch.nn.Parameter(torch.randn(int(num_of_initial_var_cov_params_of_initial_obs)))
        #
        # self.h0=torch.nn.Parameter(torch.randn(self.num_layers, 1, self.hidden_dim))
        # self.c0 = torch.nn.Parameter(torch.randn(self.num_layers, 1, self.hidden_dim))



    def forward(self,inputs):

        #output_h, (hn1, cn1) = self.lstm_for_var_params(inputs,(self.h0,self.c0))
        output_h, hn1 = self.lstm_for_var_params(inputs)
        # print('hn1')
        # print(hn1.shape)
        # print('cn1')
        # print(cn1.shape)

        num_var_cov_parmas_of_innovs = (self.m * (self.m + 1)) / 2
        num_A=self.order*self.m*self.m
        #curve:shape:(#seq_len,batch,num_of_A_and_var)
        curve = self.linear_affine_transf(output_h)
        A_coeffs_of_VAR_p=curve[:,:,0:int(num_A)]
        lower_traig_parms=curve[:,:,int(num_A):int(num_A+num_var_cov_parmas_of_innovs)]
        seq_len=inputs.shape[0]
        #inputs_2=torch.zeros(1, 1)
        #v_covariance_coeffs=self.initial_variance_model(inputs_2)
        initial_var_cov_params_of_initial_obs = self.initial_var_cov_params_of_initial_obs
        # h0=self.h0
        # c0=self.c0
        return A_coeffs_of_VAR_p,lower_traig_parms,initial_var_cov_params_of_initial_obs


class DeepTVAR_gru(torch.nn.Module):
    def __init__(self,input_size,hidden_dim,num_layers,seqence_len,m,order):
        r"""
        Initilize neural network.
        Parameters
        ----------
        input_size
           description: the dimension of t functions
           type: int
        hidden_dim
           description: the dimension of ht of lstm
           type: int
        num_layers
           description: the number of layers of lstm
           type: int
        seqence_len
           description: the lenght of time series
           type: int

        var
           description: the setup of a VAR model
           type: class

        Returns
        -------
        initilized NN
        """
        super(DeepTVAR_gru,self).__init__()
        self.input_size=input_size
        self.hidden_dim=hidden_dim
        self.num_layers=num_layers
        self.seqence_len=seqence_len
        self.m = m
        self.order=order

        self.lstm_for_var_params = nn.RNN(self.input_size, self.hidden_dim, self.num_layers)
        #number of parameters of var-cov of innovations
        num_var_cov_parmas_of_innovs = (self.m * (self.m + 1)) / 2
        #number of coefficients A
        num_A=self.order*self.m*self.m
        #affine transformations for A and var-cov
        self.linear_affine_transf=nn.Linear(self.hidden_dim,
                                 (int(num_A)+int(num_var_cov_parmas_of_innovs)))
        mp=self.m*self.order
        # num_of_params=(mp*(mp+1))/2
        # print('self.seqence_len')
        # print(self.seqence_len)
        num_of_initial_var_cov_params_of_initial_obs = (mp * (mp + 1)) / 2
        self.initial_var_cov_params_of_initial_obs = torch.nn.Parameter(torch.randn(int(num_of_initial_var_cov_params_of_initial_obs)))
        #
        # self.h0=torch.nn.Parameter(torch.randn(self.num_layers, 1, self.hidden_dim))
        # self.c0 = torch.nn.Parameter(torch.randn(self.num_layers, 1, self.hidden_dim))



    def forward(self,inputs):

        #output_h, (hn1, cn1) = self.lstm_for_var_params(inputs,(self.h0,self.c0))
        output_h, hn1 = self.lstm_for_var_params(inputs)
        # print('hn1')
        # print(hn1.shape)
        # print('cn1')
        # print(cn1.shape)

        num_var_cov_parmas_of_innovs = (self.m * (self.m + 1)) / 2
        num_A=self.order*self.m*self.m
        #curve:shape:(#seq_len,batch,num_of_A_and_var)
        curve = self.linear_affine_transf(output_h)
        A_coeffs_of_VAR_p=curve[:,:,0:int(num_A)]
        lower_traig_parms=curve[:,:,int(num_A):int(num_A+num_var_cov_parmas_of_innovs)]
        seq_len=inputs.shape[0]
        #inputs_2=torch.zeros(1, 1)
        #v_covariance_coeffs=self.initial_variance_model(inputs_2)
        initial_var_cov_params_of_initial_obs = self.initial_var_cov_params_of_initial_obs
        # h0=self.h0
        # c0=self.c0
        return A_coeffs_of_VAR_p,lower_traig_parms,initial_var_cov_params_of_initial_obs



class DeepTVAR(torch.nn.Module):
    def __init__(self,input_size,hidden_dim,num_layers,seqence_len,m,order):
        r"""
        Initilize neural network.
        Parameters
        ----------
        input_size
           description: the dimension of t functions
           type: int
        hidden_dim
           description: the dimension of ht of lstm
           type: int
        num_layers
           description: the number of layers of lstm
           type: int
        seqence_len
           description: the lenght of time series
           type: int

        var
           description: the setup of a VAR model
           type: class

        Returns
        -------
        initilized NN
        """
        super(DeepTVAR,self).__init__()
        self.input_size=input_size
        self.hidden_dim=hidden_dim
        self.num_layers=num_layers
        self.seqence_len=seqence_len
        self.m = m
        self.order=order

        #self.lstm_for_var_params = nn.RNN(self.input_size, self.hidden_dim, self.num_layers)
        #number of parameters of var-cov of innovations
        num_var_cov_parmas_of_innovs = (self.m * (self.m + 1)) / 2
        #number of coefficients A
        num_A=self.order*self.m*self.m
        #affine transformations for A and var-cov
        self.linear_affine_transf=nn.Linear(self.input_size,
                                 (int(num_A)+int(num_var_cov_parmas_of_innovs)))
        mp=self.m*self.order
        # num_of_params=(mp*(mp+1))/2
        # print('self.seqence_len')
        # print(self.seqence_len)
        num_of_initial_var_cov_params_of_initial_obs = (mp * (mp + 1)) / 2
        self.initial_var_cov_params_of_initial_obs = torch.nn.Parameter(torch.randn(int(num_of_initial_var_cov_params_of_initial_obs)))
        #
        # self.h0=torch.nn.Parameter(torch.randn(self.num_layers, 1, self.hidden_dim))
        # self.c0 = torch.nn.Parameter(torch.randn(self.num_layers, 1, self.hidden_dim))



    def forward(self,inputs):

        #output_h, (hn1, cn1) = self.lstm_for_var_params(inputs,(self.h0,self.c0))
        #output_h, hn1 = self.lstm_for_var_params(inputs)
        #

        num_var_cov_parmas_of_innovs = (self.m * (self.m + 1)) / 2
        num_A=self.order*self.m*self.m
        #curve:shape:(#seq_len,batch,num_of_A_and_var)
        curve = self.linear_affine_transf(inputs)
        A_coeffs_of_VAR_p=curve[:,:,0:int(num_A)]
        lower_traig_parms=curve[:,:,int(num_A):int(num_A+num_var_cov_parmas_of_innovs)]
        seq_len=inputs.shape[0]
        #inputs_2=torch.zeros(1, 1)
        #v_covariance_coeffs=self.initial_variance_model(inputs_2)
        initial_var_cov_params_of_initial_obs = self.initial_var_cov_params_of_initial_obs
        # h0=self.h0
        # c0=self.c0
        return A_coeffs_of_VAR_p,lower_traig_parms,initial_var_cov_params_of_initial_obs
