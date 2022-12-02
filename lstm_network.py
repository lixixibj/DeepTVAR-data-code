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


