1  #!/usr/bin/env python35
2  # -*- coding: utf-8 -*-
3  # @File  : custom_loss.py
4  # @Author: Xixi Li
5  # @Date  : 2019-12-02
6  # @Desc  :


import torch
#compute log-likehood for batch data
import pandas as pd
import numpy as np
import math







def compute_log_likelihood(
    target,
    A_coeffs_from_lstm,
    #lower_triang_params_form_lstm shape:(seq,batch_size=1,m*(m+1)/2)
    lower_triang_params_form_lstm,
    m,
    order,
var_cov_params_for_initial_obs
):
    r"""
        Compute -2*log-likelihood function using conditional density.
        Parameters
        ----------
        target
           description: the obsevations of time series
           type: array

        A_coeffs_from_lstm
           description: the initial A cofficients generated from lstm.
        type: tensor

        m
           description: the dimension of multivariate ts
           type: int

        order
           description: the order of VAR
           type: int

        var_cov_params_for_initial_obs
           description: the elmements of triangular matrix for generaing var-cov matrix for initial observations y_1, y_2,...,y_p.
           type: tensor


        Returns
        -------
        -2*log-likelihood
    """
    #print('tttttt')


    len_of_seq=target.shape[0]
    print('len_of_seq')
    print(len_of_seq)
    log_likelihood_temp=0
    #log_likelihood_temp=torch.tensor([0],dtype=torch.float64)

    #make var-cov matrix
    var_cov_matrix_for_initial_p_obs = make_var_cov_matrix_for_initial_obs(var_cov_params_for_initial_obs, m, order)
    #get first p obs
    first_p_obs=torch.zeros(order*m, 1)
    # coeff_parameters=torch.randn(m,m,order,len_of_seq)
    # var_cov_parameters=torch.randn(m,m,len_of_seq)
    # coeff_parameters=[]
    # var_cov_parameters=[]
    for i in range(order):
        b=i*m
        e=i*m+m
        first_p_obs[b:e,:]=target[i,:].reshape(m,1)

    log_likelihood_temp = log_likelihood_temp + torch.log(torch.det(var_cov_matrix_for_initial_p_obs)) + torch.mm(
        torch.mm(first_p_obs.t(), torch.inverse(var_cov_matrix_for_initial_p_obs)), first_p_obs)

    #calculate
    for t in range(order,len_of_seq):
        #calculate var-cov of innovations of var(p)
        var_cov_innovations_varp=make_var_cov_matrix_for_innovation_of_varp(lower_triang_params_form_lstm[t,0, :], m, order)

        # print('var_cov_innovations_varp')
        # print(var_cov_innovations_varp)
        #casual VAR coefficients
        lower_t_for_innovations_varp=get_lower_trang_m(lower_triang_params_form_lstm[t,0, :], m, order)
        #A_coeffs = A_coeffs_for_causal_VAR(A_coeffs_from_lstm[t,0,:], order, m,lower_t_for_innovations_varp)
        A_coeffs = A_coeffs_for_causal_VAR(A_coeffs_from_lstm[t,0,:], order, m,var_cov_innovations_varp)
        # print('A_coeffs')
        # print(A_coeffs)
        #A=put_A_coeff_together(A_coeffs, order).double()
        A=put_A_coeff_together(A_coeffs, order)
        #A.register_hook(print)
        # A.register_hook(save_grad('A'))
        # print(grads['A'])
        # print(grads['A'].shape)
        # var_cov_innovations_varp.register_hook(save_grad('var'))
        # print(grads['var'])
        # print(grads['var'].shape)
        # print('A11111111111')
        # print(A)
        lagged_obs=get_lagged_observations(target, t, order, m)

        error=target[t,:].reshape(m, 1)-torch.mm(A,lagged_obs)
        # print('error')
        # print(error)
        # print('torch.inverse(var_cov_innovations_varp)')
        # print(torch.inverse(var_cov_innovations_varp))
        # print(torch.mm(error.t(), torch.inverse(var_cov_innovations_varp)))
        log_likelihood_temp=log_likelihood_temp+torch.log(torch.det(var_cov_innovations_varp))+torch.mm(
            torch.mm(error.t(), torch.inverse(var_cov_innovations_varp)), error)

        #A_coeffs.retain_grad()
        #A.retain_grad()
        # A.retain_grad()
        # coeff_parameters.append(A_coeffs)
        # var_cov_innovations_varp.retain_grad()
        # var_cov_parameters.append(var_cov_innovations_varp)



    #return 0.5*(log_likelihood_temp+len_of_seq*torch.log(torch.tensor(math.pi))),coeff_parameters,var_cov_parameters
    return 0.5*(log_likelihood_temp+len_of_seq*torch.log(torch.tensor(math.pi)))









def check_causality(A_coeffs, m, order):
    A_coeffs_var1 = get_A_coeff_m_for_VAR_1(A_coeffs, m, order)
    L_complex = torch.linalg.eigvals(A_coeffs_var1)
    eigenvalues_num=L_complex.shape[0]
    # print('L_complex')
    # print(L_complex)
    # print(abs(L_complex))
    modul_all=abs(L_complex)
    re=True
    for i in range(eigenvalues_num):
        modul=modul_all[i]
        if modul>=1:
            re=False
    return re


def get_lagged_observations(target_y,current_t,order,m):
    lagged_list=[]
    for i in range(order):
        lagged_list.append(target_y[(current_t-i-1),:].reshape(m, 1))
    lagged_obs = torch.cat(lagged_list, dim=0)
    return lagged_obs




def put_A_coeff_together(inital_A_m,order):
    r"""
        Put A coefficients together
        Parameters
        ----------
        inital_A_m
           description: initial A coeffs
           type: tensor shape(m*m*p)
        m
           description: the dimension of ts
           type: int
        order
           description: the lag of VAR model
        type: int

        Returns
        -------
        coeffs matrix of state equation.
    """
    A_list=[]
    for c in range(order):
        A_list.append(inital_A_m[:,:,c])
    #concentrate
    A=torch.cat(A_list, dim=1)
    return A





def get_A_coeff_m_for_VAR_1(inital_A_m,m,order):
    r"""
        Put VAR(p) into VAR(1) and get coeffs matrix of state equation.
        Parameters
        ----------
        inital_A_m
           description: initial A coeffs
           type: tensor shape(m*m*p)
        m
           description: the dimension of ts
           type: int   
        order
           description: the lag of VAR model
        type: int 

        Returns
        -------
        coeffs matrix of state equation.
    """
    F_list=[]
    for c in range(order):
        F_list.append(inital_A_m[:,:,c])
    #concentrate
    F_temp1=torch.cat(F_list, dim=1)

    mp=m*order
    added_tensor= torch.eye((mp-m), mp)
    coffs_m=torch.cat((F_temp1, added_tensor), 0)
    return coffs_m




import math


def make_var_cov_of_innovations_var1(lower_triang_params_form_lstm,m,order):
    r"""
        Put VAR(p) into VAR(1) and get var-cov matrix of innovations of state equation.
        Parameters
        ----------
        lower_triang_params_form_lstm
           description: elements for lower triangular matrix for generating var-cov matrix
           type: tensor shape((m*(m+1)/2)，)
        m
           description: the dimension of ts
           type: int   
        order
           description: the lag of VAR model
        type: int 

        Returns
        -------
        var-cov matrix of innovations of state euqation
    """
    mp=m*order
    number_of_parms=m*(m+1)/2
    lower_t_matrix=torch.eye(m,m)
    count_temp=0
    for i in range(m):
        for j in range(i+1):
            lower_t_matrix[i,j]=lower_triang_params_form_lstm[count_temp]
            count_temp=count_temp+1

    var_cov_matrix = torch.mm(lower_t_matrix, lower_t_matrix.t())

    zeros_cols = torch.zeros([m, (mp - m)])
    #2.generate (mp-m)*mp matrix
    zeros_rows = torch.zeros([(mp - m),mp ])
    c = torch.cat((var_cov_matrix, zeros_cols), 1)
    var_cov_of_innovations_for_var1 = torch.cat((c, zeros_rows), 0)
    return (var_cov_of_innovations_for_var1)



def make_var_cov_matrix_for_innovation_of_varp(lower_triang_params_form_lstm,m,order):
    r"""
        Get var-cov matrix of innovations of VAR(p).
        Parameters
        ----------
        lower_triang_params_form_lstm
           description: elements for lower triangular matrix for generating var-cov matrix
           type: tensor shape((m*(m+1)/2)，)
        m
           description: the dimension of ts
           type: int   
        order
           description: the lag of VAR model
        type: int 

        Returns
        -------
        var-cov matrix of innovations of VAR(p)
    """
    #mp=m*order
    #number_of_parms=m*(m+1)/2
    lower_t_matrix=torch.eye(m,m)
    count_temp=0
    for i in range(m):
        for j in range(i+1):
            lower_t_matrix[i,j]=lower_triang_params_form_lstm[count_temp].clone()
            count_temp=count_temp+1
#diag
    var_cov_matrix = torch.mm(lower_t_matrix, lower_t_matrix.t())

    return (var_cov_matrix)



def make_var_cov_matrix_for_innovation_of_varp_three_decomp(lower_triang_params_form_lstm,m,order):
    r"""
        Get var-cov matrix of innovations of VAR(p).
        Parameters
        ----------
        lower_triang_params_form_lstm
           description: elements for lower triangular matrix for generating var-cov matrix
           type: tensor shape((m*(m+1)/2)，)
        m
           description: the dimension of ts
           type: int   
        order
           description: the lag of VAR model
        type: int 

        Returns
        -------
        var-cov matrix of innovations of VAR(p)
    """
    #mp=m*order
    #number_of_parms=m*(m+1)/2
    lower_t_matrix=torch.eye(m,m)
    count_temp=0
    for i in range(1,m):
        for j in range(i):
            lower_t_matrix[i,j]=lower_triang_params_form_lstm[count_temp].clone()
            count_temp=count_temp+1
    #
    diag_m=torch.eye(m,m)
    for i in range(m):
        diag_m[i,i]=lower_triang_params_form_lstm[count_temp].clone()*lower_triang_params_form_lstm[count_temp].clone()
        count_temp=count_temp+1




#diag
    var_cov_matrix = torch.mm(torch.mm(lower_t_matrix, diag_m),lower_t_matrix.t())

    return (var_cov_matrix)


def get_lower_trang_m(lower_triang_params_form_lstm,m,order):
    r"""
        Get var-cov matrix of innovations of VAR(p).
        Parameters
        ----------
        lower_triang_params_form_lstm
           description: elements for lower triangular matrix for generating var-cov matrix
           type: tensor shape((m*(m+1)/2)，)
        m
           description: the dimension of ts
           type: int   
        order
           description: the lag of VAR model
        type: int 

        Returns
        -------
        var-cov matrix of innovations of VAR(p)
    """
    #mp=m*order
    #number_of_parms=m*(m+1)/2
    lower_t_matrix=torch.eye(m,m)
    count_temp=0
    for i in range(m):
        for j in range(i+1):
            lower_t_matrix[i,j]=lower_triang_params_form_lstm[count_temp].clone()
            count_temp=count_temp+1
#diag
    #var_cov_matrix = torch.mm(lower_t_matrix, lower_t_matrix.t())

    return (lower_t_matrix)


import math


def make_var_cov_matrix_for_initial_obs(var_cov_params_for_initial_obs,m,order):
    r"""
        Get var-cov matrix of first p obs.
        Parameters
        ----------
        var_cov_params_for_initial_obs
           description: elements for lower triangular matrix for generating var-cov matrix 
           type: tensor shape(mp*(mp+1)/2)，)
        m
           description: the dimension of ts
           type: int   
        order
           description: the lag of VAR model
        type: int 

        Returns
        -------
        var-cov matrix of intial p obs.
    """
    mp=m*order
    upper_t_matrix=torch.eye(mp,mp)
    count_temp=0
    #make upper triangel matrix
    for i in range(mp):
        for j in range(mp-i):
            upper_t_matrix[i,i+j]=var_cov_params_for_initial_obs[count_temp].clone()
            count_temp=count_temp+1
    var_cov_m = torch.mm(upper_t_matrix.t(), upper_t_matrix)

    return (var_cov_m)









import math


def make_var_cov_matrix_for_initial_obs(var_cov_params_for_initial_obs,m,order):
    r"""
        Get var-cov matrix of first p obs.
        Parameters
        ----------
        var_cov_params_for_initial_obs
           description: elements for lower triangular matrix for generating var-cov matrix 
           type: tensor shape(mp*(mp+1)/2)，)
        m
           description: the dimension of ts
           type: int   
        order
           description: the lag of VAR model
        type: int 

        Returns
        -------
        var-cov matrix of intial p obs.
    """
    mp=m*order
    upper_t_matrix=torch.eye(mp,mp)
    count_temp=0
    #make upper triangel matrix
    for i in range(mp):
        for j in range(mp-i):
            upper_t_matrix[i,i+j]=var_cov_params_for_initial_obs[count_temp]
            count_temp=count_temp+1
    var_cov_m = torch.mm(upper_t_matrix.t(), upper_t_matrix)

    return (var_cov_m)






def A_coeffs_for_causal_VAR(A_coeffs_from_lstm,p,d,var_cov_innovations_varp):
    r"""
        Generate causal VAR coefficients.
        Parameters
        ----------
        A_coeffs_from_lstm
           description: intial A coeffs
           type: tensor 
        p
           description: the lag of VAR model
           type: int   
        d
           description: the dimension of ts
           type: int 
        
        var_cov_innovations_varp
           description: the var-cov matrix of innovations of VAR(p)
           type: tensor 
        Returns
        -------
        Causal VAR coeffs
    """
    Id = torch.eye(d)
    all_p = torch.randn(d, d, p)
    initial_A_coeffs = A_coeffs_from_lstm.reshape(d, d, p)
    #update Gamma_0
    #get initial A coeffics
    #big_A_matrix_for_var_1 = get_A_coeff_m_for_VAR_1(initial_A_coeffs, d, p)
    #Gamma_0=calculate_Gamma0(d, p, big_A_matrix_for_var_1, var_cov_innovations_var1)

    for i1 in range(p):
        A = initial_A_coeffs[:, :, i1]
        # print('A')
        # print(A)
        v=Id + torch.mm(A, A.t())
        # print('v')
        # print(np.linalg.det(v.detach().numpy()))

        B = torch.cholesky(Id + torch.mm(A, A.t())).float()

        all_p[:, :, i1] = torch.solve(A, B)[0]
        # all_p[:,:,i1]=torch.triangular_solve(A,B)[0]
        # print('solve_A_B')
        # print(torch.solve(A, B)[0])
        # print('triangle_solver')
        # print(torch.triangular_solve(A, B)[0])

    all_phi = torch.randn(d, d, p, p)  # [ , , i, j] for phi_{i, j}
    # all_phi=all_phi.clone()
    all_phi_star = torch.randn(d, d, p, p)  # [ , , i, j] for phi_{i, j}*
    # all_phi_star=all_phi_star.clone()
    # Set initial values
    Sigma = Id
    Sigma_star = Id
    L= Id
    L_star = L
    #Gamma = Id
    # Recursion algorithm (Ansley and Kohn 1986, lemma 2.3)
    for s in range(p):
        all_phi[:, :, s, s] = torch.mm(torch.mm(L, all_p[:, :, s].clone()), torch.inverse(L_star))
        # print('1')
        # print(torch.mm(torch.mm(L, all_p[:, :, s]), torch.inverse(L_star)))
        all_phi_star[:, :, s, s] = torch.mm(torch.mm(L_star, all_p[:, :, s].clone().t()), torch.inverse(L))
        # print('2')
        # print(torch.mm(torch.mm(L_star, all_p[:, :, s].t()), torch.inverse(L)))
        if s >= 1:
            for k in list(range(1, s + 1)):
                all_phi[:, :, s, k - 1] = all_phi[:, :, s - 1, k - 1].clone() - torch.mm(all_phi[:, :, s, s].clone(),
                                                                                 all_phi_star[:, :, s - 1, s - k].clone())
                all_phi_star[:, :, s, k - 1] = all_phi_star[:, :, s - 1, k - 1].clone() - torch.mm(all_phi_star[:, :, s, s].clone(),
                                                                                           all_phi[:, :, s - 1, s - k].clone())
        Sigma_next = Sigma - torch.mm(all_phi[:, :, s, s].clone(), torch.mm(Sigma_star, all_phi[:, :, s, s].clone().t()))
        Sigma_star = Sigma_star - torch.mm(all_phi_star[:, :, s, s].clone(),
                                                   torch.mm(Sigma, all_phi_star[:, :, s, s].clone().t()))
                # print('Sigma_star')
                # print(Sigma_star)
        L_star = torch.cholesky(Sigma_star)
        Sigma = Sigma_next
            # print('Sigma')
            # print(Sigma)
        L = torch.cholesky(Sigma)

#   cal T matrix
    lower_t_for_innovations_varp=torch.linalg.cholesky(var_cov_innovations_varp)
    T=torch.mm(lower_t_for_innovations_varp,torch.inverse(L))


    all_A = all_phi[:, :, p - 1, 0:p].clone()
    #all_A.shape:d*d*p
    # print(all_A.shape)
    # print('F_sub')
    for i in range(p):
        all_A[:,:,i]=torch.mm(torch.mm(T,all_A[:,:,i].clone()),torch.inverse(T))
        #print(all_A[:,:,i])
    return all_A


def make_var_cov_of_innovations_var1(Q_t,m,lag_order):
    #generate void matrix
    mp=m*lag_order
    number_of_parms=m*(m+1)/2
    covariance_matrix=torch.eye(m,m)
    count_temp=0
    for i in range(m):
        for j in range(i+1):
            covariance_matrix[i,j]=Q_t[count_temp]
            count_temp=count_temp+1
#diag
    covariance_matrix_semi_positive = torch.mm(covariance_matrix, covariance_matrix.t())

    zeros_cols = torch.zeros([m, (mp - m)])
    #2.generate (mp-m)*mp matrix
    zeros_rows = torch.zeros([(mp - m),mp ])
    #print('Q_covariance')
    #print(covariance_matrix_semi_positive)
    c = torch.cat((covariance_matrix_semi_positive, zeros_cols), 1)
    cov_diag = torch.cat((c, zeros_rows), 0)
    return (cov_diag)

#2020-02-12 add semi-postive and torch.sigmiod()
def make_covariance_matrix_p(v_covariance_coeffs,m,lag_order):
    #generate void matrix
    mp=m*lag_order
    #number_of_parms=mp*(mp+1)/2
    upper_t_matrix=torch.eye(mp,mp)
    count_temp=0
    #make upper triangel matrix
    for i in range(mp):
        for j in range(mp-i):
            upper_t_matrix[i,i+j]=v_covariance_coeffs[count_temp]
            count_temp=count_temp+1
#diag
    covariance_matrix = torch.mm(upper_t_matrix.t(), upper_t_matrix)

    return (covariance_matrix)



def make_G_m(m,order):

    mp = m*order

    return torch.eye(m, mp)



        #return torch.from_numpy(np.concatenate((dig_array, zero_array), axis=1))
       
