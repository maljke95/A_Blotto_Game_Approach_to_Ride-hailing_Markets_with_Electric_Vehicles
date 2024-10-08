# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import brentq

import matplotlib.pyplot as plt
import tikzplotlib
import os
from datetime import datetime

from mpl_toolkits.mplot3d import Axes3D

def f(x, list_of_beta_m, list_of_beta_c, list_of_eps, list_of_nu_a, list_of_nu_b, c):
    
    suma = 0.0
    m = len(list_of_beta_m)
    
    for j in range(m):
        
        bjm  = list_of_beta_m[j]
        bjc  = list_of_beta_c[j]
        epsj = list_of_eps[j]
        nuja = list_of_nu_a[j]
        nujb = list_of_nu_b[j]
        
        alphaj = 2*bjc - nuja - nujb
        
        suma += (bjm + (bjm**2 + 4*bjm*epsj*(alphaj - x))**0.5)/(2*alphaj-2*x)
        
    suma = suma - c
    
    return suma

def f1_NE2(x, b1m, b2m, b1c, b2c, eps1, eps2, Xa, Xb):
    
    s = b1m*eps1/(x + eps1)**2 - b2m*(Xa + eps2)/(Xa + Xb - x + eps2)**2 + b2c - b1c
    
    return s

def f2_NE2(x, b1m, b2m, b1c, b2c, eps1, eps2, Xa, Xb):
    
    s = b1m*(Xa + eps1)/(Xa + x + eps1)**2 - b2m*eps2/(Xb - x + eps2)**2 + b2c - b1c
    
    return s

def f3_NE2(x, b1m, b2m, b1c, b2c, eps1, eps2, Xa, Xb):
    
    s = b1m*eps1/(x + eps1)**2 - b2m*(Xb + eps2)/(Xa + Xb - x + eps2)**2 + b2c - b1c
    
    return s

def f4_NE2(x, b1m, b2m, b1c, b2c, eps1, eps2, Xa, Xb):
    
    s = b1m*(Xb + eps1)/(Xb + x + eps1)**2 - b2m*eps2/(Xa - x + eps2)**2 + b2c - b1c
    
    return s

def find_t_lambda(Xa, Xb, Sigma, list_of_beta_m, list_of_beta_c, list_of_eps, list_of_nu_a, list_of_nu_b):

    c = Xa + Xb + Sigma
    
    list_of_alphaj = []
    
    m = len(list_of_beta_m)
    for j in range(m):

        bjc  = list_of_beta_c[j]
        nuja = list_of_nu_a[j]
        nujb = list_of_nu_b[j]
        
        alphaj = 2*bjc - nuja - nujb
        list_of_alphaj.append(alphaj)
        
    alpha_min = np.sort(np.array(list_of_alphaj))[0]
    
    #----- Find left and right point for the solver -----
    
    delta = 0.1
    found = False 
    
    while not found:
        
        right_point = alpha_min - delta
        
        v = f(right_point, list_of_beta_m, list_of_beta_c, list_of_eps, list_of_nu_a, list_of_nu_b, c)
        if v>0:
            found = True
        else:
            delta = delta/2.0
            
    left_point  = -10.0
    
    found2 = f(left_point, list_of_beta_m, list_of_beta_c, list_of_eps, list_of_nu_a, list_of_nu_b, c)<0.0
    while not found2:
        
        left_point = left_point*2
        found2 = f(left_point, list_of_beta_m, list_of_beta_c, list_of_eps, list_of_nu_a, list_of_nu_b, c)<0.0
        
    t_lambda = brentq(f, left_point, right_point, args=( list_of_beta_m, list_of_beta_c, list_of_eps, list_of_nu_a, list_of_nu_b, c))
    
    return t_lambda

def determine_NE(Xa, Xb, Sigma, list_of_beta_m, list_of_beta_c, list_of_eps, list_of_nu_a, list_of_nu_b):
    
    t_lambda = find_t_lambda(Xa, Xb, Sigma, list_of_beta_m, list_of_beta_c, list_of_eps, list_of_nu_a, list_of_nu_b)
    
    list_of_kj_star = []
    
    m = len(list_of_beta_m)
    
    x = t_lambda
    for j in range(m):

        bjm  = list_of_beta_m[j]
        bjc  = list_of_beta_c[j]
        epsj = list_of_eps[j]
        nuja = list_of_nu_a[j]
        nujb = list_of_nu_b[j]
        
        alphaj = 2*bjc - nuja - nujb
        
        kj_star = (bjm + (bjm**2 + 4*bjm*epsj*(alphaj - x))**0.5)/(2*alphaj-2*x)
        list_of_kj_star.append(kj_star)
        
    suma1 = 0.0
    suma2 = 0.0
    suma3_a = 0.0
    suma3_b = 0.0
    
    for j in range(m):
        
        kj_star = list_of_kj_star[j]
        bjm  = list_of_beta_m[j]
        bjc  = list_of_beta_c[j]
        epsj = list_of_eps[j]
        nuja = list_of_nu_a[j]
        nujb = list_of_nu_b[j]
        
        suma1 += kj_star**2/bjm
        suma2 += kj_star**2 * bjc/bjm
        suma3_a += kj_star**2 * nuja / bjm
        suma3_b += kj_star**2 * nujb /bjm
        
    lambda_a = 1.0/suma1 * (-Sigma - Xb + suma2 - suma3_a)
    lambda_b = 1.0/suma1 * (-Sigma - Xa + suma2 - suma3_b)
    
    xa = []
    xb = []
    
    list_of_profit_a = []
    list_of_profit_b = []
    list_of_loss     = []
    
    check_var = True
    for j in range(m):
        
        kj_star = list_of_kj_star[j]
        bjm  = list_of_beta_m[j]
        bjc  = list_of_beta_c[j]
        epsj = list_of_eps[j]
        nuja = list_of_nu_a[j]
        nujb = list_of_nu_b[j] 
        
        xaj = kj_star**2/bjm * (bjc - lambda_b - nujb) - epsj
        xbj = kj_star**2/bjm * (bjc - lambda_a - nuja) - epsj
        
        profit_aj = xaj * (bjm/(xaj + xbj + epsj) - bjc)
        profit_bj = xbj * (bjm/(xaj + xbj + epsj) - bjc)
        loss_j    = bjm * epsj/(xaj + xbj + epsj)

        list_of_profit_a.append(profit_aj)
        list_of_profit_b.append(profit_bj)
        list_of_loss.append(loss_j)
        
        xa.append(xaj)
        xb.append(xbj)
        
        check_var = check_var and xaj>=0 and xbj>=0
        
    return xa, xb, check_var, list_of_profit_a, list_of_profit_b, list_of_loss

def various_prices(Xa, Xb, Sigma, list_of_beta_m, list_of_eps, list_of_nu_a, list_of_nu_b):
    
    N_coef = 101
    price_coef = np.linspace(1.0,20.0,N_coef)
    
    list_of_xa    = []
    list_of_xb    = []
    list_of_exist = []
    
    list_of_list_of_profits_a = []
    list_of_list_of_profits_b = []
    list_of_list_of_loss      = []
    
    for i in range(N_coef):
        
        coef = price_coef[i]
        list_of_beta_c = [5.0, coef*3, coef*5.0, 50.0]
        
        xa, xb, check_var, list_of_profit_a, list_of_profit_b, list_of_loss = determine_NE(Xa, Xb, Sigma, list_of_beta_m, list_of_beta_c, list_of_eps, list_of_nu_a, list_of_nu_b)
        
        list_of_xa.append(xa)
        list_of_xb.append(xb)
        list_of_exist.append(check_var)
        
        list_of_list_of_profits_a.append(list_of_profit_a)
        list_of_list_of_profits_b.append(list_of_profit_b) 
        list_of_list_of_loss.append(list_of_loss)
    
    #----- Save -----
    
    current_folder = os.getcwd() + '/Results'
    
    if not os.path.isdir(current_folder):
        os.makedirs(current_folder)
        
    now = datetime.now()
    date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
    name = current_folder + "/" + date_time   

    if not os.path.isdir(name):
        os.makedirs(name)
    
    np.save(name+'/list_of_xa.npy', np.array(list_of_xa))
    np.save(name+'/list_of_xb.npy', np.array(list_of_xb))
    np.save(name+'/list_of_exist.npy', np.array(list_of_exist))
    
    np.save(name+'/list_of_list_of_profits_a.npy', np.array(list_of_list_of_profits_a))
    np.save(name+'/list_of_list_of_profits_b.npy', np.array(list_of_list_of_profits_b))
    np.save(name+'/list_of_list_of_loss.npy', np.array(list_of_list_of_loss))
    
    #----- Plot -----
    
    list_of_xa    = np.array(list_of_xa)
    list_of_xb    = np.array(list_of_xb)
    list_of_exist = np.array(list_of_exist)
    
    list_of_list_of_profits_a = np.array(list_of_list_of_profits_a)
    list_of_list_of_profits_b = np.array(list_of_list_of_profits_b)
    list_of_list_of_loss      = np.array(list_of_list_of_loss)
    
    m = len(list_of_beta_m)
    
    fig2, ax2 = plt.subplots(dpi=180)
    fig3, ax3 = plt.subplots(dpi=180)
    
    lista_id = list(np.linspace(1,m,m))
    lista_id = list(map(int, lista_id))
    
    list_of_colors = [(0.9, 0.7, 0), 'tab:red', 'tab:blue' , 'tab:green', 'tab:gray', 'tab:brown', 'tab:pink', 'tab:cian', 'tab:olive', 'tab:purple']
    
    for i in range(m):
        
        fig1, ax1 = plt.subplots(dpi=180)
        
        ax1.plot(price_coef, list_of_list_of_profits_a[:,i], color='tab:orange', label='Comp A')
        ax1.plot(price_coef, list_of_list_of_profits_b[:,i], color='tab:purple', label='Comp B')
        ax1.plot(price_coef, list_of_list_of_loss[:,i], color='black', linestyle='--', label='Loss')
        ax1.grid('on')
        ax1.legend()
        ax1.set_xlabel("Coeff k")
        ax1.set_ylabel("Payof region "+str(i+1))
        ax1.set_xlim((np.min(price_coef), np.max(price_coef)))
        
        fig1.savefig(name + "/Payof_region_"+str(i+1)+".jpg", dpi=180)
    
        tikzplotlib.clean_figure()
        tikzplotlib.save(name + "/Payof_region_"+str(i+1)+".tex")
        
        ax2.plot(price_coef, list_of_xa[:,i], label='Region '+str(i), color=list_of_colors[i])
        ax3.plot(price_coef, list_of_xb[:,i], label='Region '+str(i), color=list_of_colors[i])
        
    ax2.grid('on')
    ax2.legend()
    ax2.set_xlabel("Coeff k")
    ax2.set_ylabel("Nash equilbrium, Comp A")
    ax2.set_xlim((np.min(price_coef), np.max(price_coef)))    

    fig2.savefig(name + "/NE_A.jpg", dpi=180)
    
    #tikzplotlib.clean_figure()
    tikzplotlib.save(name + "/NE_A.tex")
    
    
    ax3.grid('on')
    ax3.legend()
    ax3.set_xlabel("Coeff k")
    ax3.set_ylabel("Nash equilbrium, Comp B")
    ax3.set_xlim((np.min(price_coef), np.max(price_coef)))  
    
    fig3.savefig(name + "/NE_B.jpg", dpi=180)
    
    #tikzplotlib.clean_figure()
    tikzplotlib.save(name + "/NE_B.tex")
    
    #----- Plot 3D -----
    
    # fig4 = plt.figure()
    # ax4 = fig4.add_subplot(111, projection='3d')
    # ax4.plot(list_of_xa[:,0], list_of_xa[:,1], list_of_xa[:,2])
    # ax4.set_xlabel('reg 1')
    # ax4.set_ylabel('reg 2')
    # ax4.set_zlabel('reg 3')
    # ax4.set_xlim((np.min(list_of_xa[:,0]), np.max(list_of_xa[:,0])))
    # ax4.set_ylim((np.min(list_of_xa[:,1]), np.max(list_of_xa[:,1])))
    # ax4.set_zlim((np.min(list_of_xa[:,2]), np.max(list_of_xa[:,2])))
    
    # fig5 = plt.figure()
    # ax5 = fig5.add_subplot(111, projection='3d')
    # ax5.plot(list_of_xb[:,0], list_of_xb[:,1], list_of_xb[:,2])
    
    return list_of_xa, list_of_xb

def find_all_NE_2reg(Xa, Xb, Sigma, list_of_beta_m, list_of_beta_c, list_of_eps):
    
    b1c = list_of_beta_c[0]
    b2c = list_of_beta_c[1]
    
    b1m = list_of_beta_m[0]
    b2m = list_of_beta_m[1]
    
    eps1 = list_of_eps[0]
    eps2 = list_of_eps[1]
    
    m = len(list_of_beta_m)
    
    list_of_nu_a = m*[0.0]
    list_of_nu_b = m*[0.0]
    
    list_of_NE_a = []
    list_of_NE_b = []
    
    xa, xb, check_var, list_of_profit_a, list_of_profit_b, list_of_loss = determine_NE(Xa, Xb, Sigma, list_of_beta_m, list_of_beta_c, list_of_eps, list_of_nu_a, list_of_nu_b)
    
    if check_var:
        
        list_of_NE_a.append(xa)
        list_of_NE_b.append(xb)
        
    #----- Boundary NE -----
    
    Vup1 = b2c - b1c + b1m/eps1 -(b2m*(Xa + eps2))/(Xa + Xb + eps2)**2
    Vdw1 = b2c - b1c + b1m*eps1/(Xb + eps1)**2 - b2m/(Xa + eps2)
    
    Wup1 = b2c - b1c + b1m/(Xa + eps1) - b2m*eps2/(Xb + eps2)**2
    Wdw1 = b2c - b1c - b2m/eps2 + b1m*(Xa + eps1)/(Xa + Xb + eps1)**2   

    Vup2 = b2c - b1c + b1m/eps1 -(b2m*(Xb + eps2))/(Xa + Xb + eps2)**2
    Vdw2 = b2c - b1c + b1m*eps1/(Xa + eps1)**2 - b2m/(Xb + eps2)    

    Wup2 = b2c - b1c + b1m/(Xb + eps1) - b2m*eps2/(Xa + eps2)**2
    Wdw2 = b2c - b1c - b2m/eps2 + b1m*(Xb + eps1)/(Xa + Xb + eps1)**2
    
    # print(Vup1, Vdw1)
    # print(Wup1, Wdw1)
    # print(Vup2, Vdw2)
    # print(Wup2, Wdw2)
    # print('---------------------')
    
    if Vdw1>=0:
        z1 = Xb
    elif Vup1<=0:
        z1 = 0.0
    else:
        z1 = brentq(f1_NE2, 0.0, Xb, args=( b1m, b2m, b1c, b2c, eps1, eps2, Xa, Xb))
    
    xa_cand1 = [0.0, Xa]
    xb_cand1 = [z1, Xb-z1]
    
    #----------
    if Wdw1>=0:
        z2 = Xb
    elif Wup1<=0:
        z2 = 0.0
    else:
        z2 = brentq(f2_NE2, 0.0, Xb, args=( b1m, b2m, b1c, b2c, eps1, eps2, Xa, Xb))

    xa_cand2 = [Xa, 0.0]
    xb_cand2 = [z2, Xb-z2]
    
    #----------
    if Vdw2>=0:
        z3 = Xa
    elif Vup2<=0:
        z3 = 0.0
    else:
        z3 = brentq(f3_NE2, 0.0, Xa, args=( b1m, b2m, b1c, b2c, eps1, eps2, Xa, Xb))

    xa_cand3 = [z3, Xa-z3]
    xb_cand3 = [0.0, Xb]
    
    #----------
    if Wdw2>=0:
        z4 = Xa
    elif Wup2<=0:
        z4 = 0.0
    else:
        z4 = brentq(f4_NE2, 0.0, Xa, args=( b1m, b2m, b1c, b2c, eps1, eps2, Xa, Xb))

    xa_cand4 = [z4, Xa-z4]
    xb_cand4 = [Xb, 0.0]
    
    #-----
    
    cond1 = 0<z1 and z1<Xb
    help1 = (Xb-z1-Xa)*b2m/(Xa+Xb-z1+eps2)**2-b1m*z1/(z1+eps1)**2
    cond1 = cond1 and help1>=0
    
    cond1_ = z1==0
    help1_ = -Vup1+(Xb-Xa)*b2m/(Xa+Xb+eps2)**2
    cond1_ = cond1_ and help1_>=0
    
    c1 = cond1 or cond1_
    
    #-----

    cond2 = 0<z2 and z2<Xb
    help2 = (z2-Xa)*b1m/(Xa+z2+eps1)**2-(Xb-z2)*b2m/(Xb-z2+eps2)**2
    cond2 = cond2 and help2>=0
    
    cond2_ = z2==Xb
    help2_ = Wdw1+(Xb-Xa)*b1m/(Xa+Xb+eps1)**2
    cond2_ = cond2_ and help2_>=0    
    
    c2 = cond2 or cond2_

    #-----

    cond3 = 0<z3 and z3<Xa
    help3 = (Xa-z3-Xb)*b2m/(Xa+Xb-z3+eps2)**2-b1m*z3/(z3+eps1)**2
    cond3 = cond3 and help3>=0
    
    cond3_ = z3==0
    help3_ = -Vup2+(Xa-Xb)*b2m/(Xa+Xb+eps2)**2
    cond3_ = cond3_ and help3_>=0
    
    c3 = cond3 or cond3_ 

    #-----

    cond4 = 0<z4 and z4<Xa
    help4 = (z4-Xb)*b1m/(Xb+z4+eps1)**2-(Xa-z4)*b2m/(Xa-z4+eps2)**2
    cond4 = cond4 and help4>=0
    
    cond4_ = z4==Xa
    help4_ = Wdw2+(Xa-Xb)*b1m/(Xa+Xb+eps1)**2
    cond4_ = cond4_ and help4_>=0 
    
    c4 = cond4 or cond4_ 

    if c1:
        
        list_of_NE_a.append(xa_cand1)
        list_of_NE_b.append(xb_cand1)       

    if c2:
        
        list_of_NE_a.append(xa_cand2)
        list_of_NE_b.append(xb_cand2)
        
    if c3:
        
        list_of_NE_a.append(xa_cand3)
        list_of_NE_b.append(xb_cand3)        

    if c4:
        
        list_of_NE_a.append(xa_cand4)
        list_of_NE_b.append(xb_cand4)
    
    list_of_z = [z1, z2, z3, z4]
    list_of_cond_values = [check_var, help1, help1_, help2, help2_, help3, help3_, help4, help4_]
    
    list_of_profits_a = []
    list_of_profits_b = []

    for j in range(2):
        
        bjm  = list_of_beta_m[j]
        bjc  = list_of_beta_c[j]
        epsj = list_of_eps[j]
        
        xaj = list_of_NE_a[0][j]
        xbj = list_of_NE_b[0][j]
        
        profit_aj = xaj * (bjm/(xaj + xbj + epsj) - bjc)
        profit_bj = xbj * (bjm/(xaj + xbj + epsj) - bjc)
        
        list_of_profits_a.append(profit_aj)
        list_of_profits_b.append(profit_bj)
    
    return list_of_NE_a, list_of_NE_b, list_of_z, list_of_cond_values, list_of_profits_a, list_of_profits_b

def various_prices_2REG(Xa, Xb, Sigma, list_of_beta_m, list_of_eps):
    
    N_coef = 101
    price_coef = np.linspace(1.0,50.0,N_coef)
    
    list_of_xa    = []
    list_of_xb    = []
    list_of_zs    = []
    list_of_cond  = []
    
    how_many_NE   = []
    
    pA_x1 = []
    pA_x2 = []
    
    pB_x1 = []
    pB_x2 = []
    
    profitA_x1 = []
    profitA_x2 = []
    
    profitB_x1 = []
    profitB_x2 = []
    
    for i in range(N_coef):
        
        coef = price_coef[i]
        list_of_beta_c = [10.0, coef*10.0]
        
        list_of_NE_a, list_of_NE_b, list_of_z, list_of_cond_values, list_of_profits_a, list_of_profits_b = find_all_NE_2reg(Xa, Xb, Sigma, list_of_beta_m, list_of_beta_c, list_of_eps)
        
        how_many_NE.append(len(list_of_NE_a))
        
        list_of_xa.append(list_of_NE_a)
        list_of_xb.append(list_of_NE_b)
        list_of_zs.append(list_of_z)
        list_of_cond.append(list_of_cond_values)
        
        pA_x1.append(list_of_NE_a[0][0])
        pA_x2.append(list_of_NE_a[0][1])
        
        pB_x1.append(list_of_NE_b[0][0])
        pB_x2.append(list_of_NE_b[0][1])

        profitA_x1.append(list_of_profits_a[0])
        profitA_x2.append(list_of_profits_a[1])
    
        profitB_x1.append(list_of_profits_b[0])
        profitB_x2.append(list_of_profits_b[1])
    
    #----- Save -----
    
    current_folder = os.getcwd() + '/Results'
    
    if not os.path.isdir(current_folder):
        os.makedirs(current_folder)
        
    now = datetime.now()
    date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
    name = current_folder + "/" + date_time   

    if not os.path.isdir(name):
        os.makedirs(name)
    
    np.save(name+'/how_many_NE.npy', np.array(how_many_NE))
    
    #----- Prepare plots -----

    fig1, ax1 = plt.subplots(dpi=180)
    
    ax1.plot(price_coef, pA_x1, color=(0.9, 0.7, 0), label="Comp A, reg1")
    ax1.plot(price_coef, pA_x2, color='tab:red', label="Comp A, reg2")
    ax1.plot(price_coef, pB_x1, color=(0.9, 0.7, 0), linestyle='--', label="Comp B, reg1")
    ax1.plot(price_coef, pB_x2, color='tab:red', linestyle='--', label="Comp B, reg2")
    
    ax1.grid('on')
    ax1.legend()
    ax1.set_xlabel("Coeff k")
    ax1.set_ylabel("Nash eq")
    
    fig1.savefig(name + "/NE_2reg.jpg", dpi=180)
    
    tikzplotlib.clean_figure()
    tikzplotlib.save(name + "/NE_2reg.tex")
    
    fig2, ax2 = plt.subplots(dpi=180)

    ax2.plot(price_coef, profitA_x1, color=(0.9, 0.7, 0), label="Comp A, reg1")
    ax2.plot(price_coef, profitA_x2, color='tab:red', label="Comp A, reg2")
    ax2.plot(price_coef, profitB_x1, color=(0.9, 0.7, 0), linestyle='--', label="Comp B, reg1")
    ax2.plot(price_coef, profitB_x2, color='tab:red', linestyle='--', label="Comp B, reg2")

    ax2.grid('on')
    ax2.legend()
    ax2.set_xlabel("Coeff k")
    ax2.set_ylabel("Profits")
    
    fig2.savefig(name + "/profits_2reg.jpg", dpi=180)
    
    tikzplotlib.clean_figure()
    tikzplotlib.save(name + "/profits_2reg.tex")
    
    return how_many_NE,list_of_xa, list_of_xb, list_of_zs, list_of_cond, price_coef, np.array(profitA_x1)+np.array(profitA_x2), np.array(profitB_x1)+np.array(profitB_x2)

def change_fleet_size_2REG(Xa_list, Xb_list, Sigma, list_of_beta_m, list_of_beta_c, list_of_eps):
    
    list_of_xa = []
    list_of_xb = []    

    pA_x1 = []
    pA_x2 = []
    
    pB_x1 = []
    pB_x2 = []
    
    profitA_x1 = []
    profitA_x2 = []
    
    profitB_x1 = []
    profitB_x2 = []
    
    for i in range(len(Xb_list)):
        
        Xa = Xa_list[0]
        Xb = Xb_list[i]
        
        list_of_NE_a, list_of_NE_b, list_of_z, list_of_cond_values, list_of_profits_a, list_of_profits_b = find_all_NE_2reg(Xa, Xb, Sigma, list_of_beta_m, list_of_beta_c, list_of_eps)
        
        list_of_xa.append(list_of_NE_a)
        list_of_xb.append(list_of_NE_b)
        
        pA_x1.append(list_of_NE_a[0][0])
        pA_x2.append(list_of_NE_a[0][1])
        
        pB_x1.append(list_of_NE_b[0][0])
        pB_x2.append(list_of_NE_b[0][1])

        profitA_x1.append(list_of_profits_a[0])
        profitA_x2.append(list_of_profits_a[1])
    
        profitB_x1.append(list_of_profits_b[0])
        profitB_x2.append(list_of_profits_b[1])        

    #----- Save -----
    
    current_folder = os.getcwd() + '/Results'
    
    if not os.path.isdir(current_folder):
        os.makedirs(current_folder)
        
    now = datetime.now()
    date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
    name = current_folder + "/" + date_time   

    if not os.path.isdir(name):
        os.makedirs(name)

    #----- Prepare plots -----

    fig1, ax1 = plt.subplots(dpi=180)
    
    ax1.plot(Xb_list, pA_x1, color=(0.9, 0.7, 0), label="Comp A, reg1")
    ax1.plot(Xb_list, pA_x2, color='tab:red', label="Comp A, reg2")
    ax1.plot(Xb_list, pB_x1, color=(0.9, 0.7, 0), linestyle='--', label="Comp B, reg1")
    ax1.plot(Xb_list, pB_x2, color='tab:red', linestyle='--', label="Comp B, reg2")
    
    ax1.grid('on')
    ax1.legend()
    ax1.set_xlabel("Xb")
    ax1.set_ylabel("Nash eq")
    
    fig1.savefig(name + "/NE_2reg.jpg", dpi=180)
    
    tikzplotlib.clean_figure()
    tikzplotlib.save(name + "/NE_2reg.tex")
    
    fig2, ax2 = plt.subplots(dpi=180)

    ax2.plot(Xb_list, profitA_x1, color=(0.9, 0.7, 0), label="Comp A, reg1")
    ax2.plot(Xb_list, profitA_x2, color='tab:red', label="Comp A, reg2")
    ax2.plot(Xb_list, profitB_x1, color=(0.9, 0.7, 0), linestyle='--', label="Comp B, reg1")
    ax2.plot(Xb_list, profitB_x2, color='tab:red', linestyle='--', label="Comp B, reg2")

    ax2.grid('on')
    ax2.legend()
    ax2.set_xlabel("Xb")
    ax2.set_ylabel("Profits")
    
    fig2.savefig(name + "/profits_2reg.jpg", dpi=180)
    
    tikzplotlib.clean_figure()
    tikzplotlib.save(name + "/profits_2reg.tex")
    
    fig3, ax3 = plt.subplots(dpi=180)   
    
    ax3.plot(Xb_list, np.array(profitA_x1)+np.array(profitA_x2), color=(0.9, 0.7, 0), label="Comp A")
    ax3.plot(Xb_list, np.array(profitB_x1)+np.array(profitB_x2), color='tab:red', label="Comp B")

    ax3.grid('on')
    ax3.legend()
    ax3.set_xlabel("Xb")
    ax3.set_ylabel("Total profits")
    
    fig3.savefig(name + "/total_profits_2reg.jpg", dpi=180)
    
    tikzplotlib.clean_figure()
    tikzplotlib.save(name + "/total_profits_2reg.tex")    
    
    
    return list_of_xa, list_of_xb, np.array(profitA_x1)+np.array(profitA_x2), np.array(profitB_x1)+np.array(profitB_x2)
         
if __name__ == '__main__':

    Xa = 1000.0
    Xb = 2000.0

    # #---------------------- Test for 3 regions + multiple coeff
    
    # list_of_beta_m = [35000, 50000, 100000, 180000]
    # list_of_beta_c = [10.0, 20.0, 25, 30.0]
    # list_of_eps    = [50.0, 100.0, 120, 200.0]    
    # list_of_nu_a   = [0.0, 0.0, 0.0, 0.0]
    # list_of_nu_b   = [0.0, 0.0, 0.0, 0.0]      
    # Sigma = 0.0
    # for i in range(len(list_of_eps)):
    #     Sigma += list_of_eps[i]
    
    # #xa, xb, check_var, list_of_profit_a, list_of_profit_b, list_of_loss = determine_NE(Xa, Xb, Sigma, list_of_beta_m, list_of_beta_c, list_of_eps, list_of_nu_a, list_of_nu_b)
    
    # list_of_xa, list_of_xb = various_prices(Xa, Xb, Sigma, list_of_beta_m, list_of_eps, list_of_nu_a, list_of_nu_b)

    #--------------------- Test for 2 regions 

    # list_of_beta_m2 = [3500, 5000, 6000, 6500]
    # list_of_eps2    = [5.0 , 10.0, 20, 30]    
    # list_of_beta_c2 = [5.0, 10.0, 15, 25]
    # list_of_nu_a2 = 4*[0.0]
    # list_of_nu_b2 = 4*[0.0]
    # Sigma = 0.0
    # for i in range(len(list_of_eps2)):
    #     Sigma += list_of_eps2[i]
    # xa, xb, check_var, list_of_profit_a, list_of_profit_b, list_of_loss = determine_NE(Xa, Xb, Sigma, list_of_beta_m2, list_of_beta_c2, list_of_eps2, list_of_nu_a2, list_of_nu_b2)

    #--------------------- Test for 2 regions + multiple coeff
    
    list_of_beta_m2 = [35000, 120000]
    list_of_eps2    = [0.01, 0.01]#[100.0 , 300.0]
    Sigma = 0.0
    for i in range(len(list_of_eps2)):
        Sigma += list_of_eps2[i]
    #how_many_NE,list_of_xa, list_of_xb, list_of_z, list_of_cond, price_coef, total_pA, total_pB = various_prices_2REG(Xa, Xb, Sigma, list_of_beta_m2, list_of_eps2)
    
    #--------------------- Vary the fleet size 
    
    Xa_list = [1000.0]
    Xb_list = np.linspace(200.0, 5000.0, 200)
    list_of_beta_c2 = [10.0, 30.0]
    
    list_of_xa, list_of_xb, total_pA, total_pB = change_fleet_size_2REG(Xa_list, Xb_list, Sigma, list_of_beta_m2, list_of_beta_c2, list_of_eps2)
