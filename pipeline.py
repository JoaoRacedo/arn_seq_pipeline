# coding: utf-8

import numpy as np
import math
from numpy.linalg import multi_dot
np.set_printoptions(suppress=True)
import multiprocessing as mp
#print(mp.cpu_count())

def CovMat(X_g):
    [n,m] = X_g.shape
    Sig = np.zeros((m,m))
    vars = np.var(X_g, axis = 0)
    np.fill_diagonal(Sig, np.sqrt(vars))
    b = np.mean(X_g, axis = 0)
    I = np.ones((n,1),dtype = int)
    Xg = np.dot((X_g-I*b),(np.linalg.inv(Sig)))
    Sg = (1/(n-1))*np.dot(Xg.T,Xg)
    return Sg

def Dim_red(L):
    a=0
    sum_total = np.sum(L)
    Sum_temp = 0
    i = 0
    n = len(L)
    while (Sum_temp/sum_total <0.95):
        Sum_temp += L[i]
        i+=1
        a+=1
    return a

def Eigen(W,V):
    W = W.real
    V = V.real
    # Sorting in descending order
    idx = W.argsort()[::-1] 
    W = W[idx] # Diagonal Vector
    W_diagonal = np.zeros((len(W),len(W)))
    np.fill_diagonal(W_diagonal, W) # Convert W into diag matrix
    V = V[:,idx]  # eigvector columns
    return W_diagonal,V

def Statistic(Sc,Sv):
    [Wc,Vc] = (np.linalg.eig(Sc))
    [Wv,Vv] = (np.linalg.eig(Sv))
    Wc = np.real(Wc)
    Vc = np.real(Vc)
    Wv = np.real(Wv)
    Vv = np.real(Vv)
    [Wc,Vc] = Eigen(Wc,Vc)
    [Wv,Vv] = Eigen(Wv,Vv)
    ac = Dim_red(np.diag(Wc))
    av = Dim_red(np.diag(Wv))
    phi = 0
    a = max(ac,av)
    for i in range(a):
        phi += max(Wc[i,i],Wv[i,i])*(Wc[i,i]-Wv[i,i])*math.acos(np.dot(Vc[:,i],np.transpose(Vv[:,i][np.newaxis])))
    return phi

def Single_Classification(Xc,Xv,xp):
    [_,m] = Xc.shape
    xp = xp.copy()[np.newaxis]
    Sc = CovMat(Xc)
    Sv = CovMat(Xv)
    bc = np.mean(Xc, axis = 0)[np.newaxis].T
    bv = np.mean(Xv, axis = 0)[np.newaxis].T
    Sigmac = np.zeros((m,m))
    Sigmav = np.zeros((m,m))
    varc = np.var(Xc, axis = 0)**0.5
    varv = np.var(Xv, axis = 0)**0.5
    for i in range(m):
        Sigmac[i,i] = varc[i]
        Sigmav[i,i] = varv[i]
    n_reps = range(1,10,2)
    psi_c_inv = np.zeros(len(n_reps))
    psi_v_inv = np.zeros(len(n_reps))
    for i in range(len(n_reps)):
        nred = n_reps[i]
        bc_dis = (nred/(nred+1))*bc+ (1/(nred+1))*xp.T
        d_bc = bc_dis - bc
        bv_dis = (nred/(nred+1))*bv+ (1/(nred+1))*xp.T
        d_bv = bv_dis - bv
        Sigma_c_dis = np.zeros((m,m))
        Sigma_v_dis = np.zeros((m,m))
        temp_xp = xp.copy()[0]
        for j in range(m):
            Sigma_c_dis[j,j] = np.sqrt(((nred-1)/nred)*(Sigmac[j,j])**2
                                      + d_bc[j]**2 + (1/nred)*(temp_xp[j] -bc_dis[j])**2)
            Sigma_v_dis[j,j] = np.sqrt(((nred-1)/nred)*(Sigmav[j,j])**2
                                      + d_bv[j]**2 + (1/nred)*(temp_xp[j] -bv_dis[j])**2)
        inv_sigma_c_dis = np.linalg.inv(Sigma_c_dis)
        inv_sigma_v_dis = np.linalg.inv(Sigma_v_dis)
        xp_norm_c = np.dot((xp-bc_dis.T),inv_sigma_c_dis)
        xp_norm_v = np.dot((xp-bv_dis.T),inv_sigma_v_dis)
        Sc_dis = ((nred-1)/nred)*multi_dot([inv_sigma_c_dis,Sigmac,Sc,Sigmac,inv_sigma_c_dis]) + multi_dot([inv_sigma_c_dis,d_bc,d_bc.T,inv_sigma_c_dis])+ (1/nred)*np.dot(xp_norm_c.T,xp_norm_c)
        #Sc_dis = ((nred-1)/nred)*inv_sigma_c*Sigma_c_dis*Sc*Sigma_c_dis*inv_sigma_c + inv_sigma_c*d_bc*d_bc.T *inv_sigma_c+ (1/nred)*np.dot(xp_norm_c.T,xp_norm_c)
        Sv_dis = ((nred-1)/nred)*multi_dot([inv_sigma_v_dis,Sigmav,Sv,Sigmav,inv_sigma_v_dis]) + multi_dot([inv_sigma_v_dis,d_bv,d_bv.T,inv_sigma_v_dis])+ (1/nred)*np.dot(xp_norm_v.T,xp_norm_v)
        #Sv_dis = ((nred-1)/nred)*inv_sigma_v*Sigma_v_dis*Sv*Sigma_v_dis*inv_sigma_v + inv_sigma_v*d_bv*d_bv.T *inv_sigma_v+ (1/nred)*np.dot(xp_norm_v.T,xp_norm_v)
        psi_c_inv[i] = 1/(np.abs(Statistic(Sc,Sc_dis)))
        psi_v_inv[i] = 1/(np.abs(Statistic(Sv,Sv_dis)))    
    psi_c_mean = np.mean(psi_c_inv)
    psi_v_mean = np.mean(psi_v_inv)
    return psi_c_mean, psi_v_mean


############################
#   MAIN MULTIPROCESSING   #
############################

def Test(reps,n,m,rho1,rho2,b_m):
    b_mean_c = b_m*np.ones(m, dtype = int)
    b_mean_v = b_mean_c+20
    I = np.ones((m,1),dtype = int)
    Sc = (1-rho1)*np.eye(m, dtype=int) + rho1*(I)*I.T
    Sv = (1-rho2)*np.eye(m, dtype=int) + rho2*(I)*I.T
    Xc = np.random.multivariate_normal(b_mean_c,Sc,n)
    Xv = np.random.multivariate_normal(b_mean_v,Sv,n)
    right_class_c = 0
    for control in range(n):
        xp = Xc[control,:]
        psi_c_mean_single,psi_v_mean_single = Single_Classification(Xc,Xv,xp)
        if (psi_c_mean_single > psi_v_mean_single):
            right_class_c += 1
    MR_c = 100*(1-(right_class_c/n))
    right_class_v = 0
    for patient in range(n):
        xp = Xv[patient,:]
        psi_c_mean_single,psi_v_mean_single = Single_Classification(Xc,Xv,xp)
        if (psi_v_mean_single > psi_c_mean_single):
            right_class_v += 1
    MR_v = 100*(1-(right_class_v/n))
    return MR_c+MR_v


n = np.array([260,20])
m = np.array([40,20])
rho1 = np.array([0.1])
rho2 = np.array([0.2])
b_m = 20
Matrix = np.array(np.meshgrid(n,m,rho1,rho2)).T.reshape(-1,4)
n_reps = 2
n_exps = len(n)*len(m)
MR_c = np.zeros((n_reps,n_exps))
MR_v = np.zeros((n_reps,n_exps))
MR_total = np.zeros((n_exps))
for k in range(n_exps):
    n = Matrix[k,0]
    m = Matrix[k,1]
    rho1 = Matrix[k,2]
    rho2 = Matrix[k,3]
    pool = mp.Pool(mp.cpu_count()-1)
    result_objects = [pool.apply_async(Test, args=(reps,int(n),int(m),rho1,rho2,b_m)) for reps in range(n_reps)]
    pool.close()
    temp_results = [result.get() for result in result_objects]
    print(temp_results)
    MR_total[k] = np.mean(0.5*np.array(temp_results))
    #for reps in range(n_reps):
    #    MR_c[reps,k], MR_v[reps,k] = Test(int(n),int(m),rho1,rho2,b_m)
#pool.join()
#MR_mean = np.mean(0.5*(MR_c+MR_v),axis = 0)[np.newaxis]
MR_total = MR_total[np.newaxis]
Results = np.concatenate((Matrix, MR_total.T), axis=1)
Results


################
#   MAIN       #
################

#n = np.array([20,40,60,80,100,120,140,160,180,200,220,240,260])
#m = np.array([20,40,60,80,100,120,140])
n = np.array([260,20])
m = np.array([40,20])
rho1 = np.array([0.1])
rho2 = np.array([0.2])
b_m = 20
rhos = np.array(np.meshgrid(rho1,rho2)).T.reshape(-1,2)
Matrix = np.array(np.meshgrid(n,m)).T.reshape(-1,2)
n_reps = 2
n_exps = len(n)*len(m)
MR_c = np.zeros((n_reps,n_exps))
MR_v = np.zeros((n_reps,n_exps))
for k in range(n_exps):
    n = Matrix[k,0]
    m = Matrix[k,1]
    b_mean_c = b_m*np.ones(m, dtype = int)
    b_mean_v = b_mean_c+20
    for rho in rhos:
        I = np.ones((m,1),dtype = int)
        Sc = (1-rho[0])*np.eye(m, dtype=int) + rho[0]*(I)*I.T
        Sv = (1-rho[1])*np.eye(m, dtype=int) + rho[1]*(I)*I.T
        for reps in range(n_reps):
            Xc = np.random.multivariate_normal(b_mean_c,Sc,n)
            Xv = np.random.multivariate_normal(b_mean_v,Sv,n)
            right_class_c = 0
            #print("Experiment " + str(k) + " Repetition " + str(reps))
            #print("Control")
            for control in range(n):
                xp = Xc[control,:]
                psi_c_mean_single,psi_v_mean_single = Single_Classification(Xc,Xv,xp)
                if (psi_c_mean_single > psi_v_mean_single):
                    right_class_c += 1
            #print(right_class_c)
            MR_c[reps,k] = 100*(1-(right_class_c/n))
            #print(100*(1-(right_class_c/n)))
            #print("---------")
            right_class_v = 0
            #print("Patient")
            for patient in range(n):
                xp = Xv[patient,:]
                psi_c_mean_single,psi_v_mean_single = Single_Classification(Xc,Xv,xp)
                if (psi_v_mean_single > psi_c_mean_single):
                        right_class_v += 1
            #print(right_class_v)
            MR_v[reps,k] = 100*(1-(right_class_v/n))
            #print(100*(1-(right_class_v/n)))
            #print("--------")
MR_mean = np.mean(0.5*(MR_c+MR_v),axis = 0)[np.newaxis]
Results = np.concatenate((Matrix, MR_mean.T), axis=1)
Results
