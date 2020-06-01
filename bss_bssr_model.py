# -*- coding: utf-8 -*-
"""
Created on Sun May 31 17:00:43 2020

@author: alanw
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as si
from sklearn.metrics import mean_squared_error
import time
##############################################################################
# CRR Model - For American Options
##############################################################################
def CRR(Put_Call, n, S_0, X, rfr, vol, t, AMN_EUR):  
    deltaT = t/n 
    u = np.exp(vol*np.sqrt(deltaT))
    d = 1./u
    R = np.exp(rfr*deltaT)
    p = (R-d)/(u-d)
    q = 1-p     
    
    # simulating the underlying price paths
    S = np.zeros((n+1,n+1))
    S[0,0] = S_0
    for i in range(1,n+1):
        S[i,0] = S[i-1,0]*u
        for j in range(1,i+1):
            S[i,j] = S[i-1,j-1]*d
    
    # option value at final node   
    V = np.zeros((n+1,n+1)) # V[i,j] is the option value at node (i,j)
    for j in range(n+1):
        if Put_Call=="C":
            V[n,j] = max(0, S[n,j]-X)
        elif Put_Call=="P":
            V[n,j] = max(0, X-S[n,j])
            
    # European Otpion: backward induction to the option price V[0,0]        
    if AMN_EUR == "E":            
    
        for i in range(n-1,-1,-1):
            for j in range(i+1):
                    V[i,j] = max(0, 1/R*(p*V[i+1,j]+q*V[i+1,j+1]))
        opt_price = V[0,0]
    # American Otpion: backward induction to the option price V[0,0] 
    elif AMN_EUR == "A":
        for i in range(n-1,-1,-1):
            for j in range(i+1):
                    if Put_Call=="P":
                        V[i,j] = max(0, X-S[i,j], 
                                     1/R*(p*V[i+1,j]+q*V[i+1,j+1]))
                    elif Put_Call=="C":
                        V[i,j] = max(0, S[i,j]-X,
                                     1/R*(p*V[i+1,j]+q*V[i+1,j+1]))
        opt_price = V[0,0]
        
    return opt_price

##############################################################################
# BSS Model
##############################################################################
def BBS(Put_Call, n, S_0, X, rfr, vol, t, AMN_EUR):  
    deltaT = t/n 
    u = np.exp(vol*np.sqrt(deltaT))
    d = 1./u
    R = np.exp(rfr*deltaT)
    p = (R-d)/(u-d)
    q = 1-p     
    
    # simulating the underlying price paths
    S = np.zeros((n+1,n+1))
    S[0,0] = S_0
    for i in range(1,n+1):
        S[i,0] = S[i-1,0]*u
        for j in range(1,i+1):
            S[i,j] = S[i-1,j-1]*d
    
    # option value at final node   
    V = np.zeros((n+1,n+1)) # V[i,j] is the option value at node (i,j)
    for j in range(n+1):
        if Put_Call=="C":
            V[n,j] = max(0, S[n,j]-X)
        elif Put_Call=="P":
            V[n,j] = max(0, X-S[n,j])
    for j in range(n):
        V[n-1,j] = BSM(Put_Call, S[n-1,j], X, rfr, vol, t/n)
            
    # European Option: backward induction to the option price V[0,0]        
    if AMN_EUR == "E":
        for i in range(n-2,-1,-1):
            for j in range(i+1):
                V[i,j] = max(0, 1/R*(p*V[i+1,j]+q*V[i+1,j+1]))
        opt_price = V[0,0]
        
    # American Otpion: backward induction to the option price V[0,0] 
    elif AMN_EUR == "A":       
        for i in range(n-2,-1,-1):
            for j in range(i+1):
                    if Put_Call=="P":
                        V[i,j] = max(0, X-S[i,j], 
                                     1/R*(p*V[i+1,j]+q*V[i+1,j+1]))
                    elif Put_Call=="C":
                        V[i,j] = max(0, S[i,j]-X,
                                     1/R*(p*V[i+1,j]+q*V[i+1,j+1]))
        opt_price = V[0,0]
        
    return opt_price

##############################################################################
# BBSR Model
##############################################################################
def BBSR(Put_Call, n, S_0, X, rfr, vol, t, AMN_EUR): 
    opt_price = 2*BBS(Put_Call, n, S_0, X, rfr, vol, t, AMN_EUR) \
        - BBS(Put_Call, int(n/2), S_0, X, rfr, vol, t, AMN_EUR)
    
    return opt_price

##############################################################################
# BSM Model
##############################################################################
def BSM(Put_Call, S_0, X, rfr, vol, t):
    d1 = (np.log(S_0 / X) + (rfr + 0.5 * vol ** 2) * t) / (vol * np.sqrt(t))
    d2 = (np.log(S_0 / X) + (rfr - 0.5 * vol ** 2) * t) / (vol * np.sqrt(t))
    
    if Put_Call == "C":
        opt_price = S_0*si.norm.cdf(d1) - X*np.exp(-rfr*t)*si.norm.cdf(d2)
    elif Put_Call == "P":
        opt_price = X*np.exp(-rfr*t)*si.norm.cdf(-d2) - S_0*si.norm.cdf(-d1)
    
    return opt_price

##############################################################################
# Error
##############################################################################
def Err(Put_Call, n, S_0, X, rfr, vol, t, AMN_EUR):
    e = BSM(Put_Call,S_0,X,rfr,vol,t) - CRR(Put_Call,n,S_0,X,rfr,vol,t,AMN_EUR)
    
    return e

def Err_1(Put_Call, n, S_0, X, rfr, vol, t, AMN_EUR):
    e_1 = BSM(Put_Call,S_0,X,rfr,vol,t) - BBS(Put_Call,n,S_0,X,rfr,vol,t,AMN_EUR)
    
    return e_1

def Err_2(Put_Call, n, S_0, X, rfr, vol, t, AMN_EUR):
    e_2 = BSM(Put_Call,S_0,X,rfr,vol,t) - BBSR(Put_Call,n,S_0,X,rfr,vol,t,AMN_EUR)
    
    return e_2

##############################################################################
# Comparing Convergence of BBS-BSM and BBSR-BSM
# Plotting Error v Number of Steps
##############################################################################
S_0 = 100
rfr = 0.05
vol = 0.3
X = 90
t = 0.5

N = np.linspace(2,100,99)
err_crr = [Err("P", int(n_steps), S_0, X, rfr, vol, t, "E") for n_steps in N]
err_bbs = [Err_1("P", int(n_steps), S_0, X, rfr, vol, t, "E") for n_steps in N]
err_bbsr = [Err_2("P", int(n_steps), S_0, X, rfr, vol, t, "E") for n_steps in N]

#Error Plot CRR-BBS
fig, ax = plt.subplots()
plt.plot(N, err_crr, 'r-', label = 'CRR Error')
plt.plot(N, err_bbs, 'k-', label = 'BBS Error')
legend = ax.legend(loc='lower right', shadow=False, fontsize='large')
plt.title("Pricing Error", fontsize=20) 
plt.xlabel("Steps", fontsize=18) 
plt.ylabel("Error", fontsize=18)

plt.show()

#Error Plot CRR-BBSR
fig, ax = plt.subplots()
plt.plot(N, err_crr, 'b-', label = 'CRR Error')
plt.plot(N, err_bbsr, 'k-', label = 'BBSR Error')
legend = ax.legend(loc='lower right', shadow=False, fontsize='large')
plt.title("Pricing Error", fontsize=20) 
plt.xlabel("Steps", fontsize=18) 
plt.ylabel("Error", fontsize=18)

plt.show()

#Error Plot BBS-BBSR
fig, ax = plt.subplots()
plt.plot(N, err_bbsr, 'k-', label = 'BBSR Error')
plt.plot(N, err_bbs, 'b-', label = 'BBS Error')
legend = ax.legend(loc='lower right', shadow=False, fontsize='large')
plt.title("Pricing Error", fontsize=20) 
plt.xlabel("Steps", fontsize=18) 
plt.ylabel("Error", fontsize=18)

plt.show()
    
##############################################################################
# RMSE
##############################################################################
opt_actual = np.full((100, 1), BSM("P", S_0, X, rfr, vol, t))
opt_approx = np.zeros((100, 1))

# CRR
for i in range(0,99):
    opt_approx[i,0] = CRR("P", i+1, S_0, X, rfr, vol, t, "E")
   
rms = round(np.sqrt(mean_squared_error(opt_actual, opt_approx)),6)
print("The RMSE of the CRR approximation for 100 steps is"\
      " {}.".format(rms))

# BBS
for i in range(0,99):
    opt_approx[i,0] = BBS("P", i+1, S_0, X, rfr, vol, t, "E")
 
rms = round(np.sqrt(mean_squared_error(opt_actual, opt_approx)),6)
print("The RMSE of the BBS approximation for 100 steps is"\
      " {}.".format(rms))

# BBSR
for i in range(0,99):
    opt_approx[i,0] = BBSR("P", i+2, S_0, X, rfr, vol, t, "E")
    
rms = round(np.sqrt(mean_squared_error(opt_actual, opt_approx)),6)
print("The RMSE of the BBSR approximation for 100 steps is"\
      " {}".format(rms))
    
start_time = time.time()   
v = BBSR("P", 1000, S_0, X, rfr, vol, t, "E")
end_time = time.time()
e_1 = BSM("P",S_0,X,rfr,vol,t) - BBSR("P", 1000, S_0, X, rfr, vol, t, "E")
run_time = round(end_time - start_time,6)
print("The pricing error of the BBSR approximation for 1000 steps is"\
      " {} and code execution takes {} seconds.".format(e_1,run_time))
    
start_time = time.time()   
v = BBS("P", 1000, S_0, X, rfr, vol, t, "E")
end_time = time.time()

e_2 = BSM("P",S_0,X,rfr,vol,t) - BBS("P", 1000, S_0, X, rfr, vol, t, "E")
run_time = round(end_time - start_time,6)

print("The pricing error of the BBS approximation for 1000 steps is"\
      " {} and code execution takes {} seconds.".format(e_2,run_time))

im = (e_2 - e_1)/e_1 * 100   
print("The improvement is {}%".format(im))