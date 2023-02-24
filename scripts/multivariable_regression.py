# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 10:16:50 2023

@author: au485969
"""

def rsquared(x, y):
    import scipy
    """ Return R^2 where x and y are array-like."""
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    return r_value**2

def multivariable_regression(fitting='E', 
                             included_parameters = ['c_hat','c1','c2','eta1','eta2','tau_SD'], 
                             threshold_E = 1,
                             print_pvals=False, 
                             colors=False, 
                             scaling = False, 
                             keep_high_eta1s=True):
    # import scipy
    import matplotlib.pyplot as plt
    plt.close('all')
    import pandas as pd
    import numpy as np
    import seaborn as sn
    sn.set_theme(style="ticks")
    import warnings
    warnings.filterwarnings('ignore')
    import statsmodels.api as sm
    # from sklearn.linear_model import LinearRegression
    
    
    xticklabels = []
    xtickdic = {'c_hat':r'$\hat{c}$', 
                'c1':r'$c_c$',
                'c2':r'$c_d$',
                'eta1':r'$\eta_c$',
                'eta2':r'$\eta_d$',
                'tau_SD':r'$\tau_{SD}$'}
    for p in included_parameters:
        xticklabels.append(xtickdic[p])
        
    #%% Read data
    file = 'results/sspace_w_sectorcoupling_wo_duplicates.csv'
    # file = 'results/sspace_3888.csv'
    
    if file == 'results/sspace_w_sectorcoupling_wo_duplicates.csv':
        sectors = ['T-H-I-B','T-H','-']
        sector_names = ['Fully sector-coupled','Electricity \n + Heating \n + Land Transport', 'Electricity']
        lc_dic = 'load_coverage [%]'
    else:
        sectors = ['']
        sector_names = ['Electricity']
        lc_dic = 'load_shift [%]'
    
    sspace_og = pd.read_csv(file,index_col=0)
    
    # Remove eta1 > 1
    if not keep_high_eta1s:
        sspace_man = sspace_og.T.drop(columns='sector').astype(float)
        sspace_man = sspace_man['eta1 [-]'][sspace_man['eta1 [-]'] < 1]
        sspace_og = sspace_og[sspace_man.index]
    
    # included_parameters = ['c_hat','c1','c2','eta1','eta2','tau_SD']
    # included_parameters = ['c_hat','c1','c2','eta1','eta2'] #'tau_SD'
    # included_parameters = ['c_hat','c1','c2','eta1','tau_SD'] #'eta2'
    # included_parameters = ['c_hat','c1','c2','eta2','tau_SD'] #'eta1'
    # included_parameters = ['c_hat','c1','eta1','eta2','tau_SD'] #'c2'
    # included_parameters = ['c_hat','c2','eta1','eta2','tau_SD'] #'c1'
    # included_parameters = ['c2','eta1','eta2','tau_SD'] #'c_hat'
    
    # params_E = np.zeros([len(sectors),len(included_parameters)])
    # pvalues_E = np.zeros([len(sectors),len(included_parameters)])
    params = np.zeros([len(sectors),len(included_parameters)])
    pvalues = np.zeros([len(sectors),len(included_parameters)])
    # params_SCR = np.zeros([len(sectors),len(included_parameters)])
    # pvalues_SCR = np.zeros([len(sectors),len(included_parameters)])
    
    # E_sector = pd.DataFrame(index=np.arange(1082))
    # Multivariate regression using GLM for all sectors
    i = 0
    for sector in sectors:
        print('Sector: ', sector)
        if file == 'results/sspace_w_sectorcoupling_wo_duplicates.csv':
            sspace = sspace_og.T
            sspace['sector'] = sspace['sector'].fillna('-')
            sspace = sspace.query('sector == @sector')
            sspace = sspace.drop(columns='sector').astype(float).T
        else:
            sspace = sspace_og.copy()
        
        sspace = sspace[sspace.loc['E [GWh]'][sspace.loc['E [GWh]'] > threshold_E].index]
        
        # Input
        df1 = pd.DataFrame(columns=['c_hat'])
        df1['c_hat'] = sspace.loc['c_hat [EUR/kWh]'].astype(float)
        df1['c1'] = sspace.loc['c1'].astype(float)
        df1['eta1'] = sspace.loc['eta1 [-]'].astype(float)
        df1['c2'] = sspace.loc['c2'].astype(float)
        df1['eta2'] = sspace.loc['eta2 [-]'].astype(float)
        df1['tau_SD'] = sspace.loc['tau [n_days]'].astype(float)
        
        # Output
        df1['E_cor'] = sspace.loc['E [GWh]'].astype(float)*df1['eta2']
        df1['LC'] = sspace.loc[lc_dic].astype(float)
        df1['SCR'] = sspace.loc['c_sys [bEUR]'].astype(float)/(sspace.loc['c_sys [bEUR]'].astype(float).max())
        
        df1 = df1.sort_values(['c_hat','c1','eta1','c2','eta2','tau_SD'])
        # df1 = df1.query('E_cor > 1')
        
        #%%
        # Regression
        df_data = df1[included_parameters] 
        X = df_data
        
        # X_noscaling = X # We do not scale them before the regression. This is to compare the standardized regression coefficients instead
        # X_withscaling = (X - X.min())/(X.max()-X.min()) # We do a min-max scaling of the parameters

        if scaling:
            # X_scaling = (X - X.min())/(X.max()-X.min()) # We do a min-max scaling of the parameters
            X_scaling = (X - X.mean())/(X.std())
            y = (df1['E_cor'] - df1['E_cor'].mean())/(df1['E_cor'].std())
        else:
            X_scaling = X # We do not scale them before the regression. This is to compare the standardized regression coefficients instead
            y = df1['E_cor'].values
            
        if fitting == 'E':
            X_val = sm.add_constant(X_scaling.values)
            # est = sm.GLM(y, X_val) # Using Generalized Linear Model for regression
            est = sm.OLS(y, X_val) # Leads to the same output as GLM but contains more statistical information
            est2 = est.fit()
            
            print(est2.summary())
            
            beta = est2.params[1:] #/(np.abs(est2.params[1:]).max()))
            pvalues[i,:] = est2.pvalues[1:]
            
            beta_standardized = beta*X.std()/y.std()
            
            if scaling:
                params[i,:] = abs(beta)
            else:
                params[i,:] = abs(beta_standardized)
            
            # print('R-squared for energy capacity fit: ')
            # print(rsquared(y,est2.predict(X_val)).round(3))
        
            # model = LinearRegression()
            # model.fit(X_val, y)
            # R2 = model.score(X_val,y)
            # R2_adjusted = 1 - (len(y) - 1)*(1 - R2)/(len(y) - X_val.shape[1] - 1)
            
            # print('R2 = ',R2.round(3))
            # print('R2_adjusted = ',R2_adjusted.round(3))
            
            # F = (len(y) - X_val.shape[1] - 1)*(1/(1 - R2) - 1)/X_val.shape[1]
            
            # #display adjusted R-squared
            # 1 - (1-model.score(X_val, y))*(len(y)-1)/(len(y)-X_val.shape[1]-1)
              # 1 - (n − 1)(1 − R2)/(n − k − 1)
            
            # A = np.identity(len(est2.params))
            # print(est2.f_test(A))
        #%%
        if fitting == 'LC':
            y = df1['LC'].values
            X_val = sm.add_constant(X_scaling.values)
            est = sm.GLM(y, X_val) # Using Generalized Linear Model for regression
            est2 = est.fit()
            
            beta = est2.params[1:]
            beta_standardized = beta*X.std()/y.std()
            params[i,:] = abs(beta_standardized) # est2.params[1:] #/(np.abs(est2.params[1:]).max()))
            pvalues[i,:] = est2.pvalues[1:]
            
            # print('R-squared for load coverage fit: ')
            # print(rsquared(y,est2.predict(X_val)).round(3))
        
        if fitting == 'SCR':
            # System cost reduction
            y = (1 - df1['SCR'].values)*100
            X_val = sm.add_constant(X_scaling.values)
            est = sm.GLM(y, X_val) # Using Generalized Linear Model for regression
            est2 = est.fit()
            
            beta = est2.params[1:]
            beta_standardized = beta*X.std()/y.std()
            params[i,:] = abs(beta_standardized) #est2.params[1:] #/(np.abs(est2.params[1:]).max()))
            pvalues[i,:] = est2.pvalues[1:]
            
            # print('R-squared for system cost reduction fit: ')
            # print(rsquared(y,est2.predict(X_val)).round(3))
        
        print('')
        i += 1
    
    #%% Plotting coefficients of the GLM estimating "E_cor"
    # if fitting == 'E':
    fig,ax = plt.subplots(figsize=[10,6])
    
    extent = [0, len(included_parameters), 0, len(sectors)]
    if colors:
        if file == 'results/sspace_w_sectorcoupling_merged.csv':
            ax.imshow(np.array([params[2],params[1],params[0]]), cmap="Reds",extent=extent)
        else:
            ax.imshow(np.array(params), cmap="Reds",extent=extent)
        
    ax.set_xticks(np.arange(len(included_parameters))+0.5)
    ax.set_xticklabels(xticklabels,fontsize=18)
    ax.set_yticks(np.arange(len(sectors))+0.5)
    ax.set_yticklabels(sector_names,fontsize=18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    for i in range(len(sectors)):
        for j in range(len(included_parameters)):
            if params[i,j] < 0.2:
                col = 'k'
            else:
                col = 'white'
                
            ax.text(0.5+j,0.5+i,str(params[i,j].round(2)),fontsize=18, horizontalalignment='center', verticalalignment = 'center', color=col)
            pvalue_float = pvalues[i,j]
            if pvalue_float < 1e-3:
                pvalue_str = 'p < ' + r'$10^{-3}$'
            else:
                pvalue_str = 'p = ' + str(pvalue_float.round(3))
            
            if print_pvals:
                ax.text(0.5+j,0.3+i,pvalue_str,fontsize=15, horizontalalignment='center', verticalalignment = 'center',color='grey')
    
    # ax.grid(axis='both')
    
    # fig.savefig('figures/GLM_coefficients_sectors_E.png',
    #             bbox_inches="tight",dpi=300)
    
    #%% Plotting coefficients of the GLM estimating "LC"
    # if fitting == 'LC':
    #     fig,ax = plt.subplots(figsize=[10,6])
        
    #     extent = [0, 5, 0, len(sectors)]
    #     if colors:
    #         if file == 'results/sspace_w_sectorcoupling_merged.csv':
    #             ax.imshow(np.array([params_LC[2],params_LC[1],params_LC[0]]), cmap="cool",extent=extent)
    #         else:
    #             ax.imshow(np.array(params_LC), cmap="cool",extent=extent)
        
    #     ax.set_xticks(np.arange(5)+0.5)
    #     ax.set_xticklabels([r'$\hat{c}$', r'$c_c$',r'$c_d$',r'$\eta_c$',r'$\eta_d$'],fontsize=18)
    #     ax.set_yticks(np.arange(len(sectors))+0.5)
    #     ax.set_yticklabels(sector_names,fontsize=18)
    #     ax.spines['top'].set_visible(False)
    #     ax.spines['right'].set_visible(False)
    #     ax.spines['left'].set_visible(False)
        
    #     for i in range(len(sectors)):
    #         for j in range(5):
    #             ax.text(0.5+j,0.5+i,str(params_LC[i,j].round(2)),fontsize=18, horizontalalignment='center', verticalalignment = 'center')
    #             pvalue_float = pvalues_LC[i,j]
    #             if pvalue_float < 1e-3:
    #                 pvalue_str = 'p < ' + r'$10^{-3}$'
    #             else:
    #                 pvalue_str = 'p = ' + str(pvalue_float.round(3))
                    
    #             if print_pvals:
    #                 ax.text(0.5+j,0.3+i,pvalue_str,fontsize=15, horizontalalignment='center', verticalalignment = 'center',color='grey')
        
    #     ax.grid(axis='both')
        
    #     fig.savefig('figures/GLM_coefficients_sectors_LC.png',
    #                 bbox_inches="tight",dpi=300)
    
    # #%% Plotting coefficients of the GLM estimating "SCR"
    # if fitting == 'SCR':
    #     fig,ax = plt.subplots(figsize=[10,6])
        
    #     extent = [0, 5, 0, len(sectors)]
    #     if colors:
    #         if file == 'results/sspace_w_sectorcoupling_merged.csv':
    #             ax.imshow(np.array([params_SCR[2],params_SCR[1],params_SCR[0]]), cmap="cool",extent=extent)
    #         else:
    #             ax.imshow(np.array(params_SCR), cmap="cool",extent=extent)
        
    #     ax.set_xticks(np.arange(5)+0.5)
    #     ax.set_xticklabels([r'$\hat{c}$', r'$c_c$',r'$c_d$',r'$\eta_c$',r'$\eta_d$'],fontsize=18)
    #     ax.set_yticks(np.arange(len(sectors))+0.5)
    #     ax.set_yticklabels(sector_names,fontsize=18)
    #     ax.spines['top'].set_visible(False)
    #     ax.spines['right'].set_visible(False)
    #     ax.spines['left'].set_visible(False)
        
    #     for i in range(len(sectors)):
    #         for j in range(5):
    #             ax.text(0.5+j,0.5+i,str(params_SCR[i,j].round(2)),fontsize=18, horizontalalignment='center', verticalalignment = 'center')
    #             pvalue_float = pvalues_LC[i,j]
    #             if pvalue_float < 1e-3:
    #                 pvalue_str = 'p < ' + r'$10^{-3}$'
    #             else:
    #                 pvalue_str = 'p = ' + str(pvalue_float.round(3))
                
    #             if print_pvals:
    #                 ax.text(0.5+j,0.3+i,pvalue_str,fontsize=15, horizontalalignment='center', verticalalignment = 'center',color='grey')
        
    #     ax.grid(axis='both')
        
    #     fig.savefig('figures/GLM_coefficients_sectors_SCR.png',
    #                 bbox_inches="tight",dpi=300)
        
