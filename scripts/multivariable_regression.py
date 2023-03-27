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
                             csv_file='sspace_w_sectorcoupling_wo_duplicates.csv',
                             print_pvals=False, 
                             colors=True, 
                             scaling = True, 
                             keep_high_eta1s=False):
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
    import scipy.stats as stats
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
    file = 'results/' + csv_file
    
    if file == 'results/sspace_w_sectorcoupling_wo_duplicates.csv':
        sectors = ['T-H-I-B','T-H','-']
        sector_names = ['Fully sector-coupled (SC3)','Electricity \n + Heating \n + Land Transport (SC2)', 'Electricity (SC1)']
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

    params = np.zeros([len(sectors),len(included_parameters)])
    pvalues = np.zeros([len(sectors),len(included_parameters)])

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
        df1['E'] = sspace.loc['E [GWh]'].astype(float)*df1['eta2']
        
        df1['log_E'] = np.log(df1['E'])
        
        df1['LC'] = sspace.loc[lc_dic].astype(float)
        system_cost_norm = sspace.loc['c_sys [bEUR]'].astype(float)/(sspace.loc['c_sys [bEUR]'].astype(float).max())
        df1['SCR'] = (1 - system_cost_norm.values)*100
        
        # df1['log_SCR'] = np.log(df1['SCR'])
        
        df1 = df1.sort_values(['c_hat','c1','eta1','c2','eta2','tau_SD'])
        # df1 = df1.query('E_cor > 1')
        
        #%%
        # Regression
        df_data = df1[included_parameters] 
        X = df_data
        
        # X_noscaling = X # We do not scale them before the regression. This is to compare the standardized regression coefficients instead
        # X_withscaling = (X - X.min())/(X.max()-X.min()) # We do a min-max scaling of the parameters            
        
        # if fitting == 'SCR':
        #     # System cost reduction
        #     # X_scaling = X
        #     X_scaling = (X - X.min())/(X.max()-X.min())
        #     y_init = (1 - df1['SCR'].values)*100
        #     y = (y_init - y_init.min())/(y_init.max()-y_init.min())

        # else:
        if scaling:
            # Normalization
            X_scaling = (X - X.min())/(X.max()-X.min()) # We do a min-max scaling of the parameters
            y = (df1[fitting] - df1[fitting].min())/(df1[fitting].max() - df1[fitting].min())
            # y = df1[fitting]
            
            # Standardization
            # X_scaling = (X - X.mean())/(X.std())
            # y = (df1[fitting] - df1[fitting].mean())/(df1[fitting].std())
        
        else:
            X_scaling = X # We do not scale them before the regression. This is to compare the standardized regression coefficients instead
            y = df1[fitting].values
            
        X_val = sm.add_constant(X_scaling.values)
        # est = sm.GLM(y, X_val) # Using Generalized Linear Model for regression
        est = sm.OLS(y, X_val) # Leads to the same output as GLM but contains more statistical information
        est2 = est.fit()
        print(est2.summary())
        
        plt.figure()
        res = est2.resid # residuals
        fig = sm.qqplot(res, stats.t, fit=True, line="45")
        # plt.show()
        
        print('y_max = ', y.max())
        print('y_min = ', y.min())
        
        plt.figure()
        plt.plot(y,est2.predict(X_val),'.',markersize=3,alpha=0.3)
        plt.plot(y,y,'--',color='k')
        plt.xlabel(r'$\log E$')
        plt.ylabel(r'$\log \hat{E}$')
        plt.savefig('figures/GLM_fit_' + fitting + str(scaling) + '.png',
                bbox_inches="tight",dpi=300)
        # plt.xlim([0,max([max(y),max(est2.predict(X_val))])])
        # plt.ylim([0,max([max(y),max(est2.predict(X_val))])])
        
        beta = est2.params[1:]
        pvalues[i,:] = est2.pvalues[1:]
        beta_standardized = beta*X.std()/y.std()
        
        if scaling:
            params[i,:] = abs(beta)/max(abs(beta))
        else:
            params[i,:] = abs(beta_standardized)/max(abs(beta_standardized))
        
        print('')
        i += 1
    
    #%% Plotting coefficients of the GLM estimating "E_cor"
    # if fitting == 'E':
    fig,ax = plt.subplots(figsize=[10,6])
    
    extent = [0, len(included_parameters), 0, len(sectors)]
    if colors:
        color_threshold = 0.3
        if file == 'results/sspace_w_sectorcoupling_wo_duplicates.csv':
            ax.imshow(np.array([params[2],params[1],params[0]]), cmap="Reds",extent=extent)
        else:
            ax.imshow(np.array(params), cmap="Reds",extent=extent)
    else:
        color_threshold = 2
        if file == 'results/sspace_w_sectorcoupling_wo_duplicates.csv':
            ax.imshow(np.array([params[2],params[1],params[0]]), cmap="Reds",extent=extent,alpha=0)
        else:
            ax.imshow(np.array(params), cmap="Reds",extent=extent,alpha=0)
        
    ax.set_xticks(np.arange(len(included_parameters))+0.5)
    ax.set_xticklabels(xticklabels,fontsize=18)
    ax.set_yticks(np.arange(len(sectors))+0.5)
    ax.set_yticklabels(sector_names,fontsize=18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    added_R2_adj = [['+24.4pp','+10.5pp','+6.3pp','+11.4pp','+45.9pp','+1.6pp'],
                    ['+25.1','+13.7pp','+7.4pp','+9.6pp','+42.4pp','+1.8pp'],
                    ['+30.5pp','+16.0pp','+3.0pp','+8.5pp','+36.1pp','+0.3pp']]
    
    for i in range(len(sectors)):
        for j in range(len(included_parameters)):
            if params[i,j].round(2) < color_threshold:
                col = 'k'
            else:
                col = 'white'
                
            ax.text(0.5+j,0.5+i,str(params[i,j].round(2)),fontsize=18, horizontalalignment='center', verticalalignment = 'center', color=col)
            # pvalue_float = pvalues[i,j]
            # if pvalue_float < 1e-3:
            #     pvalue_str = 'p < ' + r'$10^{-3}$'
            # else:
            #     pvalue_str = 'p = ' + str(pvalue_float.round(3))
            
            # if print_pvals:
            #     ax.text(0.5+j,0.3+i,pvalue_str,fontsize=15, horizontalalignment='center', verticalalignment = 'center',color='grey')
            
            ax.text(0.5+j,0.3+i,added_R2_adj[i][j],fontsize=15, horizontalalignment='center', verticalalignment = 'center',color='grey')
    
    # ax.grid(axis='both')
    fig.savefig('figures/GLM_coefficients_sectors_' + fitting + str(scaling) + '.png',
                bbox_inches="tight",dpi=300)
    
