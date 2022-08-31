# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 14:27:51 2022

@author: au485969
"""
import seaborn as sns
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
import warnings
warnings.filterwarnings('ignore')
plt.close('all')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%% Define here the system %%%%%%%%%%%%%%%%%%%%%%%%%%%%
sector = 0 # no sector-coupling
# sector = 'T-H' # Transportation and heating
# sector = 'T-H-I-B' # Transportation, heating, industry (elec + feedstock), and biomass

# filename = '../results/sspace.csv'
# filename = '../results/sspace_3888.csv'
filename = '../results/sspace_w_sectorcoupling.csv'
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Plotting layout
fs = 17
plt.style.use('seaborn-ticks')
plt.rcParams['axes.labelsize'] = fs
plt.rcParams['xtick.labelsize'] = fs
plt.rcParams['ytick.labelsize'] = fs
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['axes.axisbelow'] = True

# Plotting functions
def plot_matrix(nrows,ncols,indices,configs,title,ticklab,ref,scal=1,swplot=False):
    fig,axes = plt.subplots(nrows=nrows,ncols=ncols,figsize=(18, 14))
    axes = axes.flatten()
    j = 0
    leg = {}
    titles = {}
    for var in indices:
        i = 0 
        df_list = []
        titles[j] = var[0:3]
        for ind in configs:
            if len(x[ind].loc[var].shape) > 1:
                plot = x[ind].loc[var].iloc[0]/scal
                if i == 0:
                    ref_i = ref.loc[var].iloc[0]/scal
            else:
                plot = x[ind].loc[var]/scal
                if i == 0:
                    ref_i = ref.loc[var]/scal
            
            leg[i] = axes[j].violinplot(plot,positions=[i])
            if i == 0:
                axes[j].scatter(i,ref_i,color='red',marker='X',zorder=60)
            
            if swplot == True:
                datfr_i = pd.DataFrame()
                datfr_i['x'] = [i]*len(x[ind].loc[var])
                datfr_i['data'] = x[ind].loc[var].values/scal
                df_list.append(datfr_i)
            
            i += 1
            
        if swplot == True:
            datfr = pd.concat(df_list)
            sns.swarmplot(x='x',y='data', data=datfr, ax=axes[j],size=1.5)
            axes[j].set_xlabel('')
            axes[j].set_ylabel('')
    
        if (title != 'AC Lines usage') and (title != 'DC Links usage'):
            if j > 0:
                if titles[j] == titles[j-1]:
                    axes[j].set_title(var[0:3] + ' (2)',fontsize=fs,color='grey')
                else:
                    axes[j].set_title(var[0:3],fontsize=fs,color='grey')
            else:
                axes[j].set_title(var[0:3],fontsize=fs,color='grey')
        else:
            axes[j].set_title(var,fontsize=fs,color='grey')
            
            
        axes[j].set_xticks(np.arange(len(configs)))
        
        if j > (nrows-1)*ncols-1:
            axes[j].set_xticklabels(ticklab,rotation = 90)
        else:
            axes[j].set_xticklabels([])
        
        j += 1
            
    if j <= nrows*ncols-1:
            for jj in np.arange(j,nrows*ncols):
                fig.delaxes(axes[jj])
                
    fig.subplots_adjust(wspace=0.5, hspace=0.4)
    fig.suptitle(title,fontsize=fs,fontweight='bold',y=0.96)
     
    return fig

def scatter_hist(x, y, ax_histx, ax_histy): #,bw1,bw2):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # now determine nice limits by hand:
    binwidth = 0.01*(x.max() - x.min()) # bw1
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth
    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, color='grey', bins=bins)
    
    binwidth = 0.01*(y.max() - y.min()) # bw2
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth
    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histy.hist(y, orientation='horizontal',color='grey', bins=bins)

# Setup
df = pd.read_csv(filename,index_col=0).fillna(0)
df_T = df.T
df_T = df_T.query('sector == @sector').drop(columns='sector').astype(float)
df_T = df_T.sort_values(by='c_hat [EUR/kWh]')
df_T['E [TWh]'] = df_T['E [GWh]']*1e-3 # convert GWh to TWh
df_T = df_T.loc[df_T['eta1 [-]'][df_T['eta1 [-]'] <= 1].index]
df = df_T.reset_index(drop=True).T
threshold_E = 0 #2000 # GWh
colorcase = 'wgc'
# colorcase = 'wogc'
if filename == '../results/sspace.csv' or filename == '../results/sspace_3888.csv':
    var = 'load_shift [%]'
    var_str = 'load_shift'
else:
    var = 'load_coverage [%]'
    var_str = 'load_coverage'

x_low = df[df.loc['E [GWh]'].idxmin()]
x = df[df.loc['E [GWh]'][df.loc['E [GWh]'] > threshold_E].index]

# Reference store
r_1 = df[df.loc['eta1 [-]'][df.loc['eta1 [-]'] == 0.5].index]
r_11 = r_1[r_1.loc['eta2 [-]'][r_1.loc['eta2 [-]'] == 0.5].index]
if filename == '../results/sspace_3888.csv':
    r_111 = r_11[r_11.loc['c_hat [EUR/kWh]'][r_11.loc['c_hat [EUR/kWh]'] == 1.997].index]
else:
    r_111 = r_11[r_11.loc['c_hat [EUR/kWh]'][r_11.loc['c_hat [EUR/kWh]'] == 2].index]
r_1111 = r_111[r_111.loc['c1'][round(r_111.loc['c1']) == 350].index]
r_11111 = r_1111[r_1111.loc['c2'][round(r_1111.loc['c2']) == 350].index]
r_111111 = r_11111[r_11111.loc['tau [n_days]'][r_11111.loc['tau [n_days]'] == 30].index]


Xs = {'eta1 [-]':x.loc['eta1 [-]'].unique(), 'eta2 [-]':x.loc['eta2 [-]'].unique(), 
      'c1':x.loc['c1'].unique(),'c2':x.loc['c2'].unique(),'c_hat [EUR/kWh]':x.loc['c_hat [EUR/kWh]'].unique(),
      'tau [n_days]':x.loc['tau [n_days]'].unique()}

X_name_dict = {'eta1 [-]':r'$\eta_c$','eta2 [-]':r'$\eta_d$','c1':r'$c_c$','c2':r'$c_d$','c_hat [EUR/kWh]':r'$\hat{c}$','tau [n_days]':r'$\tau_{SD}$'}
X_name_short_dict = {'RTE':'RTE','eta1 [-]':'eta1','eta2 [-]':'eta2','c1':'c1','c2':'c2','c_hat [EUR/kWh]':'chat','tau [n_days]':'tau'}
yval = 'lc'
ylab = {'dt':r'$\frac{E}{G_d}$' + ' [h]','lc':'Load coverage ' + r'$LC^X$' + ' [%]','c_sys':'System cost [bEUR]','c1':'Charge capacity cost [EUR/kW]'}

#%% Plot system cost and curtailment
data_max = 0.95*0.95 # Max RTE used for normalization
cmap1 = plt.cm.get_cmap('summer_r')
xlab = {'c_sys [bEUR]': 'Normalized system cost','battery_' + var_str +' [%]':'Battery load coverage ' + r'$LC^B$' + ' [%]','curtailment':'Curtailed energy [%]'}
for X_name in ['RTE']: 
    for t in ['c_sys [bEUR]','curtailment']: 
        print(t)
        
        # Electricity mix - Relative values
        fig = plt.figure(figsize=(8, 7))
        nrows = 5
        ncols = 5
        gs = gridspec.GridSpec(nrows, ncols)
        gs.update(wspace=0)
        gs.update(hspace=0)
        t_it = 0
        
        cp = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
        if t == 'curtailment':
            avail_energy = x.loc['avail_solar [MWh]'] + x.loc['avail_onwind [MWh]'] + x.loc['avail_offwind [MWh]']
            used_energy = x.loc['used_solar [MWh]'] + x.loc['used_onwind [MWh]'] + x.loc['used_offwind [MWh]']
            gen_tech_rel_thres = (avail_energy - used_energy)/avail_energy*100
            
        else:
            if t == 'c_sys [bEUR]':
                gen_tech_rel_all = df.loc[t]/df.loc[t].loc[r_111111.T.index].item()
                gen_tech_rel_thres = x.loc[t]/df.loc[t].loc[r_111111.T.index].item()
            else:
                gen_tech_rel_all = df.loc[t]
                gen_tech_rel_thres = x.loc[t]

        dt_thres = x.loc['E [GWh]']/x.loc['G_discharge [GW]']
        lc_thres = x.loc[var]
        
        ax = plt.subplot(gs[1:,0:-1])
        ax.grid()
        ax.set_ylabel(ylab[yval])
    
        x_plot = gen_tech_rel_thres
        
        if yval == 'lc':    
            y_plot = lc_thres
        else:
            y_plot = dt_thres
        
        z = x_plot + y_plot
        if colorcase == 'wgc':
            if X_name == 'RTE':
                RTEs = (df.loc['eta1 [-]']*df.loc['eta2 [-]']).unique()
                RTEs.sort()
                leg = {}
                it = 0
                for X1 in Xs['eta2 [-]']:
                    x_eta2 = x.T.groupby(x.T['eta2 [-]']).get_group(X1)
                    for X2 in x_eta2['eta1 [-]'].unique():
                        x_eta1 = x_eta2.groupby(x.T['eta1 [-]']).get_group(X2)
                        RTE = X1*X2
                        gen_tech_rel_x = x_plot.loc[x_eta1.index] 
                        dt_x = x_eta1['E [GWh]']/x_eta1['G_discharge [GW]']
                        lc_x = x_eta1[var] 
                        data_normalized1 = RTE/data_max
                        color1 = cmap1(data_normalized1)

                        if yval == 'lc':
                            leg[it] = ax.plot(gen_tech_rel_x,lc_x,'.',markersize=3,alpha=1,color=color1,zorder=1000-RTE)
                            
                        else:
                            leg[it] = ax.plot(gen_tech_rel_x,dt_x,'.',markersize=3,alpha=1,color=color1,zorder=1000-RTE)
                        it += 1
                
                if t == 'battery_' + var_str + ' [%]':
                    z1 = z.loc[x_plot[x_plot > 0.03].index]
                    ax.plot(x_plot.loc[z1.index],z1,'.',color='k',markersize=1)
                    xlim = ax.get_xlim()
                    sum_lc = ax.plot(np.ones(len(z1))*(-20),z1,'.',color='k',markersize=5)
                    ax.set_xlim(xlim)
                    ax.text(1,8,'Additional storage',fontsize=fs,color='grey')
                    ax.text(1,6,'Battery substitution',fontsize=fs,color='grey')
                        
                cb_ax = fig.add_axes([0.91,0.11,0.02,0.16])

                cmap = mpl.cm.summer_r
                cmaplist = [cmap(i) for i in range(cmap.N)]
                
                cmap = mpl.colors.LinearSegmentedColormap.from_list(
                'Custom cmap', cmaplist, cmap.N)
            
                # define the bins and normalize
                bounds = [0] + list((RTEs*100).astype(int))
                norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
                cb1 = mpl.colorbar.ColorbarBase(cb_ax, orientation='vertical',cmap=cmap, norm=norm,
                    spacing='proportional', ticks=bounds, boundaries=bounds, format='%1i')
                
                cb1.set_ticks([9-(9)/2,25-(25-19)/2,48-(48-32)/2,62-(62-48)/2,90-(90-62)/2])
                cb1.set_ticklabels(['9','25','48','62','90'])
                cb1.ax.tick_params(labelsize=15)
                fig.text(0.91, 0.28, r'RTE [%]',fontsize=15)
                
            else:
                leg = {}        
                for X in Xs[X_name]:
                    x_eta2 = x.T.groupby(x.T[X_name]).get_group(X)
                    gen_tech_rel_x = x_plot.loc[x_eta2.index] 
                    dt_x = x_eta2['E [GWh]']/x_eta2['G_discharge [GW]']
                    lc_x = x_eta2[var] 
                    if yval == 'lc':
                        leg[X] = ax.plot(gen_tech_rel_x,lc_x,'.',markersize=3,label=X_name + ' =' + str(X),alpha=1)
                    else:
                        leg[X] = ax.plot(gen_tech_rel_x,dt_x,'.',markersize=3,label=X_name + ' =' + str(X),alpha=1)
        else:
            ax.plot(x_plot,y_plot,'.',markersize=0.5,color='k',alpha=0.5,zorder=1)
            
        if t != 'curtailment' and t != 'c_sys [bEUR]' and t != 'Offwind' and t != 'Solar' and t != 'Onwind':
            sm4 = ax.scatter(r_111111.loc[t],r_111111.loc[var],marker='X',color='r',zorder=1000)
        elif t == 'curtailment':
            avail_energy = df.loc['avail_solar [MWh]'] + df.loc['avail_onwind [MWh]'] + df.loc['avail_offwind [MWh]']
            used_energy = df.loc['used_solar [MWh]'] + df.loc['used_onwind [MWh]'] + df.loc['used_offwind [MWh]']
            curt = (avail_energy.loc[r_111111.T.index.item()] - used_energy.loc[r_111111.T.index.item()])/avail_energy.loc[r_111111.T.index.item()]*100
            sm4 = ax.scatter(curt,r_111111.loc[var],marker='X',color='r',zorder=1000)
        elif t == 'c_sys [bEUR]':
            sm4 = ax.scatter(r_111111.loc[t]/df.loc[t].loc[r_111111.T.index],r_111111.loc[var],marker='X',color='r',zorder=1000)
            ax.set_xlim([0.9,1.01])
            ax.set_xticks(np.arange(0.9,1.01,0.01))
            ax.set_xticklabels([0.9,'','','','',
                                0.95,'','','','',
                                1])
        elif t == 'Offwind' or t == 'Solar' or t == 'Onwind':
            sm4 = ax.scatter(gen_tech_rel_all[t].loc[r_111111.T.index.item()],r_111111.loc[var],marker='X',color='r',zorder=1000)
        
        if t != 'Offwind' and t != 'Solar' and t != 'Onwind':
            ax.set_xlabel(xlab[t])
        else:
            ax.set_xlabel('% ' + t)
        
        ax.set_ylim([-0.2,17.5])    
        ax.set_ylim([-0.2,ax.get_ylim()[1]])
        
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()
        
        ax_histx = plt.subplot(gs[0:1,0:-1],sharex=ax)
        ax_histy = plt.subplot(gs[1:,-1],sharey=ax)
        
        scatter_hist(x_plot, y_plot, ax_histx, ax_histy)
        
        ax_histx.spines['bottom'].set_visible(False)
        ax_histx.spines['top'].set_visible(False)
        ax_histx.spines['right'].set_visible(False)
        ax_histx.spines['left'].set_visible(False)
        
        ax_histy.spines['bottom'].set_visible(False)
        ax_histy.spines['top'].set_visible(False)
        ax_histy.spines['right'].set_visible(False)
        ax_histy.spines['left'].set_visible(False)
        
        ax_histx.axis('off')
        ax_histy.axis('off')
        
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        
        if colorcase == 'wgc':
            if X_name == 'c_hat [EUR/kWh]':
                fig.legend(leg[Xs[X_name][0]]+leg[Xs[X_name][1]]+leg[Xs[X_name][2]]+leg[Xs[X_name][3]]+
                            leg[Xs[X_name][4]]+leg[Xs[X_name][5]]+leg[Xs[X_name][6]]+leg[Xs[X_name][7]]+
                            leg[Xs[X_name][8]]+leg[Xs[X_name][9]]+leg[Xs[X_name][10]],
                            [X_name_dict[X_name] + ' = ' + str(Xs[X_name][0]),
                            X_name_dict[X_name] + ' = ' + str(Xs[X_name][1]),X_name_dict[X_name] + ' = ' + str(Xs[X_name][2]),
                            X_name_dict[X_name] + ' = ' + str(Xs[X_name][3]),X_name_dict[X_name] + ' = ' + str(Xs[X_name][4]),
                            X_name_dict[X_name] + ' = ' + str(Xs[X_name][5]),X_name_dict[X_name] + ' = ' + str(Xs[X_name][6]),
                            X_name_dict[X_name] + ' = ' + str(Xs[X_name][7]),X_name_dict[X_name] + ' = ' + str(Xs[X_name][8]),
                            X_name_dict[X_name] + ' = ' + str(Xs[X_name][9]),X_name_dict[X_name] + ' = ' + str(Xs[X_name][10])],
                            prop={'size':15},bbox_to_anchor=(0.1, -0.125), loc=3,ncol=3,frameon=False,borderaxespad=0)
            elif X_name == 'eta2 [-]':
                fig.legend(leg[Xs[X_name][0]]+leg[Xs[X_name][1]]+leg[Xs[X_name][2]]+leg[Xs[X_name][3]],
                            [X_name_dict[X_name] + ' = ' + str(Xs[X_name][0]),X_name_dict[X_name] + ' = ' + str(Xs[X_name][1]),
                            X_name_dict[X_name] + ' = ' + str(Xs[X_name][2]),X_name_dict[X_name] + ' = ' + str(Xs[X_name][3])],prop={'size':15},bbox_to_anchor=(0.1, -0.125), loc=3,ncol=4,frameon=False,borderaxespad=0)

            elif X_name == 'c1' or X_name == 'c2':
                fig.legend(leg[Xs[X_name][0]]+leg[Xs[X_name][1]]+leg[Xs[X_name][2]],
                            [X_name_dict[X_name] + ' = ' + str(round(Xs[X_name][0])) + ' €/kW',X_name_dict[X_name] + ' = ' + str(round(Xs[X_name][1])) + ' €/kW',
                            X_name_dict[X_name] + ' = ' + str(round(Xs[X_name][2])) + ' €/kW'],prop={'size':15},bbox_to_anchor=(0.05, -0.05), loc=3,ncol=3,frameon=False,borderaxespad=0)
            elif X_name != 'RTE':
                fig.legend(leg[Xs[X_name][0]]+leg[Xs[X_name][1]]+leg[Xs[X_name][2]],
                            [X_name_dict[X_name] + ' = ' + str(round(Xs[X_name][0])) + ' days',X_name_dict[X_name] + ' = ' + str(round(Xs[X_name][1])) + ' days',
                            X_name_dict[X_name] + ' = ' + str(round(Xs[X_name][2])) + ' days'],prop={'size':15},bbox_to_anchor=(0.05, -0.05), loc=3,ncol=3,frameon=False,borderaxespad=0)

            else:
                if t == 'battery_' + var_str + ' [%]':
                    fig.legend(sum_lc + [sm4],[r'$LC^B + LC^X$','Reference'],prop={'size':15},bbox_to_anchor=(0.875, 0.4), loc=3,ncol=1,frameon=False,borderaxespad=0)
                else:
                    fig.legend([sm4],['Reference'],prop={'size':15},bbox_to_anchor=(0.875, 0.4), loc=3,ncol=1,frameon=False,borderaxespad=0)
        fig.savefig('../figures/' + X_name_short_dict[X_name] + '_' + t + '_' + yval + '_' + colorcase + str(sector) + '.png', transparent=True,
                    bbox_inches="tight",dpi=300)
