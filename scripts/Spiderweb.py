# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 11:27:10 2022

@author: au485969
"""

def set_rgrids(self, radii, labels=None, angle=None, #fmt=None,
                **kwargs):
    """
    Set the radial locations and labels of the *r* grids.
    The labels will appear at radial distances *radii* at the
    given *angle* in degrees.
    *labels*, if not None, is a ``len(radii)`` list of strings of the
    labels to use at each radius.
    If *labels* is None, the built-in formatter will be used.
    Return value is a list of tuples (*line*, *label*), where
    *line* is :class:`~matplotlib.lines.Line2D` instances and the
    *label* is :class:`~matplotlib.text.Text` instances.
    kwargs are optional text properties for the labels:
    %(Text)s
    ACCEPTS: sequence of floats
    """
    radii = self.convert_xunits(radii)
    radii = np.asarray(radii)
    self.set_yticks(radii)
    if labels is not None:
        self.set_yticklabels(labels)
    if angle is None:
        angle = self.get_rlabel_position()
    self.set_rlabel_position(angle)
    for t in self.yaxis.get_ticklabels():
        t.update(kwargs)
    return self.yaxis.get_gridlines(), self.yaxis.get_ticklabels()

def _invert(x, limits):
    """inverts a value x on a scale from
    limits[0] to limits[1]"""
    return limits[1] - (x - limits[0])

def _scale_data(data, ranges):
    """scales data[1:] to ranges[0],
    inverts if the scale is reversed"""
    # for d, (y1, y2) in zip(data[1:], ranges[1:]):
    # for d, (y1, y2) in zip(data, ranges):
    #     assert (y1 <= d <= y2) or (y2 <= d <= y1)

    x1, x2 = ranges[0]
    d = data[0]

    if x1 > x2:
        d = _invert(d, (x1, x2))
        x1, x2 = x2, x1

    sdata = [d]

    for d, (y1, y2) in zip(data[1:], ranges[1:]):
        if y1 > y2:
            d = _invert(d, (y1, y2))
            y1, y2 = y2, y1

        sdata.append((d-y1) / (y2-y1) * (x2 - x1) + x1)

    return sdata

class ComplexRadar():
    def __init__(self, fig, variables, var_fs, ranges, title,
                  n_ordinate_levels=5):
        angles = np.arange(30, 30+360, 360./len(variables))

        axes = [fig.add_axes([0.1,0.24,0.8,0.7],polar=True,
                label = "axes{}".format(i)) 
                for i in range(len(variables))]
        l, text = axes[0].set_thetagrids(angles, 
                                          labels=variables,fontsize=var_fs)
        [txt.set_rotation(angle-90) for txt, angle 
              in zip(text, angles)]
        for ax in axes[1:]:
            ax.patch.set_visible(False)
            ax.grid("off")
            ax.xaxis.set_visible(False)
        for i, ax in enumerate(axes):
            ax.patch.set_visible(False)
            grid = np.linspace(*ranges[i], 
                                num=n_ordinate_levels)
            gridlabel = ["{}".format(round(x,2)) 
                          for x in grid]
            if ranges[i][0] > ranges[i][1]:
                grid = grid[::-1] # hack to invert grid
                          # gridlabels aren't reversed
            gridlabel[0] = "" # clean up origin
            gridlabel[-1] = "" 
            
            ax.set_rgrids(grid, labels="",angle=angles[i],zorder=0)
            ax.set_ylim(*ranges[i])
            for k, spine in ax.spines.items():
                spine.set_zorder(-1)
            ax.set_yticklabels(gridlabel,fontsize=15,zorder=1)
            if i == 0:
                ax.set_title(title, ha='center',fontsize=var_fs, fontweight="bold", pad=15)
            
        # variables for plotting
        self.angle = np.deg2rad(np.r_[angles, angles[0]])
        self.ranges = ranges
        self.ax = axes[0]
        
    def plot(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.plot(self.angle, np.r_[sdata, sdata[0]], *args, **kw)
    def fill(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.fill(self.angle, np.r_[sdata, sdata[0]], *args, **kw)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
plt.close('all')



def plot_sspace(i, threshold_E, data_plot,csv,frame,colvar,radar,k,normfactor,lines,line_color,cmap,lw=0.5,factor=0.00005,zord=0): #0.00001,lw=0.1,zord=0):
    data_norm = csv.loc[i].loc[colvar]/normfactor
    # alpha = 1
    
    color = cmap(data_norm)
    if line_color == colvar:
        lcolor = color
    else:
        lcolor = line_color
    
    if colvar == 'E_cor':
        if csv.loc[i].loc[colvar] >= threshold_E:
            lcolor = line_color
            lw = 0.5
        else:
            lcolor = 'gray'
            lw = 0.5
            zord = 5
    
    if lines == 'w_lines':
        A = data_plot - (frame['range_upper'] - frame['range_lower'])*k*factor
        if zord == 0:
            radar.plot(A,alpha=1,zorder=10,color=lcolor,lw=lw)
        else:
            radar.plot(A,alpha=1,color=lcolor,lw=lw,zorder=zord)
    else:
        A = data_plot
        
    # if zord == 0:
    #     radar.fill(A,alpha=alpha,color=color,zorder=1+data_norm)
    # else:
    #     radar.fill(A,alpha=alpha,color=color,zorder=zord)

def plot_existing(data_plot_ex,frame,radar,color,marker,lw=2):
    radar.plot(data_plot_ex,color=color,lw=lw,zorder=100)
    radar.plot(data_plot_ex,color=color,lw=0,zorder=2000000,marker=marker)

def plot_spiderweb(scen = 'T-H-I-B', threshold_E = 2000, plot_ex = False, candidate = 'TES'):
    # ----------------------------------- Some settings -----------------------
    N_perfections = 6 # Number of concurrent "perfections"
    variables = (r"$\eta_d$" + ' [-]', r"$\hat{c}$" + '\n [€/kWh]', r"$c_c$" + '\n [€/kW]', r"$\eta_c$" + ' [-]', r"$\tau_{SD}$" + '\n [days]', r"$c_d$" + '\n [€/kW]')
    colorsectors = {'0':'Reds','T-H':'Reds','T-H-I-B':'Reds'}
    color_dic = {'0':'green',
                'T-H':'blue',
                'T-H-I-B':'orange'}
    # lines = 'wo_lines'
    lines = 'w_lines'
    # colvar = 'lc'
    colvar = 'E_cor' # The "corrected" energy capacity accounting for the discharge efficiency. 
                     # I.e., it is the energy capacity in units of dispatchable electricity.
    # line_color = 'k'
    line_color = color_dic[scen]
    # line_color = colvar
    # line_color = 'lc'
    # line_color = 'grey'
    tech = 'All'
    # flexbar = 'axincl'
    flexbar = 'axnotincl'
    # fig_save = 'main'
    fig_save = 'secondary'
    if colvar == 'lc':
        Ecaplab = 'wE'
    else:
        Ecaplab = 'woE'
        
    ex_stores_dic = {'aCAES':'aCAES_[1]', 
                     'LAES':'LAES_[1]', 
                     'TES':'TES_[2]', 
                     'PTES':'PTES_[1]', 
                     'MSES':'MSES_[1]', 
                     'RFB':'RFB_[2]',
                     'H2':'H2_[4]'}
    # -------------------------------------------------------------------------
    #%% Existing technologies 
    tech_cost = pd.read_csv('data/storage_techs_cost_USD.csv',sep=';',index_col=0,skiprows=[10,11], # 4
                            usecols = ['tech','eta_c','eta_d','c_c','c_d','c_hat','tau_SD'])
    tech_cost['eta_c'] /= 100
    tech_cost['eta_d'] /= 100
    USD_to_EUR = 0.96
    tech_cost['c_c'] *= USD_to_EUR
    tech_cost['c_d'] *= USD_to_EUR
    tech_cost['c_hat'] *= USD_to_EUR
    # ex_stores = tech_cost.index
    cols = ['eta_d','c_hat','c_c','eta_c','tau_SD','c_d']
    tech_cost.loc[tech_cost.query("tau_SD >= 60").index,['tau_SD']] = 60
    #%% Plotting
    if not ((tech != 'All') and (flexbar == 'axnotincl') and (fig_save == 'main') and (Ecaplab == 'woE')):
        if colvar == 'lc':
            clabel = 'LC [%]'
        else:
            clabel = 'E [TWh]'
        fs = 20
        var_fs = 25
    else:
        title = ''
        clabel = 'LC [%]'
        fs = 25
        var_fs = 40
    
    perfection = {'eta_c':0.95,
                  'eta_d':0.95,
                  'c_c':0.05,
                  'c_d':0.05,
                  'c_hat':0.025}
    
    tech_cost_norm = tech_cost.copy()
    tech_cost_norm['c_hat'] = tech_cost['c_hat']/40
    tech_cost_norm['tau_SD'] = tech_cost['tau_SD']/60
    tech_cost_norm['c_c'] = tech_cost['c_c']/700
    tech_cost_norm['c_d'] = tech_cost['c_d']/700 
    tech_cost_norm['eta_c'] = tech_cost['eta_c']
    tech_cost_norm['eta_d'] = tech_cost['eta_d'] 
    
    # c_c_range = (0,1) 
    # c_d_range = (0,1)
    eta_c_range = (0,1)
    eta_d_range = (0,1)
    # c_hat_range = (0,1)
    tau_SD_range = (0,1)
    c_c_range_plot = (-0.1,1.01)
    c_d_range_plot = (-0.1,1.01)
    c_hat_range_plot = (-0.1,1.01)
    eta_c_range_plot = (-0.1,1.01)
    eta_d_range_plot = (-0.1,1.01)
    tau_SD_range_plot = (-0.1,1.01)
    
    ranges = [eta_d_range_plot, c_hat_range_plot, c_c_range_plot, eta_c_range_plot, tau_SD_range_plot, c_d_range_plot]           
    sspace_new = pd.read_csv('results/sspace_w_sectorcoupling_merged.csv',index_col=0).fillna('0')
    csv = sspace_new.T.query('sector == @scen').drop(columns='sector').astype(float)
    
    csv['dt'] = csv['E [GWh]']/csv['G_discharge [GW]']
    csv = csv[['eta2 [-]','c_hat [EUR/kWh]','c1','eta1 [-]','tau [n_days]','c2','E [GWh]','load_coverage [%]','dt']]
    csv.columns = ['eta_d',
                    'c_hat', 
                    'c_c', 
                    'eta_c',
                    'tau_SD', 
                    'c_d', 
                    'E', 
                    'lc',
                    'dt']
    #%%
    csv = csv[csv['eta_c']*csv['eta_d'] < 1]
    # csv = csv[csv['eta_c'] <= 1]
    csv.reset_index(inplace=True)
    csv = csv.drop(columns='index')
    
    csv['E_cor'] = csv['E']*csv['eta_d']
    
    csv_norm = csv.drop(columns=['E','E_cor','lc','dt']).copy()
    csv_norm['c_hat'] = csv['c_hat']/40 
    csv_norm['tau_SD'] = csv['tau_SD']/60
    csv_norm['c_c'] = csv['c_c']/700 
    csv_norm['c_d'] = csv['c_d']/700 
    csv_norm['eta_d'] = csv['eta_d'] 
    
    cmap = plt.cm.get_cmap(colorsectors[scen])
    
    if colvar == 'E_cor':
        normfactor = 2000
    else:
        normfactor = csv[colvar].max()
    # normfactor = 10
    
    range_upper = [eta_d_range_plot[1],c_hat_range_plot[1],c_c_range_plot[1],eta_c_range_plot[1],tau_SD_range_plot[1],c_d_range_plot[1]]
    range_lower = [eta_d_range_plot[0],c_hat_range_plot[0],c_c_range_plot[0],eta_c_range_plot[0],tau_SD_range_plot[0],c_d_range_plot[0]]
    #%%
    charge_e = [0.500,1.000,1.900]
    charge_c = [0.100,1.000,1.400,2.000] 
    discharge_e = [0.500,1.000,1.900] 
    discharge_c = [0.100,1.000,1.400,2.000]
    storage_c = [0.050,0.100,0.250,0.500,1.000,1.500,2.000]
    sloss = [10/60,30/60]
    
    case = csv_norm
    
    from itertools import product
    configs_df = pd.DataFrame(list(product(discharge_e,
                                           storage_c,
                                           charge_c, 
                                           charge_e,
                                           sloss,
                                           discharge_c)), columns=case.columns)
    
    # configs_df['c_hat'] *= 20
    # configs_df['c_c'] *= 350
    # configs_df['c_d'] *= 350
    
    configs_df['c_hat'] *= 20/40
    configs_df['eta_c'] *= 0.5
    configs_df['eta_d'] *= 0.5
    configs_df['c_c'] *= 350/700
    configs_df['c_d'] *= 350/700
    #%%
    # data_plots = pd.DataFrame(index=['eta_d','c_hat','c_c','eta_c','tau_SD','c_d'])
    
    # print(csv['E_cor'][csv['E_cor'] >= threshold_E].shape)
    
    # j = 0
    fig1 = plt.figure(figsize=(10, 13))
    n_ordinate_levels = 2
    title = ''
    radar = ComplexRadar(fig1, variables, var_fs,ranges,title, n_ordinate_levels)
    
    # lc_i = []
    k = 0
    for i in case.index:
        data = case.loc[i]
        data_plot = pd.Series(data.to_list(),
                                    index = ['eta_d','c_hat','c_c','eta_c','tau_SD','c_d']) 
        
        perf_jj = []
        for jj in perfection.keys():
            conf_jj = data.loc[jj]
            if conf_jj == perfection[jj]:
                perf_jj.append(conf_jj)
            
        if len(perf_jj) <= N_perfections:
        
            # eta_1 = data_plot.loc['eta_c']
            
            if (data_plot.loc['eta_c']+eta_c_range_plot[0] > 1):
                data_plot.loc['eta_c'] = 1-0.5*eta_c_range_plot[0]
            
            data_plot.loc['eta_c'] = eta_c_range[1] - data_plot.loc['eta_c']
            data_plot.loc['eta_d'] = eta_d_range[1] - data_plot.loc['eta_d']
            data_plot.loc['tau_SD'] = tau_SD_range[1] - data_plot.loc['tau_SD']
            
            frame = pd.DataFrame(index=data_plot.index)
            frame['range_upper'] = range_upper
            frame['range_lower'] = range_lower
            # lc_i.append(csv.loc[i].lc)
                   
            # if i in csv['E_cor'][csv['E_cor'] >= threshold_E].index:
                # if (ex_i == 'PTES_[1]') and (eta_1 > 1):
                #     cmap = plt.cm.get_cmap('Greens')
                #     plot_sspace(data_plot,csv,frame,colvar,radar,k,normfactor,lines,line_color,cmap,zord=-1)
                # elif eta_1 <= 1:
            cmap = plt.cm.get_cmap(colorsectors[scen])
            plot_sspace(i, threshold_E, data_plot,csv,frame,colvar,radar,k,normfactor,lines,line_color,cmap)
                # data_plots[i] = data_plot
                
            # cmap = plt.cm.get_cmap(colorsectors[sector])
            # plot_sspace(data_plot,csv,frame,colvar,radar,k,normfactor,lines,line_color,cmap)
        
        k += 1
        
    for it, ax in enumerate(fig1.axes):
        ax.set_yticklabels(['']*n_ordinate_levels,fontsize=15,zorder=ax.zorder+10)
        ax.tick_params(axis='x', which='major', pad=50)
        ax.set_axisbelow(True)
    
        color = 'k'
        marker = 'x'
            
        A1 = pd.DataFrame(index=data_plot.index)
        A1['low'] = [0,0,0,0,0,0]
        A1['mid'] = [0.5,0.5,0.5,0.5,0.5,0.5] 
        A1['high'] = [1,1,1,1,1,1]
        
        gridcolor = 'red'
        
        radar.fill(A1['high'],alpha=1,zorder=0,linewidth = 0.3,facecolor='None', edgecolor=gridcolor)
        radar.fill(A1['mid'],alpha=1,zorder=0,linewidth = 0.3,facecolor='None', edgecolor=gridcolor)
        radar.fill(A1['low'],alpha=1,zorder=0,linewidth = 2,facecolor='None', edgecolor=gridcolor)
        
        ticklabcolor = 'red'
        
        ax.text(np.pi/2,0.03,'0',fontsize=fs,color=ticklabcolor,horizontalalignment='center',verticalalignment='center')
        ax.text(np.pi/2,0.52,'20',fontsize=fs,color=ticklabcolor,horizontalalignment='center',verticalalignment='center')
        ax.text(np.pi/2,0.95,'40',fontsize=fs,color=ticklabcolor,horizontalalignment='center',verticalalignment='center')
        
        ax.text(np.pi/6,0.03,'1',fontsize=fs,color=ticklabcolor,horizontalalignment='center',verticalalignment='center')
        ax.text(np.pi/6,0.55,'0.5',fontsize=fs,color=ticklabcolor,horizontalalignment='center',verticalalignment='center')
        ax.text(np.pi/6,0.95,'0',fontsize=fs,color=ticklabcolor,horizontalalignment='center',verticalalignment='center')
        
        ax.text(-np.pi/5,0.04,'0',fontsize=fs,color=ticklabcolor,horizontalalignment='center')
        ax.text(-np.pi/5.5,0.54,'350',fontsize=fs,color=ticklabcolor,horizontalalignment='center')
        ax.text(-np.pi/5.5,0.9,'700',fontsize=fs,color=ticklabcolor,horizontalalignment='center')
        
        ax.text(-np.pi/2,0.01,'>60',fontsize=fs,color=ticklabcolor,horizontalalignment='center',verticalalignment='top')
        ax.text(-np.pi/2,0.5,'30',fontsize=fs,color=ticklabcolor,horizontalalignment='center',verticalalignment='top')
        ax.text(-np.pi/2,0.9,'0',fontsize=fs,color=ticklabcolor,horizontalalignment='center',verticalalignment='top')
        
        ax.text(-5*np.pi/6,0.03,'1',fontsize=fs,color=ticklabcolor,horizontalalignment='center',verticalalignment='center')
        ax.text(-5*np.pi/6,0.55,'0.5',fontsize=fs,color=ticklabcolor,horizontalalignment='center',verticalalignment='center')
        ax.text(-5*np.pi/6,0.95,'0',fontsize=fs,color=ticklabcolor,horizontalalignment='center',verticalalignment='center')
        
        ax.text(0.84*np.pi,0.04,'0',fontsize=fs,color=ticklabcolor,horizontalalignment='center')
        ax.text(0.84*np.pi,0.54,'350',fontsize=fs,color=ticklabcolor,horizontalalignment='center')
        ax.text(0.84*np.pi,0.9,'700',fontsize=fs,color=ticklabcolor,horizontalalignment='center')
        
        data_plot_lim = pd.Series([csv['eta_d'].min(),
                                  csv['c_hat'].max(),
                                  csv['c_c'].max(),
                                  csv['eta_c'].min(),
                                  csv['tau_SD'].min(),
                                  csv['c_d'].max()],
                                        index = ['eta_d','c_hat','c_c','eta_c','tau_SD','c_d'])
        data_plot_lim.loc['eta_c'] = eta_c_range[1] - data_plot_lim.loc['eta_c']
        data_plot_lim.loc['eta_d'] = eta_d_range[1] - data_plot_lim.loc['eta_d']
        data_plot_lim.loc['tau_SD'] = tau_SD_range[1] - data_plot_lim.loc['tau_SD']
    
        if fig_save == 'main':
            cb_ax = fig1.add_axes([1.05,0.25,0.07,0.16])
            if ((tech != 'All') and (flexbar == 'axnotincl') and (fig_save == 'main') and (Ecaplab == 'woE')):
                cb_ax.tick_params(direction='out', length=6, width=2, colors='k',
                                  grid_color='k', grid_alpha=1)
            norm = mpl.colors.Normalize(vmin=0, vmax=normfactor)    
            cb1 = mpl.colorbar.ColorbarBase(cb_ax,orientation='vertical', cmap=cmap,norm=norm) #,ticks=bounds, boundaries=bounds) #ticks=[0.15,0.25,0.48,0.90])
            cb1.ax.tick_params(labelsize=fs)
            cb1.set_label(clabel, rotation=90,fontsize=fs,labelpad=16)
    
        if (fig_save == 'main') and (flexbar == 'axincl'): #and (tech == 'All'):
            ins_ax = fig1.add_axes([0.94,0.591,0,0.306])
            ins_ax.spines['bottom'].set_visible(False)
            ins_ax.spines['top'].set_visible(False)
            ins_ax.spines['right'].set_visible(False)
            ins_ax.set_xticks([])
            ins_ax.set_yticks([0,1])
            ins_ax.set_yticklabels(['Low','High'])
            ins_ax.yaxis.tick_right()
            ins_ax.spines['left'].set_linewidth(1.5)
            ins_ax.tick_params(axis='both', which='major', labelsize=fs)
            ins_ax.tick_params(direction='out', length=6, width=2, colors='k',
                        grid_color='k', grid_alpha=1)
            ins_ax.set_ylabel('Flexibility',labelpad=-30,fontsize=fs)
            
            ins_ax2 = fig1.add_axes([0.5,0.59,0.45,0.308])     
            ins_ax2.spines['bottom'].set_visible(False)
            ins_ax2.spines['top'].set_visible(False)
            ins_ax2.spines['right'].set_visible(False)
            ins_ax2.spines['left'].set_visible(False)
            ins_ax2.axhline(0,color='k',ls='--')
            ins_ax2.axhline(1,color='k',ls='--')
            ins_ax2.set_xticks([])
            ins_ax2.set_yticks([])
            ins_ax2.patch.set_alpha(0)
        
    if plot_ex == True:
        # for ex_i in ex_stores:
        ex_i = ex_stores_dic[candidate]
        data_plot_ex = pd.Series(tech_cost_norm[cols].loc[ex_i].to_list(),
                                        index = ['eta_d','c_hat','c_c','eta_c','tau_SD','c_d']) 
        if (data_plot_ex.loc['eta_c']+eta_c_range_plot[0] > 1):
            data_plot_ex.loc['eta_c'] = 1-0.5*eta_c_range_plot[0]
        data_plot_ex.loc['eta_c'] = eta_c_range[1] - data_plot_ex.loc['eta_c']
        data_plot_ex.loc['eta_d'] = eta_d_range[1] - data_plot_ex.loc['eta_d']
        data_plot_ex.loc['tau_SD'] = tau_SD_range[1] - data_plot_ex.loc['tau_SD']
        plot_existing(data_plot_ex,frame,radar,color,marker,lw=2)
                    
    fig1.savefig('figures/Spiderweb_result_' + scen+ '_w_' + '_' + colvar + '_Nperf' + str(N_perfections) + '.png',
                    bbox_inches="tight",dpi=300)
        # j += 1