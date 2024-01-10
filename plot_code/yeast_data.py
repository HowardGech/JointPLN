#!/usr/bin/env python3
#%%
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 17:19:44 2023

@author: gechanghao
"""

# system path is '/Users/gechanghao/py/VEM/result_1127'
import itertools
import networkx as nx
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
conditions=["Glutamine","Proline","AmmoniumSulfate","Urea","YPD","YPDRapa","CStarve","MinimalGlucose","MinimalEtOH","YPDDiauxic","YPEtOH"]
scale0 = [0.1,0.25,0.5,0.75,1.0,2.0,5.0]
scale1 = [4.0,6.25,9.0,12.25,16.0]
scales = list(itertools.product(scale0,scale1))
k = len(scales)
p=612
I = 11
y = [None]*I
n_i = [None]*I
mu = [None]*I
Offset = [None]*I
beta_joint = [None]*I
beta_pln = [None]*I
EBICs = np.array([[None]*k for _ in range(I)])
os.chdir('/Users/gechanghao/py/VEM/result_1127')
plt.switch_backend('module://ipykernel.pylab.backend_inline')

# read EBICs
for m in range(I):
    for i in range(k):
        j,l = scales[i]
        file_name = f"EBIC_{j}_{l}_{m+1}.csv"
        EBICs[m,i]=float(np.loadtxt(file_name))


def cor_from_cov(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation


index_EBIC = np.argmin(EBICs,axis=1)
    
best_scales = [None]*I
best_EBICs = [None]*I
best_EBICs_pln = [None]*I
for m in range(I):
    best_scales[m] = scales[index_EBIC[m]]
    best_EBICs[m] = EBICs[m,index_EBIC[m]]
    EBIC_pln = np.loadtxt(f'../result_pln/EBIC_real_{m+1}.csv',delimiter=',',dtype=str)
    EBIC_pln = EBIC_pln[1:6,1].astype(float)
    best_EBICs_pln[m] = np.min(EBIC_pln)
    
name = np.loadtxt('../real_2.csv',delimiter=',',dtype=str)
name = name[0,1:]
name = np.array([name[i].replace('"', '') for i in range(len(name))])


nodes_num = 50
Omegas_joint = [None]*I
Omegas_pln = [None]*I
ys = [None]*I
Nums_pln = [None]*I
Nums_joint = [None]*I
Gs_joint = [None]*I
Gs_pln = [None]*I
Poses = [None]*I
NS_joint = [None]*I
NS_pln = [None]*I
Gs_joint_num = [None]*I
Gs_pln_num = [None]*I
lab_Poses = [None]*I
lab = [None]*I



# transcription amplitude for each gene and conditions
for i in range(I):
    test = np.loadtxt(f'../real_{i+1}.csv',delimiter=",",dtype=str)
    test = test[1:test.shape[0],1:test.shape[1]]
    test = test.astype(float)
    y[i] = np.copy(test[:,0:p])
    n_i[i] = y[i].shape[0]
    Offset[i] = np.zeros((n_i[i],p))
    for j in range(n_i[i]):
        Offset[i][j,:] = test[j,p]
    mu[i] = np.loadtxt(f'mu_{i+1}.csv',delimiter=",")
    beta_joint[i] = np.mean(mu[i]-Offset[i],0)


# create networks for all conditions
for i in range(I):
    s0,s1 = best_scales[i]
    Omega_joint = np.loadtxt(f'Omega_{s0}_{s1}_{i+1}.csv',delimiter=',')
    Omega_pln = np.loadtxt(f'../result_pln/Omega_real_{i+1}.csv',delimiter=',',dtype=str)
    Omega_pln = Omega_pln[1:613,:]
    Omega_pln = Omega_pln[:,1:613]
    Omega_pln = Omega_pln.astype(float)
    Nums_pln[i] = np.sum(Omega_pln!=0)-612
    Nums_joint[i] = np.sum(Omega_joint!=0)-612
    y_real = np.loadtxt(f'../real_{i+1}.csv',delimiter=',',dtype=str)
    y_real = y_real[:,1:613]
    y_real = y_real[1:(y_real.shape[0]),:]
    y_real = y_real.astype(int)
    ys[i] = y_real
    y_exp = np.sum(y_real,axis=0)
    y_arg = np.argsort(-y_exp)[0:nodes_num]
    lab[i] = name[y_arg]
    Omega_pln_trun = Omega_pln[y_arg,:]
    Omega_pln_trun = Omega_pln_trun[:,y_arg]
    Omega_joint_trun = Omega_joint[y_arg,:]
    Omega_joint_trun = Omega_joint_trun[:,y_arg]
    NS_joint[i] = np.diag(Omega_joint_trun)
    NS_pln[i] = np.diag(Omega_pln_trun)
    Omega_pln_trun = cor_from_cov(Omega_pln_trun)
    np.fill_diagonal(Omega_pln_trun,0)
    Omega_joint_trun = cor_from_cov(Omega_joint_trun)
    np.fill_diagonal(Omega_joint_trun,0)
    Omegas_joint[i] = Omega_joint_trun
    Omegas_pln[i] = Omega_pln_trun
    Gs_pln[i] = nx.Graph(Omega_pln_trun!=0)
    Gs_joint[i] = nx.Graph(Omega_joint_trun!=0)
    #####################################################################################################################################################
    ############## uncomment below if need to remove 0 degree nodes #####################################################################################
    #####################################################################################################################################################
    #isolated_nodes = [node for node in Gs_pln[i].nodes() if Gs_pln[i].degree(node) == 0 and Gs_joint[i].degree(node) == 0]
    #connected_nodes = np.array(list(set(range(nodes_num))-set(isolated_nodes)))
    #NS_joint[i] = NS_joint[i][connected_nodes]
    #NS_pln[i] = NS_pln[i][connected_nodes]
    #Omegas_joint[i] = Omegas_joint[i][connected_nodes,:]
    #Omegas_joint[i] = Omegas_joint[i][:,connected_nodes]
    #Omegas_pln[i] = Omegas_pln[i][connected_nodes,:]
    #Omegas_pln[i] = Omegas_pln[i][:,connected_nodes]
    #Gs_pln[i].remove_nodes_from(isolated_nodes)
    #Gs_joint[i].remove_nodes_from(isolated_nodes)
    #####################################################################################################################################################
    #####################################################################################################################################################
    Gs_joint_num[i] = Gs_joint[i]
    Gs_pln_num[i] = nx.Graph(Omegas_pln[i]!=0)
    Gs_joint_num[i] = nx.Graph(Omegas_joint[i]!=0)
    Gs_pln[i]=nx.relabel_nodes(Gs_pln[i],dict(zip(range(nodes_num),lab[i])))
    Gs_joint[i]=nx.relabel_nodes(Gs_joint[i],dict(zip(range(nodes_num),lab[i])))
    Poses[i] = nx.spring_layout(Gs_joint[i],k=0.5,iterations=50)
    




# plot precision matrices in the appendix
mat_temp = [None]*I
for k in range(I):
    fig=plt.figure(figsize=(5,5))
    sorted_lab = np.argsort(lab[k])
    mat_temp[k] = np.triu(Omegas_joint[k][sorted_lab,:][:,sorted_lab])
    mat_temp[k] += np.triu(Omegas_pln[k][sorted_lab,:][:,sorted_lab]).T
    np.fill_diagonal(mat_temp[k],1)
    ax = sns.heatmap(mat_temp[k],square=True,cbar=False,cmap='RdBu',linewidth=0.5,vmin=-0.2,vmax=0.2,xticklabels=lab[k][sorted_lab], yticklabels=lab[k][sorted_lab])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right',fontsize=4.5)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=4.5)
    ax.set_title(f"Condition: {conditions[k]}")
    plt.tight_layout()
    plt.show()
    plt.savefig(f'../plot/matrix_{k}.pdf')
    plt.close('all')
    

# plot precision matrices in the main text
## Create subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

## Create heatmaps for each subplot
heatmap1 = sns.heatmap(mat_temp[5],cbar=False,cmap='RdBu',linewidth=0.5,vmin=-0.2,vmax=0.2,xticklabels=lab[k][sorted_lab], yticklabels=lab[k][sorted_lab], ax=axs[0])
heatmap2 = sns.heatmap(mat_temp[8],cbar=False,cmap='RdBu',linewidth=0.5,vmin=-0.2,vmax=0.2,xticklabels=lab[k][sorted_lab], yticklabels=lab[k][sorted_lab], ax=axs[1])
heatmap1.set_xticklabels(heatmap1.get_xticklabels(), rotation=45, ha='right',fontsize=4.5)
heatmap1.set_yticklabels(heatmap1.get_yticklabels(), fontsize=4.5)
heatmap2.set_xticklabels(heatmap2.get_xticklabels(), rotation=45, ha='right',fontsize=4.5)
heatmap2.set_yticklabels(heatmap2.get_yticklabels(), fontsize=4.5)
## Create a common colorbar at the bottom
cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.018]) ## Adjust the position and size as needed
cbar = fig.colorbar(heatmap2.get_children()[0], cax=cbar_ax,orientation='horizontal')
## Adjust layout
plt.tight_layout(pad=2.5)
heatmap1.set_title(f'Condition: {conditions[5]}')
heatmap2.set_title(f'Condition: {conditions[8]}')
plt.show()
plt.savefig('../plot/matrix_main.pdf')
    

# subplot of joint estimation and separate estimation in the main text
for k in range(I):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].set_frame_on(False)
    axs[1].set_frame_on(False)
    label_pos = {key: [value[0], value[1] + 0.045] for key, value in Poses[k].items()}
    nx.draw_networkx_nodes(Gs_joint[k],Poses[k],node_size=5*NS_joint[k]+10,alpha=0.8, edgecolors='darkorchid',node_color='darkorchid',ax=axs[0],linewidths=0.3)
    nx.draw_networkx_nodes(Gs_joint[k],Poses[k],node_size=5*NS_joint[k]+10,alpha=1, edgecolors='darkorchid',node_color='none',ax=axs[0],linewidths=0.7)
    nx.draw_networkx_edges(Gs_joint[k],Poses[k],width=[4*abs(Omegas_joint[k])[i, j]+0.2 for i, j in Gs_joint_num[k].edges()],alpha=0.8, edge_color='grey',ax=axs[0])
    #nx.draw(Gs_joint[k],Poses[k],node_size=5*NS_joint[k]+10,width=[5*abs(Omegas_joint[k])[i, j] for i, j in Gs_joint_num[k].edges()],alpha=0.8,with_labels=False, edgecolors='darkorchid',font_size=4,node_color='darkorchid',font_color='black',edge_color='grey',linewidths=0.3,ax=axs[0])
    nx.draw_networkx_labels(Gs_joint[k],label_pos,font_size=4, verticalalignment='top',ax=axs[0])

    nx.draw_networkx_nodes(Gs_pln[k],Poses[k],node_size=5*NS_pln[k]+10,alpha=0.8, edgecolors='darkorchid',node_color='darkorchid',ax=axs[1],linewidths=0.3)
    nx.draw_networkx_nodes(Gs_pln[k],Poses[k],node_size=5*NS_pln[k]+10,alpha=1, edgecolors='darkorchid',node_color='none',ax=axs[1],linewidths=0.7)
    nx.draw_networkx_edges(Gs_pln[k],Poses[k],width=[4*abs(Omegas_joint[k])[i, j]+0.2 for i, j in Gs_joint_num[k].edges()],alpha=0.8, edge_color='grey',ax=axs[1])
    #nx.draw(Gs_joint[k],Poses[k],node_size=5*NS_joint[k]+10,width=[5*abs(Omegas_joint[k])[i, j] for i, j in Gs_joint_num[k].edges()],alpha=0.8,with_labels=False, edgecolors='darkorchid',font_size=4,node_color='darkorchid',font_color='black',edge_color='grey',linewidths=0.3,ax=axs[0])
    nx.draw_networkx_labels(Gs_pln[k],label_pos,font_size=4, verticalalignment='top',ax=axs[1])
    node_sizes_legend = [5, 12]
    node_sizes_labels = ["Low","High"]
    node_sizes_dict = dict(zip(node_sizes_labels, node_sizes_legend))
    axs[0].set_title('Joint estimation')
    axs[1].set_title('PLNnetwork')

    edge_widths_legend = [0.5,1.8]
    edge_widths_labels = ["Weak","Strong"]
    edge_widths_dict = dict(zip(edge_widths_labels, edge_widths_legend))

    legend1=axs[0].legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', label=label, alpha=0.8, markersize=size, markerfacecolor='darkorchid', markeredgecolor='darkorchid') 
        for label, size in node_sizes_dict.items()],
        title="Transcription variance", loc="lower left", ncol=3, bbox_to_anchor=(0.6, -0.1),frameon=False
    )
   

    legend2=axs[1].legend(handles=[plt.Line2D([0], [0], color='grey', alpha=0.8,linewidth=width, label=label) 
    for label, width in edge_widths_dict.items()],
    title="Gene expression correlation", loc="lower right", ncol=3,bbox_to_anchor=(0.4, -0.1),frameon=False
    )
    plt.tight_layout()
    plt.savefig('../plot/AmmoniumSulfate_graph.pdf')
    plt.close('all')
    
    
# appendix subplots for graphs
for k in range(I):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].set_frame_on(False)
    axs[1].set_frame_on(False)
    label_pos = {key: [value[0], value[1] + 0.045] for key, value in Poses[k].items()}
    nx.draw_networkx_nodes(Gs_joint[k],Poses[k],node_size=5*NS_joint[k]+10,alpha=0.8, edgecolors='darkorchid',node_color='darkorchid',ax=axs[0],linewidths=0.3)
    nx.draw_networkx_nodes(Gs_joint[k],Poses[k],node_size=5*NS_joint[k]+10,alpha=1, edgecolors='darkorchid',node_color='none',ax=axs[0],linewidths=0.7)
    nx.draw_networkx_edges(Gs_joint[k],Poses[k],width=[4*abs(Omegas_joint[k])[i, j]+0.2 for i, j in Gs_joint_num[k].edges()],alpha=0.8, edge_color='grey',ax=axs[0])
    #nx.draw(Gs_joint[k],Poses[k],node_size=5*NS_joint[k]+10,width=[5*abs(Omegas_joint[k])[i, j] for i, j in Gs_joint_num[k].edges()],alpha=0.8,with_labels=False, edgecolors='darkorchid',font_size=4,node_color='darkorchid',font_color='black',edge_color='grey',linewidths=0.3,ax=axs[0])
    nx.draw_networkx_labels(Gs_joint[k],label_pos,font_size=4, verticalalignment='top',ax=axs[0])

    nx.draw_networkx_nodes(Gs_pln[k],Poses[k],node_size=5*NS_joint[k]+10,alpha=0.8, edgecolors='darkorchid',node_color='darkorchid',ax=axs[1],linewidths=0.3)
    nx.draw_networkx_nodes(Gs_pln[k],Poses[k],node_size=5*NS_joint[k]+10,alpha=1, edgecolors='darkorchid',node_color='none',ax=axs[1],linewidths=0.7)
    nx.draw_networkx_edges(Gs_pln[k],Poses[k],width=[4*abs(Omegas_joint[k])[i, j]+0.2 for i, j in Gs_joint_num[k].edges()],alpha=0.8, edge_color='grey',ax=axs[1])
    #nx.draw(Gs_joint[k],Poses[k],node_size=5*NS_joint[k]+10,width=[5*abs(Omegas_joint[k])[i, j] for i, j in Gs_joint_num[k].edges()],alpha=0.8,with_labels=False, edgecolors='darkorchid',font_size=4,node_color='darkorchid',font_color='black',edge_color='grey',linewidths=0.3,ax=axs[0])
    nx.draw_networkx_labels(Gs_pln[k],label_pos,font_size=4, verticalalignment='top',ax=axs[1])

    node_sizes_legend = [3, 10]
    node_sizes_labels = ["Low","High"]
    node_sizes_dict = dict(zip(node_sizes_labels, node_sizes_legend))
    plt.suptitle(f"Condition: {conditions[k]}",size=20)
    plt.tight_layout()
    plt.savefig(f'../plot/{k}_graph.pdf')
    plt.close('all')
    
    
    