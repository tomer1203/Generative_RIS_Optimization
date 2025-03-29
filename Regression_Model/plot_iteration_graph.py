import torch
import numpy as np
from matplotlib import pyplot as plt
import os
def save_fig(fname,path):
    """
    The function saves the current figure to the specified file by using plt.savefig with the file path obtained by joining FIGURES_PATH and fname
    :param fname:  File name or path to save the figure.
    """
    plt.savefig(os.path.join(path, fname), bbox_inches='tight', format='pdf', transparent=True)
iterations_results = np.load("./outputs/iteration_graph_results.npy")
number_of_repetitions = 4
fig1, ax1 = plt.subplots()
p0, = ax1.plot(iterations_results[1] / (120*(number_of_repetitions+1)),'--v',linewidth=1.5,fillstyle='right',markersize=5)#,linestyle='dashed')
p1, = ax1.plot(iterations_results[2] / (120*(number_of_repetitions+1)),'-.^',linewidth=1.5,fillstyle='right',markersize=5)#,linestyle='dashed')
p2, = ax1.plot(iterations_results[3] / (120*(number_of_repetitions+1)),':o',linewidth=1.5,fillstyle='right',markersize=5)#,linestyle='dashed')
p3, = ax1.plot(iterations_results[0] / (120*(number_of_repetitions+1)),'-s',linewidth=2,fillstyle='right',markersize=5)#,linestyle='dashed')
ax1.legend([p0,p1,p2,p3],["Simulation","Simulation+Environment Noise","ZOGD","mapALD"])
plt.xlabel("iteration")
plt.ylabel("Rate [Bits/Channel use]")
plt.grid(visible=True)
save_fig("SPAWC2025_iteration_rate_graph.pdf","plots")
plt.show()
