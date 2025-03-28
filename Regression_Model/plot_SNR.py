import numpy as np
from matplotlib import pyplot as plt
# SNR_results = np.load(".\\outputs\\SNR_results_very_long_run.npy")
SNR_results = np.load(".\\outputs\\SNR_results.npy")
# num_snr_points = 5
num_snr_points = 3
from UliEngineering.Math.Decibel import *
import matplotlib.ticker as mtick
import matplotlib
import os
import math
def save_fig(fname,path):
    """
    The function saves the current figure to the specified file by using plt.savefig with the file path obtained by joining FIGURES_PATH and fname
    :param fname:  File name or path to save the figure.
    """
    plt.savefig(os.path.join(path, fname), bbox_inches='tight', format='pdf', transparent=True)

font = {'size'   : 22}
matplotlib.rc('font', **font)
plt.rcParams["font.family"] = "Times New Roman"
def decibel_formatter(v0=1.0, unit='dB'):
    def format_value(value, pos=None):
        dB = value_to_dB(value, v0=v0)
        return f'{dB:.1f}'
    return format_value
def plotting(axes,SNR):
    p0, = axes.plot(snr_values_db, np.mean(SNR[:, 0, :], axis=1) / 120, '--',
                   color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0], linewidth=1.65)
    p1, = axes.plot(snr_values_db, np.mean(SNR[:, 1, :], axis=1) / 120, '-.',
                   color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1], linewidth=1.65)
    p2, = axes.plot(snr_values_db, np.mean(SNR[:, 2, :], axis=1) / 120, '-.v',
                   color=plt.rcParams['axes.prop_cycle'].by_key()['color'][2], linewidth=1.65)
    p3, = axes.plot(snr_values_db, np.mean(SNR[:, 3, :], axis=1) / 120, '-',
                   color=plt.rcParams['axes.prop_cycle'].by_key()['color'][3], linewidth=1.65)
    p4, = axes.plot(snr_values_db, np.mean(SNR[:, 4, :], axis=1) / 120, '--v',
                   color=plt.rcParams['axes.prop_cycle'].by_key()['color'][4], linewidth=1.65)
    return p0,p1,p2,p3,p4


snr_values = np.linspace(0.5,60,num_snr_points)
snr_values_db = 10*np.log10(snr_values)
noise_array = 1/snr_values
fig1, ax1 = plt.subplots()
# fig1.set_figwidth(8.5)
# fig1.set_figheight(6.5)
# fig1.set_figwidth(9.5)
# fig1.set_figheight(7.25)
fig1.set_figwidth(8)
fig1.set_figheight(5.7)
p0,p1,p2,p3,p4=plotting(ax1,SNR_results)
# p0, = ax1.plot(1/noise_array,np.mean(SNR_results[:,0,:],axis=1)/120,'--',color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0],linewidth=1.65)
# p1, = ax1.plot(1/noise_array,np.mean(SNR_results[:,1,:],axis=1)/120,'-.',color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1],linewidth=1.65)
# p2, = ax1.plot(1/noise_array,np.mean(SNR_results[:,2,:],axis=1)/120,'-.v',color=plt.rcParams['axes.prop_cycle'].by_key()['color'][2],linewidth=1.65)
# p3, = ax1.plot(1/noise_array,np.mean(SNR_results[:,3,:],axis=1)/120,'-',color=plt.rcParams['axes.prop_cycle'].by_key()['color'][3],linewidth=1.65)
# p4, = ax1.plot(1/noise_array,np.mean(SNR_results[:,4,:],axis=1)/120,'--v',color=plt.rcParams['axes.prop_cycle'].by_key()['color'][4],linewidth=1.65)
# p5, = ax1.plot(1/noise_array,np.mean(SNR_results[:,5,:],axis=1)/120,'-.v',color=plt.rcParams['axes.prop_cycle'].by_key()['color'][4],linewidth=1.65)
# p6, = ax1.plot(1/noise_array,np.mean(SNR_results[:,6,:],axis=1)/120,':v',color=plt.rcParams['axes.prop_cycle'].by_key()['color'][3],linewidth=1.65)

legend_properties = {'weight':'bold'}
# ax1.legend([p1,p2,p3,p4,p5,p6],["SimTwin","ZO","DNNTwin","SimTwin limited","ZO limited","Random limited"],prop=legend_properties,labelspacing=0.04,borderpad=0.2)
ax1.legend([p0,p1,p2,p3,p4],["Simulation GD","Simulation GD + 0.01 Noise","ZOGD","Random","ZO aided ALD"],prop=legend_properties,labelspacing=0.04,borderpad=0.2)
ax1.grid('True')
# plt.xticks([20*math.log(1,10),
#             20*math.log(5,10),
#             20*math.log(15,10),
#             20*math.log(20,10),
#             20*math.log(25,10),
#             20*math.log(30,10),
#             20*math.log(35,10),
#             20*math.log(40,10),
#             20*math.log(45,10),
#             20*math.log(50,10),
#             20*math.log(55,10),
#             20*math.log(60,10)])

# plt.xlim(0,62)
# plt.xscale('log')
# plt.xticks([0.5,5,10,15,20,25,30,35,40,45,50,55,60])
# plt.xticks('log')
ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
# ax1.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
ax1.grid(which = "major",linewidth=1)
ax1.grid(which = "minor",linewidth=0.2)
# ax1.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.5,10,20,30,40,50,60]))
# plt.gca().xaxis.set_major_formatter(mtick.FuncFormatter(decibel_formatter()))
# ax1.set_xscale("log")
# ax1.set_yscale("log")
ax1.minorticks_on()

plt.xlabel("SNR[dB]")
plt.ylabel("Rate [Bits/Channel use]")
plt.tight_layout()
x1, x2, y1, y2 = 0, 3, 0.5, 1.5  # subregion of the original image
axins = ax1.inset_axes(
    [0.35, 0.03, 0.3, 0.3],
    xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
plotting(axins,SNR_results)
ax1.indicate_inset_zoom(axins, edgecolor="black")

x1, x2, y1, y2 = 12, 16, 2.6, 4  # subregion of the original image
axins2 = ax1.inset_axes(
    [0.67, 0.1, 0.3, 0.3],
    xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
plotting(axins2,SNR_results)
ax1.indicate_inset_zoom(axins2, edgecolor="black")

save_fig("SNR_graph_SPAWC2025.pdf",".\\outputs\\")
plt.show()