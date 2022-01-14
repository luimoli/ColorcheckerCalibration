import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plotHeatmap(arr_raw, save_path):
    """[plot the heatmap of a 2D array and save to png-img]

    Args:
        arr ([np array]): [2-dimention]
        save_path ([str]): [r'xxxxx.png']
    """
    arr = np.round(arr_raw, 4)
    f, ax = plt.subplots(figsize=(30, 15))
    # vmin=int(np.min(a_difference)),vmax=int(np.max(a_difference))  
    res = sns.heatmap(arr, cmap='RdYlGn_r',annot=False, ax=ax,cbar=True, xticklabels=False,yticklabels=False,square=True)
    # cbar = res.collections[0].colorbar
    # cbar.ax.tick_params(labelsize=20)
    fig = res.get_figure()
    fig.savefig(save_path, bbox_inches="tight")



if __name__ == '__main__':
    tmp = np.random.randint(0, 255, (540, 960))
    plotHeatmap(tmp, './test_heat.png')