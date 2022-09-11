import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE



class EmbeddingVisualisation:
    def __init__(self, dataset, plot_dir, class_size):
        self.dataset = dataset
        self.class_size = class_size

        self.plot_dir = plot_dir
        if not os.path.exists(self.plot_dir):
            os.mkdir(self.plot_dir)


    def plot_tSNE(self,data,lbl, exp_id,label_name):

        target = np.array(np.argmax(lbl,axis=-1))
        tsne = TSNE(2, verbose=1)
        tsne_proj = tsne.fit_transform(data)
        # Plot those points as a scatter plot and label them based on the pred labels
        cmap = cm.get_cmap('tab20')
        fig, ax = plt.subplots(figsize=(8, 8))
        num_categories = self.class_size
        for lab in range(num_categories):
            indices = target == lab
            ax.scatter(tsne_proj[indices, 0], tsne_proj[indices, 1], c=np.array(cmap(lab)).reshape(1, 4), label=lab,
                       alpha=0.5)
        ax.axis('off')
        ax.legend(label_name,fontsize='large', markerscale=2)
        plt.savefig(os.path.join(self.plot_dir, exp_id + "_TSNE"+".png"))
