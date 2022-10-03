from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class DrawPCA:
    def __init__(self, df, target, n_components):
        self._numpyArray = None
        self.pca = None
        self.n_components = n_components
        self.df = df
        self.target = target
        self.fit_transform()

    def fit_transform(self):
        self.pca = PCA(n_components=self.n_components)
        X = self.df.loc[:, self.df.columns != self.target]
        y = self.df[self.target]
        self.pca.fit(X)
        X_reduction = self.pca.transform(X)
        self._numpyArray = np.hstack((np.array(y)[:, None], X_reduction))

    def draw(self):
        if self.n_components == 2:
            panda_df = pd.DataFrame(data=self._numpyArray,
                                    columns=[self.target,
                                             "x", "y"])

            fig, axes = plt.subplots(1, 1, figsize=(25, 10))
            sns.scatterplot(data=panda_df, x="x", y="y", ax=axes, hue=self.target)

        if self.n_components == 3:
            fig = plt.figure(figsize=(30, 20))
            ax = plt.axes(projection="3d")

            my_cmap = plt.get_cmap('bwr')
            panda_df = pd.DataFrame(data=self._numpyArray,
                                    columns=[self.target,
                                             "x", "y", "z"])
            sctt = ax.scatter3D(panda_df["x"], panda_df["y"], panda_df["z"], alpha=0.8, c=panda_df[self.target],
                                cmap=my_cmap,
                                marker='^')
            fig.colorbar(sctt, ax=ax, shrink=0.5, aspect=5)