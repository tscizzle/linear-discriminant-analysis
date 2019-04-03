import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import ArrowStyle, FancyArrowPatch

import code
from pprint import pprint


class RandomData(object):
    DATA_FILENAME = 'iris.data'
    LABEL_DICT = {1: 'Setosa', 2: 'Versicolor', 3: 'Virginica'}
    FEATURE_LABELS = [
        'sepal length in cm',
        'sepal width in cm',
        'petal length in cm',
        'petal width in cm',
    ]
    FEATURE_DICT = {i: label for i, label in enumerate(FEATURE_LABELS)}

    def __init__(self):

        df = pd.io.parsers.read_csv(
            filepath_or_buffer=self.DATA_FILENAME,
            header=None,
            sep=',',
        )

        df.columns = [
            l for i, l in sorted(self.FEATURE_DICT.items())
        ] + ['class label']

        df.dropna(how="all", inplace=True) # to drop the empty line at file-end

        df.tail()

        self.X = df.values[:,0:4]

        self.y = df['class label']

        enc = LabelEncoder()
        label_encoder = enc.fit(self.y)
        self.y = label_encoder.transform(self.y) + 1

        mean_vectors = [
            np.mean(self.X[self.y==c], axis=0)
            for c in range(1, 4)
        ]

        within_class_scatter = np.zeros((4, 4))
        for c in range(1, 4):
            class_scatter = np.zeros((4, 4))
            mean_vec = mean_vectors[c-1]
            mean_col = mean_vec.reshape(4, 1)
            for datum in self.X[self.y==c]:
                datum_col = datum.reshape(4, 1)
                offset = datum_col - mean_col
                datum_contribution = offset.dot(offset.T)
                class_scatter = class_scatter + datum_contribution
            within_class_scatter = within_class_scatter + class_scatter
        within_class_scatter = within_class_scatter.astype(np.cdouble)

        overall_mean = np.mean(self.X, axis=0)
        between_class_scatter = np.zeros((4, 4))
        overall_mean_col = overall_mean.reshape(4, 1)
        for c in range(1, 4):
            class_n = self.X[self.y == c,:].shape[0]
            mean_vec = mean_vectors[c-1]
            mean_col = mean_vec.reshape(4, 1)
            offset = mean_col - overall_mean_col
            class_contribution = class_n * offset.dot(offset.T)
            between_class_scatter = between_class_scatter + class_contribution
        between_class_scatter = between_class_scatter.astype(np.cdouble)

        eig_vals, eig_vecs = np.linalg.eig(
            np.linalg.inv(within_class_scatter).dot(between_class_scatter)
        )

        eig_pairs = [
            {
                'eig_val': eig_vals[i],
                'eig_vec': eig_vecs[:,i]
            }
            for i in range(len(eig_vals))
        ]
        eig_pairs = sorted(
            eig_pairs,
            key=lambda x: abs(x['eig_val']),
            reverse=True
        )
        new_axes = np.hstack([
            p['eig_vec'].reshape(4, 1) for p in eig_pairs[:2]
        ])

        self.X_reduced = self.X.dot(new_axes)
        self.X_reduced = self.X_reduced.astype(np.cdouble)

        sklearn_pca = sklearnPCA(n_components=2)
        self.X_reduced_pca = sklearn_pca.fit_transform(self.X)

        sklearn_lda = LDA(n_components=2)
        self.X_reduced_lda = sklearn_lda.fit_transform(self.X, self.y)

    def showFeatureDists(self):
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,6))

        for ax, cnt in zip(axes.ravel(), range(4)):

            # set bin sizes
            min_b = math.floor(np.min(self.X[:,cnt]))
            max_b = math.ceil(np.max(self.X[:,cnt]))
            bins = np.linspace(min_b, max_b, 25)

            # plotting the histograms
            for label, color in zip(range(1,4), ('blue', 'red', 'green')):
                ax.hist(
                    [self.X[self.y==label, cnt]],
                    color=color,
                    label='class %s' % self.LABEL_DICT[label],
                    bins=bins,
                    alpha=0.5,
                )
            ylims = ax.get_ylim()

            # plot annotation
            leg = ax.legend(loc='upper right', fancybox=True, fontsize=8)
            leg.get_frame().set_alpha(0.5)
            ax.set_ylim([0, max(ylims) + 2])
            ax.set_xlabel(self.FEATURE_DICT[cnt])
            ax.set_title('Iris histogram #%s' % str(cnt + 1))

            # hide axis ticks
            ax.tick_params(
                axis="both",
                which="both",
                bottom="off",
                top="off",
                labelbottom="on",
                left="off",
                right="off",
                labelleft="on"
            )

            # remove axis spines
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)

        axes[0][0].set_ylabel('count')
        axes[1][0].set_ylabel('count')

        fig.tight_layout()

        plt.show()

    def showTransformedData(self, method='custom_lda'):
        reduced_data = {
            'custom_lda': self.X_reduced,
            'pca': self.X_reduced_pca,
            'lda': self.X_reduced_lda,
        }[method]

        ax = plt.subplot(111)
        for label, marker, color in zip(
            range(1, 4) ,('^', 's', 'o'), ('blue', 'red', 'green')
        ):
            plt.scatter(
                x=reduced_data[:,0].real[self.y == label],
                y=reduced_data[:,1].real[self.y == label],
                marker=marker,
                color=color,
                alpha=0.5,
                label=self.LABEL_DICT[label]
            )

        plt.xlabel('LD1')
        plt.ylabel('LD2')

        leg = plt.legend(loc='upper right', fancybox=True)
        leg.get_frame().set_alpha(0.5)
        plt.title('LDA: Iris projection onto the first 2 linear discriminants')

        # hide axis ticks
        plt.tick_params(
            axis="both",
            which="both",
            bottom="off",
            top="off",
            labelbottom="on",
            left="off",
            right="off",
            labelleft="on"
        )

        # remove axis spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        plt.grid()
        plt.tight_layout
        plt.show()


def main():
    np.set_printoptions(precision=4)

    rd = RandomData()
    rd.showTransformedData()


if __name__ == '__main__':
    main()
