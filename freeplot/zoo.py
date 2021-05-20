

from typing import Tuple, Union, Optional, Dict
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from .base import *
from .config import zoo_cfg



def tsne(
    features: np.ndarray, 
    labels: np.ndarray, 
    fp: FreePlot, 
    index: Union[int, str] = 0, 
    **kwargs: "other kwargs of fp.scatterplot"
) -> None:
    """
    features: n x d
    labels: n
    """
    from sklearn.manifold import TSNE
    data_embedded = TSNE(n_components=2, learning_rate=10, n_iter=1000).fit_transform(features)
    fp[index].set_xticks([])
    fp[index].set_yticks([])
    data = pd.DataFrame(
        {
            "x": data_embedded[:, 0],
            "y": data_embedded[:, 1],
            "label": labels
        }
    )
    for label in np.unique(labels):
        event = data.loc[data['label']==label]
        x = event['x']
        y = event['y']
        x_mean = x.median()
        y_mean = y.median()
        plt.text(x_mean, y_mean, label)
        fp.scatterplot(x, y, index, label=label, s=1.5, edgecolors="none", **kwargs)
    sns.despine(left=True, bottom=True)


def roc_curve(
    y_pred: np.ndarray, 
    y_labels: np.ndarray, 
    fp: FreePlot, 
    index: Union[int, str] = 0, 
    name: Optional[str] = None,
    estimator_name: Optional[str] = None,
    style: str = "whitegrid",
    dict_: Optional[Dict] = None,
) -> "tpr, fpr, roc_auc":
    """
    y_pred: the prediction
    y_labels: the corresponding labels of instances
    fp: ...
    index: ...
    name: for labelling the roc_curve, is None, use the estimator_name
    estimator_name: the name of classifier
    style: the style of seaborn
    dict_: the correspoding properties dict
    """
    from sklearn import metrics
    fpr, tpr, thresholds = metrics.roc_curve(y_labels, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, 
                            roc_auc=roc_auc, estimator_name=estimator_name)
    with sns.axes_style(style, dict_):
        display.plot(fp[index], name)
    return tpr, fpr, roc_auc


@style_env(zoo_cfg.radar_style)
def pre_radar(num_vars: int, frame: str = "circle") -> np.ndarray:
    from matplotlib.patches import Circle, RegularPolygon
    from matplotlib.path import Path
    from matplotlib.projections.polar import PolarAxes
    from matplotlib.projections import register_projection
    from matplotlib.spines import Spine
    from matplotlib.transforms import Affine2D

    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

@style_env(zoo_cfg.radar_style)
def pos_radar(
    data: Dict, 
    labels: np.ndarray, 
    fp: FreePlot, 
    theta: Optional[np.ndarray] = None, 
    index: Union[int, str] = 0
) -> None:
    fp.fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)
    theta = np.linspace(0, 2*np.pi, len(labels), endpoint=False) if theta is None else theta
    ax = fp[index]
    for i, (key, value) in enumerate(data.items()): 
        ax.plot(theta, value)
        ax.fill(theta, value, alpha=0.5, label=labels[i])
    ax.set_varlabels(labels)