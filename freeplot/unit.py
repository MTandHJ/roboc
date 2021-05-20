



from typing import Tuple, Optional, Dict, Union
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as AA
import os
from collections.abc import Iterable

from .config import cfg
from .utils import axis, reset



_ROOT = cfg['root']




class UnitAX:

    def __init__(
        self, axes, gs, anchor: 'UnitAX',
        sharey: bool = True, **kwargs
    ):
        self.axes = axes
        self.gs = gs
        self.anchor = anchor
        self.sharey = sharey
        self.kwargs = kwargs
        self.__ax = None

    @property
    def ax(self):
        if self.__ax is None:
            if not self.sharey or self.anchor is None:
                self.__ax = self.axes.fig.add_subplot(
                    self.gs, **self.kwargs
                )
            else:
                self.__ax = self.axes.fig.add_subplot(
                    self.gs, sharey=self.anchor.ax, **self.kwargs
                )
                plt.setp(self.__ax.get_yticklabels(), visible=False)
        return self.__ax


class FreeAxes:

    def __init__(
        self, fig, shapes: Tuple, 
        titles: Optional[Tuple] = None,
        sharey: bool = True, projection: Optional[str] = None,
    ):

        self.fig = fig
        self.axes = []
        self.titles = titles

        grids = fig.add_gridspec(*shapes)
        for i in range(shapes[0]):
            anchor = UnitAX(self, grids[i, 0], anchor=None, sharey=False, projection=projection)
            self.axes.append(anchor)
            for j in range(1, shapes[1]):
                ax = UnitAX(self, grids[i, j], anchor=anchor, sharey=sharey, projection=projection)
                self.axes.append(ax)

        self.links = self._get_links(titles)

    def _get_links(self, titles: Optional[Iterable]) -> Dict:
        n = len(self.axes)
        names = dict()
        if titles is None:
            for i in range(n):
                s = "(" + chr(i + 97) + ")"
                names.update({s:i})
        else:
            for i in range(n):
                title = titles[i]
                names.update({title:i})
        return names

    def set(self, index: Union[int, str, Iterable, None] = None, **kwargs) -> None:
        if isinstance(index, (int, str)):
            index = [index]
        elif index is None:
            index = list(range(len(self.axes)))
        for idx in index:
            ax = self[idx]
            ax.set(**kwargs)

    def set_title(self, y: float = -0.1) -> None:
        for title in self.titles:
            ax = self[title]
            ax.set_title(title, y=y)

    def __iter__(self):
        return (ax.ax for ax in self.axes)

    def __getitem__(self, idx: Union[int, str]):
        if isinstance(idx, str):
            idx = self.links[idx]
        ax = self.axes[idx]
        return ax.ax


class UnitPlot:
    """
    A simple implement is used to draw some easy figures in my sense. 
    It is actually a rewrite based on matplotlib and seaborn as the former 
    is flexible but difficult to use and the latter is eaiser but not flexible.
    Therefore, I try my best to combine the both to make it easy to draw.
    At least, in my opinion, it's helpful.
    """
    def __init__(
        self, 
        shape: Tuple[int, int], 
        figsize: Tuple[float, float], 
        titles: Optional[Iterable]=None,
        sharey: bool = True,
        projection: Optional[str] = None,
        **kwargs: "other kwargs of plt.subplots"
    ):
        """
        If you are familiar with plt.subplots, you will find most of 
        kwargs can be used here directly except
        titles: a list or tuple including the subtitles for differents axes.
        You can ignore this argument and we will assign (a), (b) ... as a default setting.
        Titles will be useful if you want call a axe by the subtitles or endowing the axes 
        different titles together.
        """
        # the default settings
        plt.style.use(cfg.default_style)
        for group, params in cfg['rc_params'].items():
            plt.rc(group, **params)

        self.root = _ROOT
        self.fig = plt.figure(figsize=figsize, **kwargs)
        self.grids = self.fig.add_gridspec(*shape)
        self.axes = FreeAxes(self.fig, shape, titles, sharey, projection=projection)

    def set(self, index: Union[int, str, Iterable, None] = None, **kwargs) -> None:
        self.axes.set(index=index, **kwargs)

    def set_title(self, y: float = -0.3) -> None:
        self.axes.set_title(y=y)

    def set_ticks(self, values, index=0, fmt: str = "%2f", axis: str = 'y') -> Dict:
        labels = [fmt%value for value in values]
        kwargs = dict()
        kwargs['index'] = index
        kwargs[axis + 'ticks'] = values
        kwargs[axis + 'ticklabels'] = labels
        return self.set(**kwargs)

    def set_lim(self, lim: Tuple[float], index=0, axis='y'):
        kwargs = dict()
        kwargs['index'] = index
        kwargs[axis + 'lim'] = lim
        return self.set(**kwargs)

    def set_label(self, label: str, index=0, axis='y'):
        kwargs = dict()
        kwargs['index'] = index
        kwargs[axis + 'label'] = label
        return self.set(**kwargs)

    def legend(
        self, 
        x: float, y: float, ncol: int, 
        index: Union[int, str] = 0, 
        loc: str = "lower left",
        **kwargs
    ) -> None:
        self[index].legend(bbox_to_anchor=(x, y), loc=loc,
        bbox_transform=plt.gcf().transFigure, ncol=ncol, **kwargs)

    def subplots_adjust(
        self,
        left: Optional[float] = None, 
        bottom: Optional[float] = None, 
        right: Optional[float] = None, 
        top: Optional[float] = None, 
        wspace: Optional[float] = None, 
        hspace: Optional[float] = None
    ) -> None:
        plt.subplots_adjust(left, bottom, right, top, wspace, hspace)

    def savefig(
        self, filename: str, 
        bbox_inches: str = 'tight', 
        tight_layout: bool = True,
        **kwargs: "other kwargs of plg.savefig"
    ) -> None:
        if tight_layout:
            plt.tight_layout()
        plt.savefig(
            filename,
            bbox_inches=bbox_inches,
            **kwargs
        )

    def __getitem__(self, index: Union[int, str, Iterable]) -> "ax or [axes]":
        if isinstance(index, (list, tuple)):
            axes = [self.axes[idx] for idx in index]
        else:
            return self.axes[index]

