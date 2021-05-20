



from typing import Tuple, Optional, Dict, Union
import numpy as np
import pandas as pd 
import seaborn as sns

from .config import cfg
from .unit import UnitPlot
from .utils import style_env

 

class FreePlot(UnitPlot):

    @style_env(cfg.heatmap_style)
    def heatmap(
        self, data: pd.DataFrame, 
        index: Union[int, str] = 0, 
        annot: bool = True, 
        fmt: str = ".4f",
        cmap: str = 'GnBu', 
        linewidth: float = .5,
        **kwargs: "other kwargs of sns.heatmap"
    ) -> None:
        """
        data: M x N dataframe.
        cmap: GnBu, Oranges are recommanded.
        annot: annotation.
        fmt: the format for annotation.
        kwargs:
            cbar: bool
        """
        ax = self[index]
        sns.heatmap(
            data, ax=ax, 
            annot=annot, fmt=fmt,
            cmap=cmap, linewidth=linewidth,
            **kwargs
        )

    @style_env(cfg.lineplot_style)
    def lineplot(
        self, x: np.ndarray, y: np.ndarray, 
        index: Union[int, str] = 0, 
        seaborn: bool = False, 
        **kwargs: "other kwargs of ax.plot or sns.lineplot"
    ) -> None:
        ax = self[index]
        if seaborn:
            sns.lineplot(x, y, ax=ax, **kwargs)
        else:
            ax.plot(x, y, **kwargs)
        
    @style_env(cfg.scatterplot_style)
    def scatterplot(
        self, x: np.ndarray, y: np.ndarray, 
        index: Union[int, str] = 0, 
        seaborn: bool = False, 
        **kwargs: "other kwargs of ax.scatter or sns.scatterplot"
    ) -> None:
        ax = self[index]
        if seaborn:
            sns.scatterplot(x, y, ax=ax, **kwargs)
        else:
            ax.scatter(x, y, **kwargs)

    @style_env(cfg.imageplot_style)
    def imageplot(
        self, img: np.ndarray, 
        index: Union[int, str]=0, 
        show_ticks: bool = False, 
        **kwargs: "other kwargs of ax.imshow"
    ) -> None:
        ax = self[index]
        try:
            assert img.shape[2] == 3
            ax.imshow(img, **kwargs)
        except AssertionError:
            ax.imshow(img.squeeze(), cmap="gray", **kwargs)
        if not show_ticks:
            ax.set(xticks=[], yticks=[])

    @style_env(cfg.barplot_style)
    def barplot(
        self, x: str, y: str, hue: str, 
        data: pd.DataFrame, 
        index: Union[int, str] = 0, 
        auto_fmt: bool = False,
        **kwargs: "other kwargs of sns.barplot"
    ) -> None:
        ax = self[index]
        sns.barplot(x=x, y=y, hue=hue, data=data, ax=ax, **kwargs)
        if auto_fmt:
            self.fig.autofmt_xdate()

   
        




