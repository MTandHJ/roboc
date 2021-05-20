



from cycler import cycler


class Config(dict):
    '''
    >>> cfg = Config({1:2}, a=3)
    Traceback (most recent call last):
    ...
    TypeError: attribute name must be string, not 'int'
    >>> cfg = Config(a=1, b=2)
    >>> cfg.a
    1
    >>> cfg['a']
    1
    >>> cfg['c'] = 3
    >>> cfg.c
    3
    >>> cfg.d = 4
    >>> cfg['d']
    Traceback (most recent call last):
    ...
    KeyError: 'd'
    '''
    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)
        for name, attr in self.items():
            self.__setattr__(name, attr)

    def __setitem__(self, key, value):
        super(Config, self).__setitem__(key, value)
        self.__setattr__(key, value)





COLORS = ('#1f77b4',
          '#ff7f0e',
          '#2ca02c',
          '#d62728',
          '#9467bd',
          '#8c564b',
          '#e377c2',
          '#7f7f7f',
          '#bcbd22',
          '#17becf',
          '#1a55FF')

MARKERS = (
    'o',
    '+',
    'x',
    'p',
    'h',
    'd',
    '1',
    '2',
    '3',
    '4',
    '*'
)

cfg = Config()

#filepath
cfg['root'] = "./media"

#rc

_axes = {
    "prop_cycle": cycler(marker=MARKERS, color=COLORS),
    "titlesize": 'large',
    "labelsize": 'large',
    "facecolor": 'white',
}

_font = {
        "family": ["serif"],
        "weight": "normal",
        "size": 5
    }

_lines = {
    "linewidth": 1.,
    #"marker": None,
    "markersize": 4,
    "markeredgewidth": 1.,
    #"markerfacecolor": "auto",
    #"markeredgecolor": "white",
}

_markers = {
    "fillstyle": "none"
}


_legend = {
    'borderaxespad': 0.5,
    'borderpad': 0.4,
    'columnspacing': 2.0,
    'edgecolor': '0.8',
    'facecolor': 'inherit',
    'fancybox': True,
    'fontsize': 'medium',
    'framealpha': 0.8,
    'frameon': True,
    'handleheight': 0.7,
    'handlelength': 2.0,
    'handletextpad': 0.8,
    'labelspacing': 0.5,
    'loc': 'best',
    'markerscale': 1.0,
    'numpoints': 1,
    'scatterpoints': 1,
    'shadow': False,
    'title_fontsize': None,
}

_xtick = {
    'alignment': 'center',
    'bottom': True,
    'color': 'black',
    'direction': 'out',
    'labelbottom': True,
    'labelsize': 'large',
    'labeltop': False,
    'major.bottom': True,
    'major.pad': 3.5,
    'major.size': 3.5,
    'major.top': True,
    'major.width': 0.5,
    'minor.bottom': True,
    'minor.pad': 3.4,
    'minor.size': 2.0,
    'minor.top': True,
    'minor.visible': False,
    'minor.width': 0.4,
    'top': False,
}

_ytick = {
    'alignment': 'center_baseline',
    'color': 'black',
    'direction': 'out',
    'labelleft': True,
    'labelright': False,
    'labelsize': 'large',
    'left': True,
    'major.left': True,
    'major.pad': 3.5,
    'major.right': True,
    'major.size': 3.5,
    'major.width': 0.5,
    'minor.left': True,
    'minor.pad': 3.4,
    'minor.right': True,
    'minor.size': 2.0,
    'minor.visible': False,
    'minor.width': 0.4,
    'right': False
}

cfg['rc_params'] = Config(
    axes=_axes,
    font=_font,
    lines=_lines,
    markers=_markers,
    xtick=_xtick,
    ytick=_ytick
)

cfg.default_style = ["science", "no-latex"]  # color style: bright, vibrant, muted, high-contrast, light, high-vis, retro
cfg.lineplot_style = ['bright'] # use  'grid' to add markers
cfg.scatterplot_style = ["bright"]
cfg.heatmap_style = ["seaborn-darkgrid", {"axes.facecolor":".9"}]
cfg.imageplot_style = ["seaborn-white"]
cfg.barplot_style = ["seaborn-darkgrid", "high-vis", {"axes.facecolor":".9"}]


# zoo
zoo_cfg = Config()
zoo_cfg.radar_style = ['seaborn-whitegrid']

    

