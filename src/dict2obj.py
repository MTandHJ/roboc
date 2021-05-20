




from typing import Any

# Convert the dict to a object.


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
    >>> cfg.update(**Config({'a':4, 'd':5, 'e':6}))
    >>> cfg.a
    4
    >>> cfg['d']
    Traceback (most recent call last):
    ...
    KeyError: 'd'
    >>> cfg.e
    Traceback (most recent call last):
    ...
    AttributeError: 'Config' object has no attribute 'e'
    '''
    def __init__(
        self, *args, 
        prefix: str = "",
        **kwargs
    ):
        super(Config, self).__init__(*args, **kwargs)
        for name, attr in self.items():
            self.__setattr__(name, attr)
        
        self.prefix = prefix

    def __setitem__(self, key: str, value: Any) -> None:
        super(Config, self).__setitem__(key, value)
        self.__setattr__(key, value)

    def update(self, **kwargs) -> None:
        # Note that, we only update the keys and 
        # correspoding values existed.
        for key, value in kwargs.items():
            if key in self.keys():
                self[key] = value

    def __str__(self) -> str:
        item = " [{name}: {val}] "
        infos = "վ'ᴗ' ի-" + self.prefix + "   "
        for name, val in self.items():
            infos += item.format(name=name, val=val)
        return infos





if __name__ == "__main__":
    import doctest
    doctest.testmod()











