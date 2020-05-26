import math
import typing


# D7 the same as D6, therefore we repeat the 6 PHI
PHIs = list(range(0, 7)) + [6]


class EfficientDetBaseConfig(typing.NamedTuple):
    # Input scaling
    input_size: int = 512
    # Backbone scaling
    backbone: int = 0
    # BiFPN scaling
    Wbifpn: int = 64
    Dbifpn: int = 3
    # Box predictor head scaling
    Dclass: int = 3

    def print_table(self, min_D: int = 0, max_D: int = 7) -> None:
        for i in range(min_D, max_D + 1):
            EfficientDetCompudScaling(D=i).print_conf()


class EfficientDetCompudScaling(object):
    def __init__(self, 
                 config : EfficientDetBaseConfig = EfficientDetBaseConfig(), 
                 D : int = 0):
        assert D >= 0 and D <= 7, 'D must be between [0, 7]'
        self.D = D
        self.base_conf = config
    
    @property
    def input_size(self) -> typing.Tuple[int, int]:
        if self.D == 7:
            size = 1536
        else:
            size = self.base_conf.input_size + PHIs[self.D] * 128
        return size, size
    
    @property
    def Wbifpn(self) -> int:
        return int(self.base_conf.Wbifpn * 1.35 ** PHIs[self.D])
    
    @property
    def Dbifpn(self) -> int:
        return self.base_conf.Dbifpn + PHIs[self.D]
    
    @property
    def Dclass(self) -> int:
        return self.base_conf.Dclass + math.floor(PHIs[self.D] / 3)
    
    @property
    def B(self) -> int:
        return self.D
    
    def print_conf(self) -> None:
        print(f'D{self.D} | B{self.B} | {self.input_size:5d} | '
              f'{self.Wbifpn:4d} | {self.Dbifpn} | {self.Dclass} |')
    

class AnchorsConfig(typing.NamedTuple):
    sizes: typing.Sequence[int] = (32, 64, 128, 256, 512)
    strides: typing.Sequence[int] = (8, 16, 32, 64, 128)
    ratios: typing.Sequence[float] = (1, 2, .5)
    scales: typing.Sequence[float] = (2 ** 0, 2 ** (1 / 3.0), 2 ** (2 / 3.0))
