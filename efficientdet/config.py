import typing


PHIs = list(range(0, 8))


class EfficientDetBaseConfig(typing.NamedTuple):
    # Input scaling
    input_size: int = 512
    # Backbone scaling
    backbone: int = 0
    # BiFPN scaling
    Wbifpn: int = 64
    Dbifpn: int = 2
    # Box predictor head scaling
    Dclass: int = 3


class EfficientDetCompudScaling(object):
    def __init__(self, 
                 config : EfficientDetBaseConfig = EfficientDetBaseConfig(), 
                 D : int = 0):
        assert D in PHIs, 'D must be between [0, 7]'
        self.D = D
        self.base_conf = config
    
    @property
    def input_size(self) -> int:
        return self.base_conf.input_size + PHIs[self.D] * 128
    
    @property
    def Wbifpn(self) -> int:
        return int(self.base_conf.Wbifpn * 1.35 ** PHIs[self.D])
    
    @property
    def Dbifpn(self) -> int:
        return self.base_conf.Dbifpn + PHIs[self.D]
    
    @property
    def Dclass(self) -> int:
        return self.base_conf.Dclass + int(PHIs[self.D] / 3)
    
    @property
    def B(self) -> int:
        return self.base_conf.backbone
    

class AnchorsConfig(typing.NamedTuple):
    sizes: typing.Sequence[int] = (32, 64, 128, 256, 512)
    strides: typing.Sequence[int] = (8, 16, 32, 64, 128)
    ratios: typing.Sequence[float] = (1, 2, .5)
    scales: typing.Sequence[float] = (2 ** 0, 2 ** (1 / 3.0), 2 ** (2 / 3.0))
