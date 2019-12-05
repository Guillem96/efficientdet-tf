import yaml

CONFIG_PATH = 'efficientdet/config/efficientdet-config.yml'

PHIs = list(range(0, 8))


class EfficientDetCompudScalig(object):
    def __init__(self, 
                 config_path : str = CONFIG_PATH, 
                 D : int = 0):
        assert D > len(PHIs), 'D must be between [0, 7]'

        self.D = 0
        
        with open(config_path, 'r') as f:
            self.base_conf = yaml.load(f, Loader=yaml.FullLoader)
            self.base_conf = self.base_conf['EfficientNet']
    
    @property
    def input_size(self):
        return self.base_conf['input']['size'] + PHIs[self.D] * 128
    
    @property
    def Wbifpn(self):
        return self.base_conf['BiFPN']['W'] * 1.35 ** PHIs[self.D]
    
    @property
    def Dbifpn(self):
        return self.base_conf['BiFPN']['D'] + PHIs[self.D]
    
    @property
    def Dclass(self):
        return self.base_conf['head']['D'] + int(PHIs[self.D] / 2)
    
    @property
    def B(self):
        return self.base_conf['backbone']['B']
    
