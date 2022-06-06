import numpy as np

from model.attribute import Attribute
from model.dataset import Dataset

def read_arff(path):
    attributes = []
    datas = []
    lines = [line.rstrip('\n') for line in open(path)]

    for line in lines:
        if line.startswith('@attribute'):
            split = line.split(' ', 2)
            _name = split[1]
            _type = ''
            _properties = []

            if split[1] == 'class':
                _type = 'class'
                formated_proterties = split[2].replace('{', '').replace('}', '').replace(' ', '')
                _properties = [_property for _property in formated_proterties.split(',')]
            elif split[2] == 'numeric' or split[2] == 'NUMERIC':
                _type = 'numeric'
            else:
                print('未対応の特徴量形式が書き込まれています')
            
            attribute = Attribute(_name, _type, _properties)
            attributes.append(attribute)
        elif not line.startswith('@data') and not line.startswith('@relation') and not line.startswith('%'):
            data = line.replace(' ', '').split(',')
            datas = [data] if np.size(datas) == 0 else np.insert(datas, 0, data, axis = 0)
    
    datas = np.where(datas == '?', np.NaN, datas)

    return Dataset(attributes = attributes, datas = datas)
