import numpy as np
import pandas as pd

class Dataset:
    def __init__(self, attributes:list, datas:np.ndarray) -> None:
        self.attributes = attributes
        self.datas = datas
        self.pandas = self.create_pandas()

    def create_pandas(self):
        pandas = pd.DataFrame(self.datas, columns = [attribute.name for attribute in self.attributes])
        futures = pandas.drop(columns = 'class').astype(float)
        classes = pandas[['class']].astype(str)

        return pd.concat([futures, classes], axis = 1)
