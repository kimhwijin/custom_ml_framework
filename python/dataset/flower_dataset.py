from calendar import day_abbr
from python.dataset.dataset import Dataset

class FlowerDataset(Dataset):
    def __init__(self, image_size=[100, 100], input_shape=[-1]):
        super(FlowerDataset, self).__init__('flowers', 'select')
        