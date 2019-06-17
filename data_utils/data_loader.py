import glob
import os
from torch.utils.data import Dataset
from torchvision import transforms
from skimage import io
import torch


class FaceClassificationDataset(Dataset):
    """
    Face classification dataset
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.images_path = glob.glob(os.path.join(root_dir, "*.jpg"))
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        image_name = os.path.join(self.root_dir, self.images_path[item])
        image = io.imread(image_name)
        l = int(image_name.split('.')[-2] == "00")
        label = torch.tensor(l, dtype=torch.long)
        _sample = {'image': self.transform(image),
                   'label': label}
        return _sample


def categorical_accuracy(y_pred, y_true):
    # y_pred is [N, C]
    # y_true is [N, 1]
    # output is in shape []
    argmax1 = torch.argmax(y_pred, dim=1)
    size0 = y_true.size()[0]
    sum = torch.sum(torch.eq(y_true, argmax1), dtype=torch.float32)
    return torch.div(sum, size0)


if __name__ == '__main__':
    a = torch.tensor([[0, 1], [1, 0]], dtype=torch.float32)
    b = torch.tensor([[0.9, 0.1], [0.9, 0.1]], dtype=torch.float32)
    print(categorical_accuracy(a, b))
