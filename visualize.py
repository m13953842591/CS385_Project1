from PIL import Image
from config import *
import glob
import cv2
import torch
import tensorboardX as tbx
from data_utils.hog import hog_skimage
from torchvision import transforms
# PROJECTOR需要的日志文件名和地址相关参数
LOG_DIR = 'logs'
SPRITE_FILE = 'sprite.jpg'
META_FIEL = "meta.tsv"
TENSOR_NAME = "FINAL_LOGITS"

transform = transforms.Compose([
    transforms.Resize((50, 50)),
    transforms.ToTensor(),
])


def main(argv=None):
    image_paths = glob.glob(os.path.join(DATA_PATH, "image_small", "*.jpg"))
    n = len(image_paths)
    label_imgs = torch.zeros(0)
    labels = [0.0] * n
    features = torch.zeros((n, 900), dtype=torch.float32)

    for i in range(n):
        img = cv2.imread(image_paths[i], cv2.IMREAD_COLOR)
        feature = torch.tensor(hog_skimage(img), dtype=torch.float32)
        features[i] = feature
        label_img = transform(Image.open(image_paths[i]).convert('RGB'))
        label_imgs = torch.cat((label_imgs, label_img))
        labels[i] = int(image_paths[i].split('.')[-2] == "00")

    features = features.view(n, 900)
    label_imgs = label_imgs.view(n, 3, 50, 50)
    writer = tbx.SummaryWriter()
    writer.add_embedding(features, metadata=labels, label_img=label_imgs)
    writer.close()


if __name__ == '__main__':
    main()