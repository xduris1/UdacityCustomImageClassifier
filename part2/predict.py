'''
Author: Roman Duris
'''
import json

from PIL import Image
import torchvision.transforms as T
import torch
import argparse
import numpy as np
from pathlib import Path

# parse command line args
parser = argparse.ArgumentParser(
    prog='Image Classifier Predict Script',
    description='predicts an image on custom image classifier',
    epilog='print -h to see possible arguments'
)
parser.add_argument('path_to_image', type=str)
parser.add_argument('checkpoint', type=str)
parser.add_argument('--top_k', type=int, default=3)
parser.add_argument('--category_names', type=str, default='')
parser.add_argument('--gpu', action='store_true')
args = parser.parse_args()

# function to process image - taken from notebook
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im = Image.open(Path(image_path))
    transforms = T.Compose([
        T.ToTensor(),
        T.Resize((224, 224)),
        T.Normalize(mean=[0.485, 0.465, 0.406], std=[0.229, 0.224, 0.225]),
        T.Lambda(lambda image: image.numpy())
    ])
    im = transforms(im)
    return im

def predict():
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model = torch.load(Path(args.checkpoint))
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    model.to(device)
    model.eval()

    image = process_image(args.path_to_image)
    transform = T.Compose([T.Lambda(lambda im: torch.unsqueeze(torch.from_numpy(im).to(device), 0))])
    image = transform(image)

    clas_idx_dict = model.class_to_idx
    clas_idx_dict = dict([(value, key) for key, value in clas_idx_dict.items()])
    with torch.no_grad():
        out = torch.exp(model(image.to(device)))
    probs, classes = out.topk(args.top_k, dim=1)
    probs = probs.cpu().squeeze().numpy()
    classes = (classes.cpu().numpy().squeeze())
    classes = np.array([clas_idx_dict[i] for i in classes])
    if args.category_names != '':
        with open('cat_to_name.json', 'r') as f:
            cat_to_name = json.load(f)
            classes = [cat_to_name[clas] for clas in classes]
    print(f'top {args.top_k}:')
    for prob, clas in zip(probs, classes):
        print(f"predicted class: {clas:>15} \t\t probability: {prob:.4f}")


if __name__ == '__main__':
    predict()
