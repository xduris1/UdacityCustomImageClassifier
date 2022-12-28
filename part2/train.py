'''
Author: Roman Duris
'''
import torch
from torch import nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.models as models
from collections import OrderedDict
import torch.optim as optim
import sys
from pathlib import Path
import argparse
import tqdm

# Parse arguments from command line
parser = argparse.ArgumentParser(
    prog='Image Classifier Train Script',
    description='trains a custom image classifier',
    epilog='print -h to see possible arguments'
)
parser.add_argument('data_dir', type=str)
parser.add_argument('--save_dir', type=str, default='.')
parser.add_argument('--arch', type=str, default='vgg19')
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--hidden_units', type=int, default=4096)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--gpu', action='store_true')
args = parser.parse_args()


# craft model according to command line arguments
def build_model(num_outs):
    '''
    build the model according to specs
    :param num_outs: number of output classes for the network
    :return: model with custom classifier ready to train on the dataset
    '''
    print(f"creating {args.arch} model with custom classifier")
    match args.arch:
        case 'vgg19':
            model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        case 'vgg16':
            model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        case 'vgg13':
            model = models.vgg13(weights=models.VGG13_Weights.IMAGENET1K_V1)
        case 'vgg11':
            model = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1)
        case 'vgg19_bn':
            model = models.vgg19_bn(weights=models.VGG19_BN_Weights.IMAGENET1K_V1)
        case 'vgg16_bn':
            model = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
        case 'vgg13_bn':
            model = models.vgg13_bn(weights=models.VGG13_BN_Weights.IMAGENET1K_V1)
        case 'vgg11_bn':
            model = models.vgg11_bn(weights=models.VGG11_BN_Weights.IMAGENET1K_V1)
        case _:
            print("Unsupported Arch")
            sys.exit(1)
    for param in model.parameters():
        param.requires_grad = False
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(in_features=25088, out_features=args.hidden_units, bias=True)),
        ('relu1', nn.ReLU(inplace=True)),
        ('dropout1', nn.Dropout(p=0.2, inplace=False)),
        ('fc2', nn.Linear(in_features=args.hidden_units, out_features=args.hidden_units, bias=True)),
        ('relu2', nn.ReLU(inplace=True)),
        ('dropout2', nn.Dropout(p=0.2, inplace=False)),
        ('fc3', nn.Linear(in_features=args.hidden_units, out_features=num_outs, bias=True)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    model.classifier = classifier

    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    model.to(device)
    print(f"model {args.arch} created")
    return model


# set up dataset, train and save the model
def train():
    '''
    Function to create datasets and train and export model
    I know I should decompose this but have no time to spare
    '''
    print('creating datasets and dataloaders')
    data_dir = Path(args.data_dir)
    train_dir = data_dir.joinpath('train')
    valid_dir = data_dir.joinpath('valid')
    test_dir = data_dir.joinpath('test')

    data_transforms = {'train': T.Compose([
        # T.RandomCrop(size=(112, 112)), -> this was causing more trouble than benefits regarding accuracy
        T.RandomRotation(180),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.465, 0.406], std=[0.229, 0.224, 0.225])
    ]),
        'test': T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.465, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val': T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.465, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

    image_datasets = {'train': ImageFolder(str(train_dir), transform=data_transforms['train']),
                      'test': ImageFolder(str(test_dir), transform=data_transforms['test']),
                      'val': ImageFolder(str(valid_dir), transform=data_transforms['val'])}

    dataloaders = {'train': DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
                   'test': DataLoader(image_datasets['test'], batch_size=64),
                   'val': DataLoader(image_datasets['val'], batch_size=64)}
    print('datasets and dataloaders created')
    criterion = nn.NLLLoss()
    model = build_model(len(image_datasets['train'].class_to_idx))
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    steps = 0
    running_loss = 0
    print_every = 5
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    pbar = tqdm.tqdm(range(args.epochs), f"training model on {device}")

    # Train the model, heavily based on the training script from intro to Pytorch
    for epoch in pbar:
        for inputs, labels in dataloaders['train']:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            logps = model.forward(inputs)
            loss = criterion(logps, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in dataloaders['test']:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                pbar.set_description_str(f"Epoch {epoch + 1}/{args.epochs}.. "
                                         f"Train loss: {running_loss / print_every:.3f}.. "
                                         f"Test loss: {test_loss / len(dataloaders['test']):.3f}.. "
                                         f"Test accuracy: {accuracy / len(dataloaders['test']):.3f}")
                pbar.refresh()
                running_loss = 0
                model.train()
            model.class_to_idx = image_datasets['train'].class_to_idx
            export_dir = Path(args.save_dir)
            export_path = export_dir.joinpath(
                f'arch_{args.arch}_hidden_{args.hidden_units}_eps_{args.epochs}_data_{data_dir}.pth')
            torch.save(model, export_path)


if __name__ == '__main__':
    train()
