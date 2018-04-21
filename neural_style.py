import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import utils
from transformer_net import TransformerNet
from vgg import Vgg16
import enums


def check_paths():
    try:
        if not os.path.exists(enums.save_model_dir):
            os.makedirs(enums.save_model_dir)
        if enums.checkpoint_model_dir is not None and not (os.path.exists(enums.checkpoint_model_dir)):
            os.makedirs(enums.checkpoint_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)


def save_checkpoint(state, filename):
    torch.save(state, filename)


def train(start_epoch = 0):
    np.random.seed(enums.seed)
    torch.manual_seed(enums.seed)

    if enums.cuda:
        torch.cuda.manual_seed(enums.seed)

    transform = transforms.Compose([
        transforms.Resize(enums.image_size),
        transforms.CenterCrop(enums.image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    train_dataset = datasets.ImageFolder(enums.dataset, transform)
    train_loader = DataLoader(train_dataset, batch_size=enums.batch_size)

    transformer = TransformerNet()
    if enums.subcommand == 'resume':
        ckpt_state = torch.load(enums.checkpoint_model_dir)
        transformer.load_state_dict(ckpt_state['state_dict'])
        start_epoch = ckpt_state['epoch']
        optimizer.load_state_dict(ckpt_state['optimizer'])
    else:
        optimizer = Adam(transformer.parameters(), enums.lr)
    
    mse_loss = torch.nn.MSELoss()

    vgg = Vgg16(requires_grad=False)
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    style = utils.load_image(enums.style_image, size=enums.style_size)
    style = style_transform(style)
    style = style.repeat(enums.batch_size, 1, 1, 1) # N,C,H,W

    if enums.cuda:
        transformer.cuda()
        vgg.cuda()
        style = style.cuda()

    style_v = Variable(style)
    style_v = utils.normalize_batch(style_v)
    features_style = vgg(style_v)
    gram_style = [utils.gram_matrix(y) for y in features_style]

    for e in range(start_epoch, enums.epochs):
        transformer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0
        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()
            x = Variable(x)
            if enums.cuda:
                x = x.cuda()

            y = transformer(x)

            y = utils.normalize_batch(y)
            x = utils.normalize_batch(x)

            features_y = vgg(y)
            features_x = vgg(x)

            content_loss = enums.content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)

            style_loss = 0.
            for ft_y, gm_s in zip(features_y, gram_style):
                gm_y = utils.gram_matrix(ft_y)
                style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
            style_loss *= enums.style_weight

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            agg_content_loss += content_loss.data[0]
            agg_style_loss += style_loss.data[0]

            if (batch_id + 1) % enums.log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), e + 1, count, len(train_dataset),
                                  agg_content_loss / (batch_id + 1),
                                  agg_style_loss / (batch_id + 1),
                                  (agg_content_loss + agg_style_loss) / (batch_id + 1)
                )
                print(mesg)

        if enums.checkpoint_model_dir is not None and (e + 1) % enums.checkpoint_interval == 0:
            # transformer.eval()
            if enums.cuda:
                transformer.cpu()
            ckpt_model_filename = "ckpt_epoch_" + str(e) + ".pth"
            ckpt_model_path = os.path.join(enums.checkpoint_model_dir, ckpt_model_filename)
            save_checkpoint({'epoch': e + 1, 'state_dict': transformer.state_dict(), 'optimizer': optimizer.state_dict()}, ckpt_model_path)
            if enums.cuda:
                transformer.cuda()
            # transformer.train()

    # save model
    # transformer.eval()
    if enums.cuda:
        transformer.cpu()
    save_model_filename = "epoch_" + str(enums.epochs) + "_" + str(time.ctime()).replace(' ', '_') + "_" + str(
        enums.content_weight) + "_" + str(enums.style_weight) + ".model"
    save_model_path = os.path.join(enums.save_model_dir, save_model_filename)
    save_checkpoint({'epoch': e + 1, 'state_dict': transformer.state_dict(), 'optimizer': optimizer.state_dict()}, save_model_path)

    print("\nDone, trained model saved at", save_model_path)


def stylize(model_path):
    content_image = utils.load_image(enums.content_image, scale=enums.content_scale)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0)
    if enums.cuda:
        content_image = content_image.cuda()
    content_image = Variable(content_image, volatile=True)

    model_state = torch.load(model_path)
    style_model = TransformerNet()
    style_model.load_state_dict(model_state['state_dict'])
    style_model.eval()

    if enums.cuda:
        style_model.cuda()
    output = style_model(content_image)
    if enums.cuda:
        output = output.cpu()
    output_data = output.data[0]
    utils.save_image(enums.output_image, output_data)


def main():

    if enums.subcommand is None:
        print("ERROR: specify either train or eval")
        sys.exit(1)
    if enums.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

    if enums.subcommand == 'train' or enums.subcommand == 'resume':
        check_paths()
        train()
    else:
    	model_path = enums.model # change to enums.checkpoint_model to use that model for stylize
        stylize(model_path)


if __name__ == "__main__":
    main()