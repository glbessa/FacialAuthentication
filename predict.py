import os
import sys
import random

import matplotlib.pyplot as plt

from model import SiameseNetwork

import torch
from torch.utils.io import read_image

def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model_path = sys.argv[1]
    image1_path = sys.argv[2]
    image2_path = sys.argv[3]

    image1 = read_image(image1_path)
    image1 = image1[None, :, :, :]
    image1 = image1.to(device)

    image2 = read_image(image2_path)
    image2 = image2[None, :, :, :]
    image2 = image2.to(device)

    model = SiameseNetwork()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    with torch.no_grad():
        result = model(image1, image2)
        result = torch.where(result > 0.5, 1, 0).to("cpu")

    fig = plt.figure()

    text_result = "Usuário não reconhecido! Chamando segurança..."
    if result[0] == 1:
        text_result = "Usuário reconhecido! Acesso liberado."

    plt.suptitle(text_result)

    plt.subplot(1, 2, 1)
    plt.imshow(image1[0].permute(1, 2, 0))
    plt.title(f"Image 1")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(image2[0].permute(1, 2, 0))
    plt.title(f"Image 2")
    plt.axis('off')

    plt.show()
    input()


if __name__ == '__main__':
    main()