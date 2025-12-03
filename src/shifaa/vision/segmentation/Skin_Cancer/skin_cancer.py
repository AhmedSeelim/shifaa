import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore")


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU()
            )
        self.enc1 = conv_block(1, 16); self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(16, 32); self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(32, 64); self.pool3 = nn.MaxPool2d(2)
        self.enc4 = conv_block(64, 128); self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = conv_block(128, 256)
        self.up4 = nn.ConvTranspose2d(256, 128, 2, 2); self.dec4 = conv_block(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, 2); self.dec3 = conv_block(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, 2); self.dec2 = conv_block(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, 2, 2); self.dec1 = conv_block(32, 16)
        self.final = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        c1 = self.enc1(x); p1 = self.pool1(c1)
        c2 = self.enc2(p1); p2 = self.pool2(c2)
        c3 = self.enc3(p2); p3 = self.pool3(c3)
        c4 = self.enc4(p3); p4 = self.pool4(c4)
        bn = self.bottleneck(p4)
        u4 = torch.cat([self.up4(bn), c4], 1); d4 = self.dec4(u4)
        u3 = torch.cat([self.up3(d4), c3], 1); d3 = self.dec3(u3)
        u2 = torch.cat([self.up2(d3), c2], 1); d2 = self.dec2(u2)
        u1 = torch.cat([self.up1(d2), c1], 1); d1 = self.dec1(u1)
        return torch.sigmoid(self.final(d1))


class skin_cancer:

    def __init__(self, model_path, input_size=(128, 128), device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size

        self.model = UNet()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])

    def predict(self, image_path, show_image=False):
        # Open and resize image to match model input
        image = Image.open(image_path).convert('L')
        resized_image = image.resize(self.input_size, resample=Image.BILINEAR)
        input_tensor = transforms.ToTensor()(resized_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            pred_mask = (output > 0.5).float().cpu().squeeze().numpy()

        if show_image:
            fig, axs = plt.subplots(1, 2, figsize=(8, 4))
            axs[0].imshow(np.array(resized_image), cmap='gray')
            axs[0].set_title("Resized Image")
            axs[0].axis('off')

            axs[1].imshow(pred_mask, cmap='gray')
            axs[1].set_title("Predicted Mask")
            axs[1].axis('off')
            plt.tight_layout()
            plt.show()

        return {
            'image': np.array(resized_image),
            'predicted_mask': pred_mask
        }




