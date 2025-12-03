import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from efficientnet_pytorch import EfficientNet

import warnings
warnings.filterwarnings("ignore")

class EyeDiseaseClassifier:
    def __init__(self, model_path, input_size=(224, 224), device=None):
        """
        Initialize the Eye Disease Classification class.

        Args:
            model_path (str): Path to the trained model weights.
            input_size (tuple): Input size for the model.
            device (str): Device to use ('cuda' or 'cpu').
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
        self.input_size = input_size

        # Initialize EfficientNet-b0 architecture
        self.model = EfficientNet.from_name('efficientnet-b0')
        self.model._fc = nn.Sequential(
            nn.Linear(self.model._fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, len(self.class_names))
        )

        # Load trained weights
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
        ])

    def infer(self, image_path, show_image=False):
        """
        Perform inference on an input image.

        Args:
            image_path (str): Path to the input image.
            show_image (bool): Whether to display the image with prediction.

        Returns:
            dict: Predicted label and confidence score.
        """
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)
            class_index = torch.argmax(probs).item()
            confidence = probs[0][class_index].item()

        if show_image:
            npimg = input_tensor.cpu().squeeze(0).numpy()
            npimg = np.transpose(npimg, (1, 2, 0))
            plt.imshow(npimg)
            plt.title(f"{self.class_names[class_index]} ({confidence * 100:.2f}%)")
            plt.axis('off')
            plt.show()

        return {'predicted_class': self.class_names[class_index], 'confidence': confidence}







