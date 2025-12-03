
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore")


class DiabeticRetinopathyClassifier:
    def __init__(self, model_path, input_size=(224, 224), device=None):
        """
        Initialize the Diabetic Retinopathy Classifier.

        Args:
            model_path (str): Path to the trained model weights.
            input_size (tuple): Input size for the model.
            device (str): Device to use ('cuda' or 'cpu').
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = ['Mild', 'Moderate', 'No_DR', 'Proliferate_DR', 'Severe']
        self.input_size = input_size

        # Load pretrained EfficientNet B0 and modify classifier
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        self.model = models.efficientnet_b0(weights=weights)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, len(self.class_names))

        # Load trained weights
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
        ])

    def infer(self, image_path, show_image=False):
        """
        Perform inference on a single image.

        Args:
            image_path (str): Path to the image.
            show_image (bool): Whether to display the image with prediction.

        Returns:
            dict: Prediction result with class label and confidence.
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

#
# model_path = r'E:\graduation projects\Vision Models\DR\DiabeticRetinopathy.pth'
# classifier = DiabeticRetinopathyClassifier(model_path=Model_Path)
# image_path = r'D:\ff\Vision Models\classification\DR\sample\moderate\0fffa73e2402.png'
# result = classifier.infer(image_path, show_image=True)
# print(result)





