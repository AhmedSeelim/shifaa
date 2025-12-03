import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image as PILImage
import matplotlib.pyplot as plt
import os

class BrainTumorClassifier:
    def __init__(self, model_weights_path, input_size=(224, 224), device=None):
        """
        Initialize the Brain Tumor Classifier.

        Args:
            model_weights_path (str): Path to the trained model weights.
            input_size (tuple): Input size for model input.
            device (str): Device to use ('cuda' or 'cpu'). Auto-detected by default.
        """
        self.class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
        self.input_size = input_size
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"[INFO] Using device: {self.device}")

        # Load and modify ResNet18
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(self.class_labels))

        state_dict = torch.load(model_weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # Define preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def preprocess_image(self, image_path):
        """
        Preprocess the input image.

        Args:
            image_path (str): Path to the image.

        Returns:
            torch.Tensor: Preprocessed image tensor.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = PILImage.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return self.transform(image).unsqueeze(0)

    def infer(self, image_path, show_image=False):
        """
        Perform inference on a single image.

        Args:
            image_path (str): Path to the image.
            show_image (bool): Whether to display the image with prediction.

        Returns:
            dict: Dictionary containing predicted label and confidence score.
        """
        input_tensor = self.preprocess_image(image_path).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)
            class_index = torch.argmax(probs).item()
            confidence = probs[0][class_index].item()

        prediction = {
            'predicted_class': self.class_labels[class_index],
            'confidence': confidence
        }

        if show_image:
            img = PILImage.open(image_path)
            plt.imshow(img)
            plt.title(f"{prediction['predicted_class']} ({confidence * 100:.2f}%)")
            plt.axis('off')
            plt.show()

        return prediction


# # === Example Usage ===
# image_path = r"D:\ff\Vision Models\Classification Tasks\Brain Tumer\sample\glioma\Te-gl_0013.jpg"
# classifier = BrainTumorClassifier(model_weights_path=model_path)
# result = classifier.infer(image_path, show_image=True)







