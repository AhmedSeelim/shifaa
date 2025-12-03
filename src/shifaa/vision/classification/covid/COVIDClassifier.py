
import torch
import torch.nn as nn
from PIL import Image as PILImage
import matplotlib.pyplot as plt
from torchvision import transforms, models
from PIL import Image
import os

class COVIDClassifier:
    def __init__(self, model_path, device=None):
        # Class labels
        self.class_names = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]
        
        # Device configuration
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[INFO] Using device: {self.device}")

        # Define model architecture
        self.model = models.resnet50(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(self.class_names))

        # Load trained weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

        # Define transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def infer(self, image_path,show_image=False):
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"[ERROR] Failed to open image: {e}")
            return None

        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            class_index = torch.argmax(probabilities).item()
            confidence = probabilities[0][class_index].item()

        prediction = {
            'predicted_class': self.class_names[class_index],
            'confidence': confidence
        }

        if show_image:
            img = PILImage.open(image_path)
            plt.imshow(img)
            plt.title(f"{prediction['predicted_class']} ({confidence * 100:.2f}%)")
            plt.axis('off')
            plt.show()

        return prediction

    def predict_directory(self, folder_path):
        results = []
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(folder_path, filename)
                result = self.infer(file_path)
                if result:
                    results.append(result)
                    print(f"[INFO] File: {result['filename']} | Prediction: {result['predicted_class']} | Confidence: {result['confidence']:.4f}")
        return results

