import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, Subset
import gradio as gr
from PIL import Image
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disease information database
DISEASE_INFO = {
    "Apple___Apple_scab": {
        "disease_name": "Apple Scab",
        "description": "A fungal disease that causes dark, scaly lesions on leaves and fruit.",
        "symptoms": [
            "Dark olive-green spots on leaves",
            "Dark, scaly lesions on fruit",
            "Twisted or distorted leaves",
            "Premature leaf drop"
        ],
        "treatment": [
            "Apply fungicides early in the growing season",
            "Remove fallen leaves and infected fruit",
            "Improve air circulation by pruning",
            "Plant resistant apple varieties"
        ],
        "prevention": [
            "Space trees for good air circulation",
            "Sanitize pruning tools",
            "Water at the base of trees",
            "Apply preventive fungicides before rainy periods"
        ]
    },
    "Apple___Black_rot": {
        "disease_name": "Black Rot",
        "description": "A fungal disease affecting apples, causing rotting of fruit and leaf spots.",
        "symptoms": [
            "Purple spots on leaves",
            "Rotting fruit with dark rings",
            "Cankers on branches",
            "Leaf drop"
        ],
        "treatment": [
            "Remove infected fruit and branches",
            "Apply appropriate fungicides",
            "Prune out dead or diseased wood",
            "Maintain tree vigor with proper fertilization"
        ],
        "prevention": [
            "Remove mummified fruit",
            "Prune during dry weather",
            "Maintain proper tree spacing",
            "Apply protective fungicides"
        ]
    },
    "Apple___healthy": {
        "disease_name": "Healthy Apple Plant",
        "description": "The plant shows no signs of disease.",
        "symptoms": ["No visible symptoms of disease"],
        "treatment": ["No treatment needed"],
        "prevention": [
            "Continue regular monitoring",
            "Maintain good orchard hygiene",
            "Follow recommended fertilization schedule",
            "Proper pruning and maintenance"
        ]
    },
    # Add more diseases for other plants...
}


class PlantDiseaseModel(nn.Module):
    def __init__(self, num_classes):
        super(PlantDiseaseModel, self).__init__()
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        n_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(n_features, num_classes)

    def forward(self, x):
        return self.model(x)


class Config:
    data_dir = "dataset/PlantVillage"
    selected_classes = ['Apple', 'Blueberry', 'Corn']
    batch_size = 4
    num_epochs = 1
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = 224
    train_split = 0.8
    val_split = 0.1
    test_split = 0.1
    model_path = 'plant_disease_model.pth'


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((Config.image_size, Config.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((Config.image_size, Config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def format_prediction_output(class_name, confidence):
    """Format the prediction output with detailed information"""
    try:
        # Split class name into plant and condition
        plant, condition = class_name.split('___')

        # Get disease information
        disease_info = DISEASE_INFO.get(class_name, {
            "disease_name": condition.replace('_', ' '),
            "description": "Information not available",
            "symptoms": ["Information not available"],
            "treatment": ["Information not available"],
            "prevention": ["Information not available"]
        })

        # Format output
        output = f"""
## Analysis Results

🌱 **Plant Type:** {plant}
🔍 **Condition:** {disease_info['disease_name']}
📊 **Confidence:** {confidence:.2f}%

### Description
{disease_info['description']}

### Symptoms
{"".join(['• ' + s + '\n' for s in disease_info['symptoms']])}

### Recommended Treatment
{"".join(['• ' + t + '\n' for t in disease_info['treatment']])}

### Prevention Tips
{"".join(['• ' + p + '\n' for p in disease_info['prevention']])}

Note: Please consult with a local agricultural expert for confirmation and specific treatment advice.
"""
        return output
    except Exception as e:
        logger.error(f"Error formatting prediction output: {str(e)}")
        return f"Error processing results: {str(e)}"


def predict_disease(image):
    """Modified prediction function with detailed output"""
    try:
        # Load model and classes
        checkpoint = torch.load(Config.model_path, map_location=Config.device)
        class_names = checkpoint['class_names']

        model = PlantDiseaseModel(num_classes=len(class_names))
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(Config.device)
        model.eval()

        # Process image
        transform = get_transforms()[1]
        if isinstance(image, str):
            image = Image.open(image)
        image = image.convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(Config.device)

        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        predicted_class = class_names[predicted.item()]
        confidence_value = confidence.item() * 100

        # Generate detailed output
        return format_prediction_output(predicted_class, confidence_value)
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return f"Error during prediction: {str(e)}"


def create_gradio_interface():
    """Create and configure Gradio interface with enhanced output"""
    try:
        interface = gr.Interface(
            fn=predict_disease,
            inputs=gr.Image(type="pil", label="Upload Plant Image"),
            outputs=gr.Markdown(label="Analysis Results"),
            title="Plant Disease Diagnostic Tool",
            description="""Upload an image of a plant leaf to:
            1. Identify potential diseases
            2. Get detailed symptoms analysis
            3. Receive treatment recommendations
            4. Learn prevention strategies""",
            examples=[],
            cache_examples=False,
            theme="default"
        )
        return interface
    except Exception as e:
        logger.error(f"Error creating Gradio interface: {str(e)}")
        raise


def main():
    try:
        if not os.path.exists(Config.model_path):
            logger.info("Training new model...")
            train_model()
            logger.info("Training complete!")

        logger.info("Starting Gradio interface...")
        interface = create_gradio_interface()
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True,
            debug=True
        )
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        raise


if __name__ == "__main__":
    main()