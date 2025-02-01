import os
import shutil
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the class labels
class_names = ["flat", "non_flat"]

# Load the trained ResNet model
def load_model(model_path):
    resnet = models.resnet18(pretrained=False)  # No need for pretraining weights now
    num_features = resnet.fc.in_features
    resnet.fc = nn.Sequential(
        nn.Linear(num_features, 2),  # Binary classification
        nn.LogSoftmax(dim=1)
    )
    resnet.load_state_dict(torch.load(model_path, map_location=device))
    resnet = resnet.to(device)
    resnet.eval()  # Set the model to evaluation mode
    return resnet

model = load_model("resnet_classification5.0.h5")

# Define image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match ResNet input size
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize based on ImageNet stats
])

# Predict the class of a single image
def predict_image(image_path, model):
    try:
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
        input_tensor = input_tensor.to(device)

        # Perform inference
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)

        # Get class label
        predicted_class = class_names[predicted.item()]
        return predicted_class

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# Process multiple images in a folder and copy flat images to a new folder
def predict_and_copy(folder_path, model, output_folder, save_results=False, results_file="results.txt"):
    results = []

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through images in the folder
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        if os.path.isfile(image_path) and filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            predicted_class = predict_image(image_path, model)
            if predicted_class:
                results.append((filename, predicted_class))
                print(f"{filename}: {predicted_class}")

                # Copy flat images to the new folder
                if predicted_class == "flat":
                    shutil.copy(image_path, os.path.join(output_folder, filename))

    # Save results to a file if requested
    if save_results:
        with open(results_file, "w") as f:
            for filename, predicted_class in results:
                f.write(f"{filename}: {predicted_class}\n")
        print(f"Results saved to {results_file}")

    return results

# Example usage
folder_path = "D:/KP/Wabtech/ExtractedImg"  # input folder path
output_folder = "D:/KP/Wabtech/FinalResult"  #desired output folder
results = predict_and_copy(folder_path, model, output_folder, save_results=True)

"""
#multiple image testing using resnet without copying to new folder
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the class labels
class_names = ["flat", "non_flat"]


# Load the trained ResNet model
def load_model(model_path):
    resnet = models.resnet18(pretrained=False)  # No need for pretraining weights now
    num_features = resnet.fc.in_features
    resnet.fc = nn.Sequential(
        nn.Linear(num_features, 2),  # Binary classification
        nn.LogSoftmax(dim=1)
    )
    resnet.load_state_dict(torch.load(model_path, map_location=device))
    resnet = resnet.to(device)
    resnet.eval()  # Set the model to evaluation mode
    return resnet


model = load_model("best_resnet_model.h5")

# Define image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match ResNet input size
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize based on ImageNet stats
])


# Predict the class of a single image
def predict_image(image_path, model):
    try:
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
        input_tensor = input_tensor.to(device)

        # Perform inference
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)

        # Get class label
        predicted_class = class_names[predicted.item()]
        return predicted_class

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None


# Process multiple images in a folder
def predict_folder(folder_path, model, save_results=False, results_file="results.txt"):
    results = []

    # Loop through images in the folder
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        if os.path.isfile(image_path) and filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            predicted_class = predict_image(image_path, model)
            if predicted_class:
                results.append((filename, predicted_class))
                print(f"{filename}: {predicted_class}")

    # Save results to a file if requested
    if save_results:
        with open(results_file, "w") as f:
            for filename, predicted_class in results:
                f.write(f"{filename}: {predicted_class}\n")
        print(f"Results saved to {results_file}")

    return results


# Example usage
folder_path = "D:/KP/Wabtech/ExtractedImg2"  # Replace with your folder path
results = predict_folder(folder_path, model, save_results=True)
"""


"""
#single image testing using resnet

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the class labels
class_names = ["flat", "non_flat"]


# Load the trained ResNet model
def load_model(model_path):
    resnet = models.resnet18(pretrained=False)  # No need for pretraining weights now
    num_features = resnet.fc.in_features
    resnet.fc = nn.Sequential(
        nn.Linear(num_features, 2),  # Binary classification
        nn.LogSoftmax(dim=1)
    )
    resnet.load_state_dict(torch.load(model_path, map_location=device))
    resnet = resnet.to(device)
    resnet.eval()  # Set the model to evaluation mode
    return resnet


model = load_model("best_resnet_model.h5")

# Define image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match ResNet input size
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize based on ImageNet stats
])


# Predict the class of a single image
def predict_image(image_path, model):
    try:
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
        input_tensor = input_tensor.to(device)

        # Perform inference
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)

        # Get class label
        predicted_class = class_names[predicted.item()]
        return predicted_class

    except Exception as e:
        print(f"Error processing image: {e}")
        return None


# Example usage
image_path = "D:/KP/Wabtech/ExtractedImg/wheel_11.jpg"  # Replace with your image path
result = predict_image(image_path, model)
if result:
    print(f"The image is classified as: {result}")

"""
