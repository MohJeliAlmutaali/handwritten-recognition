import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
import torchvision.transforms as transforms

# Load device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define model architecture (make sure it matches the saved model)
class EmnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(inplace=True))  # 32x28x28
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(2))  # 64x14x14
        self.res1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                  nn.ReLU(inplace=True))  # 64x14x14
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(inplace=True))  # 128x14x14
        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(2))  # 256x7x7
        self.res2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                  nn.ReLU(inplace=True))  # 256x7x7
        self.classifier = nn.Sequential(nn.Flatten(),
                                        nn.Linear(256*7*7, 1024),
                                        nn.ReLU(),
                                        nn.Linear(1024, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, 26))

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

# Function to load model
def load_model_pytorch(model_path):
    model = EmnistModel()
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Convert all model parameters to the same type as input (float)
    model = model.float()

    model.to(device)
    model.eval()
    return model

# Load model
model_path = './emnist_model.pth'
model = load_model_pytorch(model_path)

# Streamlit application
st.title('EMNIST Character Recognition')

# Create two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Draw Here")
    canvas_result = st_canvas(
        fill_color="#FFFFFF",  # White background
        stroke_width=10,
        stroke_color="#000000",  # Black color for drawing
        background_color="#FFFFFF",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    if canvas_result.image_data is not None:
        # Convert the canvas result to a PIL image
        image = Image.fromarray(np.uint8(canvas_result.image_data)).convert('L')

        # Save the image for debugging
        image.save('canvas_output.png')

        # Display image
        st.image(image, caption='Drawn Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Rotate the image 90 degrees clockwise
        image = image.rotate(-90)

        # Mirror the image
        image = ImageOps.mirror(image)

        # Save the processed image for debugging
        image.save('canvas_processed_output.png')

        # Resize to 28x28
        image = ImageOps.invert(image)  # Invert the image so the background is black and the drawing is white
        image = image.resize((28, 28))

        # Convert to numpy array
        x = np.array(image)

        # Normalize the image
        x = x / 255.0

        # Convert to tensor and adjust type to float
        x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float()  # Convert to tensor and adjust shape

        # Predict from model
        with torch.no_grad():
            output = model(x.to(device))
            _, predicted = torch.max(output, 1)
            confidence = F.softmax(output, dim=1)[0][predicted].item() * 100

        # Generate response
        response = {
            'prediction': chr(predicted.item() + ord('A')),  # Adjust for EMNIST class range
            'confidence': f'{confidence:.2f}'
        }

        # Display result
        st.write(f"Prediction: {response['prediction']}")
        st.write(f"Confidence: {response['confidence']}%")
