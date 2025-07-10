import os
import torch
import torch.nn as nn
import cv2
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
from torchvision import transforms
from PIL import Image
from datetime import datetime

# Define paths
MODEL_PATH = "D:\\mini project\\best_discriminator.pth"
OUTPUT_FOLDER = "D:\\mini project\\outputs"
UPLOAD_FOLDER = "D:\\mini project\\uploads"

# Ensure output and upload folders exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Define SelfAttention layer
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        batch_size, C, H, W = x.size()
        q = self.query(x).view(batch_size, -1, H * W)
        k = self.key(x).view(batch_size, -1, H * W)
        v = self.value(x).view(batch_size, -1, H * W)
        attention = torch.softmax(torch.bmm(q.permute(0, 2, 1), k), dim=-1)
        attention = self.dropout(attention)
        out = torch.bmm(v, attention.permute(0, 2, 1)).view(batch_size, C, H, W)
        return self.gamma * out + x

# Define the Discriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(3, 64, 4, 2, 1, bias=False)),    # 64x32x32
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(64, 128, 4, 2, 1, bias=False)),  # 128x16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            SelfAttention(128),
            nn.utils.spectral_norm(nn.Conv2d(128, 256, 4, 2, 1, bias=False)), # 256x8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(256, 512, 4, 2, 1, bias=False)), # 512x4x4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(512, 1, 4, 1, 0, bias=False)),   # 1x1x1
            nn.Flatten()  # Output: batch_size x 1
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        assert x.shape[1:] == (3, 64, 64), f"Expected input (batch, 3, 64, 64), got {x.shape}"
        return self.model(x)

# Load the pre-trained model with error handling
print("Loading model...")
model = Discriminator()
try:
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully!")
except RuntimeError as e:
    print(f"Error loading model: {e}")
    print("Attempting to load with strict=False as a fallback...")
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("Model loaded with strict=False (some weights may be missing or ignored).")

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Function to process the image and add a border based on classification
def process_image(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(input_tensor).item()

    label = "Real" if output < 0.5 else "Fake"
    color = (0, 255, 0) if label == "Real" else (0, 0, 255)  # Green for real, Red for fake

    # Convert image to OpenCV format
    image_cv = cv2.imread(image_path)
    image_cv = cv2.resize(image_cv, (256, 256))
    border_size = 10
    image_cv = cv2.copyMakeBorder(image_cv, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=color)

    # ✅ Add current date and time on the image
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    text_color = (255, 255, 255)
    text_position = (10, 20)

    cv2.putText(image_cv, timestamp, text_position, font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    # Save the image
    output_filename = os.path.join(OUTPUT_FOLDER, os.path.basename(image_path))
    cv2.imwrite(output_filename, image_cv)

    return output_filename, label

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    processed_path, label = process_image(filepath)
    
    return render_template('result.html', input_image=filename, output_image=os.path.basename(processed_path), label=label)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/outputs/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

# Run the Flask App
if __name__ == "__main__":
    print("Running Flask app...")
    app.run(debug=True)