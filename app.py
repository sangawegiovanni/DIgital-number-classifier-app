from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import io
from classifierModel import DIGITCLASSIFIER
# -----------------------------
model = DIGITCLASSIFIER()
state_dict=torch.load("DIGIT-CLASSIFIER-PARAMS.pth",weights_only=True,map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# -----------------------------
# 3. Initialize FastAPI

app = FastAPI()

# -----------------------------
# 4. Image transform
# -----------------------------

transform=transforms.Compose([
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5),(0.5))
])
# -----------------------------
# 5. Predict endpoint
# -----------------------------
@app.post("/predict")
def predict(file: UploadFile = File(...)):
    # Read image bytes
    image_bytes = file.file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    
    # Preprocess image
    img_tensor = transform(image).unsqueeze(0)  # add batch dimension
    
    # Make prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        predicted_digit = torch.argmax(outputs, dim=1).item()
    
    # Return as simple JSON
    return {"prediction": predicted_digit}

# -----------------------------
# 6. Simple home endpoint
# -----------------------------
@app.get("/")
def home():
    return {"message": "Digit Classifier API is running!"}
