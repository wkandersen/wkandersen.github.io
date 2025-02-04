import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import numpy as np
from model import TimmModel  # Import the model architecture
import io
from data import load_data


# Initialize FastAPI app
app = FastAPI(
    title="ML Model API",  # Swagger UI title
    description="API for uploading files and making predictions using a trained model.",
    version="1.0.0",  # Version of your API
)


# Get a list of class names
data, _, class_names, _ = load_data()


# Define the checkpoint path (update the checkpoint filename)
checkpoint_path = os.path.abspath(r"python_api/best-model-epoch=04-val_loss=0.77.ckpt")

# Load the model and checkpoint
model = TimmModel.load_from_checkpoint(checkpoint_path, class_names = 23, num_features=3)  # PyTorch Lightning-specific
model.eval()  # Set the model to evaluation mode

@app.get("/")
async def root():
    return {"message": "Welcome to the FastAPI app!"}

@app.post("/predict/", summary="Predict from Image", description="Upload an image to get the model's prediction.")
async def predict(file: UploadFile = File(...)):
    """
    Upload an image, preprocess it, and make predictions using the trained model.
    """
    try:
        # Read the uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Preprocess the image (resize, normalize, convert to tensor)
        image = image.resize((224, 224))  # Resize to match the model's input size
        image_array = np.array(image).transpose(2, 0, 1) / 255.0  # Normalize and convert to CHW format
        image_tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

        # Perform inference
        with torch.no_grad():
            predictions = model(image_tensor)
            predicted_class = torch.argmax(predictions, dim=1).item()

        # Return the prediction
        return JSONResponse(content={
            "filename": file.filename,
            "predicted_class": class_names[predicted_class]
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
