from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import tensorflow as tf
import numpy as np
import io
import nibabel as nib
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

app = FastAPI()

# Load trained model
MODEL_PATH = "/home/ctp/brain/Brain_segmentation_UNET_RESNET.h5"
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_nii(nii_path):
    nii_image = nib.load(nii_path)
    mri_array = nii_image.get_fdata()

    # Resize MRI to (128, 128, 128)
    
    mri_resized = tf.image.resize_with_pad(mri_array, 128, 128).numpy()

    mri_resized = mri_resized[..., np.newaxis]  # Add channel dimension
    return np.expand_dims(mri_resized, axis=0)  # Add batch dimension

@app.post("/visualize")
async def visualize(file: UploadFile = File(...)):
    # Read uploaded file
    contents = await file.read()
    with open("temp.nii.gz", "wb") as f:
        f.write(contents)

    # Preprocess and predict
    mri_input = preprocess_nii("temp.nii.gz")
    # pred_mask = model.predict(mri_input)[0, :, :, pred_mask.shape[2] // 2, 0]  # Middle slice

    pred_mask = model.predict(mri_input)
    middle_slice = pred_mask.shape[2] // 2  # Get the middle slice using the predicted mask
    pred_mask = pred_mask[0, :, :, middle_slice, 0] 

    # Plot and return image
    buf = io.BytesIO()
    plt.imshow(pred_mask, cmap="gray")
    plt.axis("off")
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")
