from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
import os
import io
from PIL import Image
import cv2
from fastapi.staticfiles import StaticFiles

from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

# Initialize FastAPI app
app = FastAPI()
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load trained model
model_path = os.path.join(os.path.dirname(__file__), "best_mammo_model.keras")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
model = tf.keras.models.load_model(model_path, compile=False)

# ✅ Mammogram validation
def is_valid_mammogram(image: Image.Image) -> bool:
    if image.mode != "L":
        return False
    if image.width < 200 or image.height < 200:
        return False

    np_img = np.array(image)
    border_thresh = 15
    edge_pixels = np.concatenate([
        np_img[:10, :].flatten(),
        np_img[-10:, :].flatten(),
        np_img[:, :10].flatten(),
        np_img[:, -10:].flatten()
    ])
    black_border_ratio = np.sum(edge_pixels < border_thresh) / len(edge_pixels)
    return black_border_ratio > 0.5

# Crop black borders
def crop_black(img):
    gray = np.array(img.convert("L"))
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        cropped = img.crop((x, y, x + w, y + h))
        return cropped.resize((224, 224))
    return img.resize((224, 224))

# Preprocess input for grayscale-trained model
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    img = crop_black(img)
    img = img.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=(0, -1)).astype(np.float32)
    return img_array, img.convert("RGB")

# LIME explanation with background masking
def generate_lime_explanation(model, image_array, original_image, filename):
    try:
        from skimage.segmentation import mark_boundaries
        from skimage.morphology import remove_small_objects
        from skimage.measure import label

        cropped_img = crop_black(original_image)
        image_np = np.array(cropped_img.resize((224, 224)))

        gray = np.array(cropped_img.convert("L"))
        _, tissue_mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
        tissue_mask = cv2.resize(tissue_mask, (224, 224)) // 255

        explainer = lime_image.LimeImageExplainer()

        def predict_fn(images):
            gray_images = np.expand_dims(np.mean(images, axis=-1), axis=-1)
            return model.predict(gray_images.astype(np.float32))

        explanation = explainer.explain_instance(
            image=image_np,
            classifier_fn=predict_fn,
            top_labels=1,
            hide_color=0,
            num_samples=1500
        )

        temp, mask = explanation.get_image_and_mask(
            label=explanation.top_labels[0],
            positive_only=True,
            num_features=10,
            hide_rest=False
        )

        mask = mask * tissue_mask
        cleaned_mask = remove_small_objects(label(mask), min_size=500)
        cleaned_mask = (cleaned_mask > 0).astype(np.uint8)

        outlined = mark_boundaries(image_np / 255.0, cleaned_mask, color=(1, 1, 0))

        os.makedirs("static/lime", exist_ok=True)
        base_name = os.path.splitext(os.path.basename(filename))[0]
        path = f"static/lime/{base_name}_lime.png"
        plt.imsave(path, outlined)
        plt.close()
        return path

    except Exception as e:
        print("LIME Error:", e)
        return None

# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()

        # ✅ Mammogram validation before preprocessing
        validation_image = Image.open(io.BytesIO(image_bytes)).convert("L")
        if not is_valid_mammogram(validation_image):
            return {"error": "Uploaded image does not appear to be a valid mammogram."}

        img_array, img = preprocess_image(image_bytes)
        predictions = model.predict(img_array)
        raw_score = predictions[0][0]
        confidence = None if np.isnan(raw_score) else round(float(raw_score), 4)
        predicted_class = int(raw_score > 0.5)
        label = ["Benign", "Malignant"][predicted_class]

        print(f"Prediction: {label} ({confidence})")

        lime_path = generate_lime_explanation(model, img_array, img, file.filename)

        return {
            "success": True,
            "filename": file.filename,
            "prediction": predicted_class,
            "prediction_label": label,
            "confidence": confidence,
            "lime_heatmap_url": f"/static/lime/{os.path.basename(lime_path)}" if lime_path else None
        }

    except Exception as e:
        import traceback
        print("ERROR:", traceback.format_exc())
        return {"error": str(e)}
