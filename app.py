# Uses model.pkl

# ✅ 1. Import necessary libraries
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import pickle
import os
import cv2 # OpenCV for resizing, as used in the original concepts
st.set_page_config(page_title="Ather Scooter Classifier", page_icon="🛵")

# ✅ 2. Function to load the trained model (with caching)
# IMPORTANT: Using pickle for Keras models is generally not recommended.
# Use model.save('my_model.h5') and tf.keras.models.load_model() in the future.
@st.cache_resource # Cache the model loading for efficiency
def load_keras_model(model_path='model.pkl'):
    """Loads the pickled Keras model."""
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found at '{model_path}'. Please ensure it's in the same directory as the script.")
        st.stop() # Stop execution if model file is missing
    try:
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
        # Check if it's a Keras model (basic check)
        if not isinstance(model, tf.keras.Model):
            st.error(f"❌ The file '{model_path}' does not contain a valid Keras Model.")
            st.stop()
        print("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

# ✅ 3. Function to preprocess the uploaded image
def preprocess_image(image_pil):
    """Preprocesses the PIL image for the model."""
    try:
        # Convert PIL image to OpenCV format (NumPy array)
        image_np = np.array(image_pil.convert('RGB')) # Ensure 3 channels (RGB)

        # Resize image to the target size expected by the model
        target_size = (224, 224)
        image_resized = cv2.resize(image_np, target_size)

        # Rescale pixel values to [0, 1] (matching ImageDataGenerator)
        image_rescaled = image_resized / 255.0

        # Expand dimensions to create a batch (1, height, width, channels)
        image_expanded = np.expand_dims(image_rescaled, axis=0)
        return image_expanded
    except Exception as e:
        st.error(f"❌ Error preprocessing image: {e}")
        return None

# --- Streamlit App ---

# ✅ 4. Set App Title
st.title("🛴 Ather Scooter Classifier")
st.write("Upload an image of a scooter, and the app will predict if it's an Ather scooter or not.")

# ✅ 5. Load the Model
model = load_keras_model('model.pkl') # Assumes model.pkl is in the same directory

# ✅ 6. File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# ✅ 7. Process and Predict if a file is uploaded
if uploaded_file is not None:
    # Display the uploaded image
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)
        st.write("") # Add a little space
        st.write("Classifying...")

        # Preprocess the image
        processed_image = preprocess_image(image)

        if processed_image is not None:
            # Make prediction
            prediction = model.predict(processed_image)
            score = prediction[0][0] # Get the single probability value

            # --- Determine Class Label ---
            # This assumes class 0 = 'Not Ather', class 1 = 'Ather' based on common practice
            # where the positive class gets index 1.
            # You might need to ADJUST this based on how ImageDataGenerator assigned labels
            # (often alphabetical: 'Ather' might be 0, 'Not_Ather' might be 1).
            # Check train_generator.class_indices in your original notebook if unsure.
            threshold = 0.5
            if score < threshold:
                st.success(f"Prediction: **Ather Scooter** (Confidence: {score:.2f})")
            else:
                st.info(f"Prediction: **Not Ather Scooter** (Confidence: {1-score:.2f})")

            # Optional: Display raw score
            # st.write(f"Raw Prediction Score: {score:.4f}")

    except Exception as e:
        st.error(f"❌ An error occurred while processing the image: {e}")

else:
    st.info("Please upload an image file.")

st.sidebar.header("About")
st.sidebar.info("This app uses a pre-trained MobileNetV2 model (fine-tuned) to classify images of scooters as 'Ather' or 'Not Ather'.")
st.sidebar.warning("Note: Model loaded using 'pickle'. Standard practice recommends `model.save()` and `tf.keras.models.load_model()`.")