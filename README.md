
# Ather Scooter Classifier

## Overview

This project is a **machine learning-based image classifier** that identifies whether a given scooter image belongs to an Ather scooter or not. It utilizes a **MobileNetV2-based deep learning model**, trained on scooter images, and is deployed using **Streamlit** for easy interaction.

## Features

- Upload an image via a **Streamlit web app**
- Classify whether the image is an **Ather Scooter** or **Not an Ather Scooter**
- Uses **a pre-trained deep learning model** stored in a Pickle (`.pkl`) file
- Provides a **confidence score** for the classification

## Technologies Used

- **Python**
- **TensorFlow/Keras**
- **MobileNetV2 (Pretrained Model)**
- **Streamlit (Web App Framework)**
- **Pickle (Model Serialization)**
- **OpenCV & PIL (Image Processing)**
- **NumPy**

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/KaushalChawale/Ather.git
cd Ather
```


### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App

```bash
streamlit run app.py
```

## Model Training (Optional)

If you want to retrain the model:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import pickle

# Load MobileNetV2 and add custom layers
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Save the trained model
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
```

## Usage

1. **Upload an image** in the Streamlit web interface
2. Click **'Classify Image'**
3. View **classification result** (Ather Scooter / Not an Ather Scooter)
4. See **confidence score**

## Directory Structure

```
/ather-scooter-classifier
│── app.py                # Streamlit app
│── model.pkl             # Trained Model
│── requirements.txt      # Dependencies
│── README.md             # Project Documentation
│── data/
│   ├── train/            # Training images
│   ├── validation/       # Validation images
│   ├── test/       # Validation images

```

## Future Improvements

- Improve model accuracy with more diverse training data
- Add support for **real-time webcam classification**
- Optimize for **faster inference** using TensorFlow Lite
- Deploy as a **web service (FastAPI / Flask)**

## Contributing

Feel free to **fork** this repository and submit a pull request with improvements.

## License

This project is licensed under the MIT License.

```
Streamlit url: 
```
