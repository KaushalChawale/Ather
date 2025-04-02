
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

### 4. Run the Streamlit App (Optimized Version)

```bash
streamlit run app1.py
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
│── app.py                                        # Streamlit app
│── app1.py                                       # Streamlit app (Optimized Version) 
│── model.pkl                                     # Trained Model
│── best_image_classifier_model.keras             # Trained Model
│── requirements.txt                              # Dependencies
│── README.md                                     # Project Documentation
│── data/
│   ├── train/                                    # Training images
│   ├── validation/                               # Validation images
│   ├── test/                                     # Validation images

```

This document outlines the key strategies implemented for training the image classification model.

## Fine-Tuned ModelInformation:

* **Partial Freezing:** Initially, the base model (MobileNetV2) is partially frozen, and only the newly added top layers are trained. This allows the model to quickly adapt to the specific classification task.
* **Layer Unfreezing:** Subsequently, a subset of the later layers in the base model is unfrozen. Training continues with a significantly lower learning rate. This fine-tuning process enables the pre-trained model to further refine its learned features to better suit the dataset.

## Callbacks

* **ModelCheckpoint:** This callback is used to save the model with the best validation accuracy during training. This ensures that the optimal model is preserved, rather than simply the final model.
* **EarlyStopping:** To prevent overfitting and save computational resources, training is stopped automatically if the validation accuracy does not improve over a specified number of epochs.
* **ReduceLROnPlateau:** If the validation loss plateaus, this callback automatically reduces the learning rate. This can help the model escape local minima and find a better solution.

## Training Parameters

* **Learning Rate:** A standard learning rate is used for the initial training phase. A much smaller learning rate is employed during the fine-tuning stage.
* **Optimizer:** The Adam optimizer is explicitly defined to provide precise control over the learning rate and other optimization parameters.

## Model Saving

* **Keras Model Saving:** The `model.save()` method from Keras is utilized for model persistence. This method offers improved robustness compared to pickle for saving complex models.

## Evaluation Visualization

* **Training/Validation Plots:** The training and validation accuracy and loss are plotted to visualize the model's learning progress. These plots are crucial for:
    * Understanding the model's learning dynamics.
    * Identifying potential issues such as overfitting or underfitting.
    * Evaluating the effectiveness of the training strategies.

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
Streamlit url: https://kaushalchawale-ather-app-ywtcvl.streamlit.app/
Streamlit url (Optimized App): https://kaushalchawale-ather-app-ywtcvl.streamlit.app/
```
