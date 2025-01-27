## Multiclassification with VGG16
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import os
from sklearn.model_selection import train_test_split

    # Load and Preprocess Data

    # Constants
    IMG_HEIGHT, IMG_WIDTH = 224, 224
    BATCH_SIZE = 16
    NUM_CLASSES = 6

    # Load CSV files
    train_csv = pd.read_csv('train_multiclass/train_labels.csv')
    val_csv = pd.read_csv('val_multiclass/val_labels.csv')
    test_csv = pd.read_csv('test_multiclass/test_labels.csv')

    # Preprocessing Function
    def preprocess_data(data_csv, base_dir):
        # Remove duplicate folder prefix and join with the base directory
        data_csv['image_path'] = data_csv['image_path'].apply(
            lambda x: os.path.abspath(os.path.join(base_dir, os.path.basename(x)))
        )
        data_csv['label'] = data_csv['label'].astype(str)  # Ensure labels are strings
        return data_csv

    # Preprocess datasets
    train_data = preprocess_data(train_csv, 'train_multiclass')
    val_data = preprocess_data(val_csv, 'val_multiclass')
    test_data = preprocess_data(test_csv, 'test_multiclass')

    # Debugging: Print the corrected paths
    print(train_data.head())
    print(val_data.head())
    print(test_data.head())

    # Data Augmentation
    datagen = ImageDataGenerator(
        rescale=1.0/255.0,  # Normalize pixel values
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    val_test_datagen = ImageDataGenerator(rescale=1.0/255.0)  # Only rescale for validation and test

    # Create Data Generators
    train_generator = datagen.flow_from_dataframe(
        dataframe=train_data,
        x_col='image_path',
        y_col='label',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    val_generator = val_test_datagen.flow_from_dataframe(
        dataframe=val_data,
        x_col='image_path',
        y_col='label',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    test_generator = val_test_datagen.flow_from_dataframe(
        dataframe=test_data,
        x_col='image_path',
        y_col='label',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False  # Don't shuffle for evaluation
    )


# Define model VGG16

# Load Pre-trained VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# Freeze the base model layers
base_model.trainable = False

# Add Custom Top Layers
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')  # Output layer for 6 classes
])

# Compile the Model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Train model

steps_per_epoch = len(train_generator)
validation_steps = len(val_generator)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps
)


# Evaluate on the Test Set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Fine-Tune the Model

# Unfreeze the base model
base_model.trainable = True

# Compile with a lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Fine-tune the model
history_fine = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_steps=val_generator.samples // BATCH_SIZE
)
# Classification report
y_true = test_generator.classes
y_pred = model.predict(test_generator)
y_pred_classes = y_pred.argmax(axis=1)

print("Classification Report")
print(classification_report(y_true, y_pred_classes, target_names=class_mapping.keys()))

# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)
print("Confusion Matrix")
print(conf_matrix)
# Inspect model layers and output shapes
for layer in model.layers:
    try:
        print(f"Layer Name: {layer.name}, Output Shape: {layer.output_shape}")
    except AttributeError:
        print(f"Layer Name: {layer.name}, Output Shape: Not Available")

# Model architecture

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

def draw_vgg16_architecture():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')  # Hide axes

    # Define positions for the flowchart components
    positions = {
        "input": (0.5, 0.9),
        "vgg16": (0.5, 0.75),
        "flatten": (0.5, 0.6),
        "dense_256": (0.5, 0.45),
        "dropout": (0.5, 0.35),
        "output": (0.5, 0.2),
    }

    labels = {
        "input": "Input Image\n(224x224x3)",
        "vgg16": "VGG-16 Backbone\n(Feature Extraction, Frozen)",
        "flatten": "Flatten Layer",
        "dense_256": "Dense Layer (256 units)\nActivation: ReLU",
        "dropout": "Dropout (0.5)",
        "output": "Output Layer\n(6 Classes, Softmax)",
    }

    # Draw components as rectangles
    for key, (x, y) in positions.items():
        ax.add_patch(Rectangle((x - 0.15, y - 0.05), 0.3, 0.1, edgecolor='black', facecolor='lightblue', lw=2))
        ax.text(x, y, labels[key], ha='center', va='center', fontsize=10, wrap=True)

    # Draw arrows
    arrow_params = dict(arrowstyle='->', color='black', lw=2)
    arrow_positions = [
        ("input", "vgg16"),
        ("vgg16", "flatten"),
        ("flatten", "dense_256"),
        ("dense_256", "dropout"),
        ("dropout", "output"),
    ]

    for start, end in arrow_positions:
        x1, y1 = positions[start]
        x2, y2 = positions[end]
        ax.add_patch(FancyArrowPatch((x1, y1 - 0.05), (x2, y2 + 0.05), **arrow_params))

    # Display the diagram
    plt.title("VGG-16-Based Model Architecture", fontsize=14)
    plt.show()

# Call the function to draw the diagram
draw_vgg16_architecture()

# Grad-CAM Integration for Explainability
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model


def grad_cam(model, img_array, last_conv_layer_name, pred_index=None):
    """
    Generates Grad-CAM heatmap.
    """
    grad_model = Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap

# Visualize Grad-CAM
# Visualize Grad-CAM
def display_grad_cam(image_path, model, last_conv_layer_name, class_names):
    """
    Visualizes Grad-CAM heatmap on the image.
    """
    img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    heatmap = grad_cam(model, img_array, last_conv_layer_name)
    
    # Get predictions and determine the predicted class
    predictions = model.predict(img_array)
    predicted_index = tf.argmax(predictions[0]).numpy()
    predicted_class = class_names[predicted_index]

    # Convert PIL image to NumPy array
    img_np = np.array(img) / 255.0  # Normalize to range [0, 1]
    
    # Resize heatmap to match image size
    heatmap_resized = np.expand_dims(heatmap, axis=-1)
    heatmap_resized = tf.image.resize(heatmap_resized, (IMG_HEIGHT, IMG_WIDTH)).numpy().squeeze()
    
    # Overlay heatmap on the original image
    superimposed_img = np.uint8(255 * (0.6 * img_np + 0.4 * heatmap_resized[..., np.newaxis]))
    
    # Display results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")
    
    plt.subplot(1, 2, 2)
    plt.imshow(superimposed_img)
    plt.title(f"Grad-CAM Heatmap\nPredicted: {predicted_class}")
    
    plt.show()


# Use Grad-CAM on a sample test image
sample_test_image = test_data['image_path'].iloc[0]  # Example test image path
last_conv_layer_name = 'conv5_block3_out'  # Change if using a different architecture
class_names = ['gametocyte', 'leukocyte', 'red blood cell', 'ring', 'schizont', 'trophozoite']

display_grad_cam(sample_test_image, model, last_conv_layer_name, class_names)

#Grad-CAM

# Load the model
model = tf.keras.models.load_model("vgg16_multiclass_model.h5")

# Inspect model layers and output shapes
print("Inspecting Model: vgg16_multiclass_model.h5")
for layer in mobilenet_model.layers:
    try:
        print(f"Layer Name: {layer.name}, Output Shape: {layer.output_shape}")
    except AttributeError:
        print(f"Layer Name: {layer.name}, Output Shape: Not Available")

pip install opencv-python

# Grad-CAM Integration for Explainability
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model


def grad_cam(model, img_array, last_conv_layer_name, pred_index=None):
    """
    Generates Grad-CAM heatmap.
    """
    grad_model = Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap

# Visualize Grad-CAM
# Visualize Grad-CAM
# Visualize Grad-CAM
import cv2

def display_grad_cam(image_path, model, last_conv_layer_name, class_names):
    """
    Visualizes Grad-CAM heatmap on the image in color.
    """
    img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    heatmap = grad_cam(model, img_array, last_conv_layer_name)
    
    # Get predictions and determine the predicted class
    predictions = model.predict(img_array)
    predicted_index = tf.argmax(predictions[0]).numpy()
    predicted_class = class_names[predicted_index]

    # Convert PIL image to NumPy array
    img_np = np.array(img) / 255.0  # Normalize to range [0, 1]
    
    # Resize heatmap to match image size
    heatmap_resized = tf.image.resize(heatmap[..., np.newaxis], (IMG_HEIGHT, IMG_WIDTH)).numpy().squeeze()
    
    # Apply colormap to heatmap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB format

    # Normalize the original image for overlay
    img_np_uint8 = np.uint8(255 * img_np)

    # Overlay the heatmap on the image
    superimposed_img = cv2.addWeighted(img_np_uint8, 0.6, heatmap_colored, 0.4, 0)

    # Display results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")
    
    plt.subplot(1, 2, 2)
    plt.imshow(superimposed_img)
    plt.title(f"Grad-CAM Heatmap\nPredicted: {predicted_class}")
    
    plt.show()



# Use Grad-CAM on a sample test image
sample_test_image = test_data['image_path'].iloc[0]  # Example test image path
last_conv_layer_name = 'conv5_block3_out'  # Change if using a different architecture
class_names = ['gametocyte', 'leukocyte', 'red blood cell', 'ring', 'schizont', 'trophozoite']

display_grad_cam(sample_test_image, model, last_conv_layer_name, class_names)

# Plot performance metrics

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
# Save the trained model
model.save('vgg16_multiclass_model.h5')
print("Model saved successfully!")
