import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
import os

import os
import shutil

# Define source directories
infected_dir = 'cropped_infected_cells'
uninfected_dir = 'cropped_uninfected_cells'
train_dir = 'train_uninfected'
val_dir = 'val_uninfected'

# Define class names for infected and uninfected cells
infected_classes = ['gametocyte', 'ring', 'schizont', 'trophozoite']
uninfected_classes = ['leukocyte', 'red blood cell']

# Create subdirectories for infected classes
for class_name in infected_classes:
    os.makedirs(os.path.join(infected_dir, class_name), exist_ok=True)

# Create subdirectories for uninfected classes
for class_name in uninfected_classes:
    os.makedirs(os.path.join(uninfected_dir, class_name), exist_ok=True)

# Move infected images to their respective subdirectories
for filename in os.listdir(infected_dir):
    filepath = os.path.join(infected_dir, filename)
    # Ensure it's a file
    if os.path.isfile(filepath):
        for class_name in infected_classes:
            if class_name in filename.lower():
                shutil.move(filepath, os.path.join(infected_dir, class_name, filename))

# Move uninfected images to their respective subdirectories
for filename in os.listdir(uninfected_dir):
    filepath = os.path.join(uninfected_dir, filename)
    # Ensure it's a file
    if os.path.isfile(filepath):
        for class_name in uninfected_classes:
            if class_name in filename.lower():
                shutil.move(filepath, os.path.join(uninfected_dir, class_name, filename))

## Multi-class Classification: MobileNet-based CNNs for 6 classes
#Data Preparation

import os
import shutil

source_dir = 'balanced_cells'
infected_dir = 'cropped_infected_cells'
uninfected_dir = 'cropped_uninfected_cells'

os.makedirs(infected_dir, exist_ok=True)
os.makedirs(uninfected_dir, exist_ok=True)

for filename in os.listdir(source_dir):
    if filename.endswith('.png'):
        # Prioritize uninfected first
        if 'uninfected' in filename.lower():
            shutil.copy(os.path.join(source_dir, filename), os.path.join(uninfected_dir, filename))
        elif 'infected' in filename.lower():
            shutil.copy(os.path.join(source_dir, filename), os.path.join(infected_dir, filename))

print("Files successfully separated into infected and uninfected directories.")

# Augment leukocytes to prevent class imbalance

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Input and output directory
input_dir = 'cropped_uninfected_cells/leukocyte'

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

image_count = 0
target_count = 2000 
for filename in os.listdir(input_dir):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        # Load image
        img_path = os.path.join(input_dir, filename)
        img = load_img(img_path)  # Load image as PIL object
        x = img_to_array(img)  # Convert to numpy array
        x = x.reshape((1,) + x.shape)  # Reshape for data generator

        # Generate augmented images and save them
        for batch in datagen.flow(x, batch_size=1, save_to_dir=input_dir, save_prefix='augmented_leukocyte', save_format='png'):
            image_count += 1
            if image_count >= target_count - len(os.listdir(input_dir)):
                break

print(f"Augmented images saved to {input_dir}. Total images: {len(os.listdir(input_dir))}")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import pandas as pd
from tensorflow.keras.applications import MobileNet
# MOdel

IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
NUM_CLASSES = 6
EPOCHS = 10

train_csv = pd.read_csv('train_multiclass/train_labels.csv')
val_csv = pd.read_csv('val_multiclass/val_labels.csv')
test_csv = pd.read_csv('test_multiclass/test_labels.csv')

def preprocess_data(data_csv, base_dir):
    data_csv['image_path'] = data_csv['image_path'].apply(
        lambda x: os.path.abspath(os.path.join(base_dir, os.path.basename(x)))
    )
    data_csv['label'] = data_csv['label'].astype(str)
    return data_csv

train_data = preprocess_data(train_csv, 'train_multiclass')
val_data = preprocess_data(val_csv, 'val_multiclass')
test_data = preprocess_data(test_csv, 'test_multiclass')

train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
val_test_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_dataframe(
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
    shuffle=False
)
#Fine -tuning
def create_mobilenet_model(num_classes):
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    base_model.trainable = True
    model = Sequential([
        base_model,
        Conv2D(512, (3, 3), activation='relu', padding='same', name="custom_conv_layer"),
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = create_mobilenet_model(NUM_CLASSES)

steps_per_epoch = len(train_generator)
validation_steps = len(val_generator)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps
)

# Evaluate on Test Data
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Plots 

import matplotlib.pyplot as plt

def plot_training_history(history, title):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label='Training Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_training_history(history, 'MobileNet Multiclass Model')

# Save the model
model.save("mobilenet_multiclass_model.h5")
print("Model saved as 'mobilenet_multiclass_model.h5'")
# Explicitly map class indices to labels
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class_mapping = {
    0: 'gametocyte',
    1: 'leukocyte',
    2: 'red blood cell',
    3: 'ring',
    4: 'schizont',
    5: 'trophozoite'
}

# Load the saved model
model = tf.keras.models.load_model("mobilenet_multiclass_model.h5")

IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32

test_csv = pd.read_csv("test_multiclass/test_labels.csv")
test_csv['label'] = test_csv['label'].astype(str)

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_csv,
    x_col='image_path',
    y_col='label',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

y_true = test_generator.classes
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

print("Classification Report")
print(classification_report(y_true, y_pred_classes, target_names=[class_mapping[i] for i in range(len(class_mapping))]))

conf_matrix = confusion_matrix(y_true, y_pred_classes)
print("Confusion Matrix")
print(conf_matrix)


# Model architecture flowing chart

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

def draw_mobilenet_fine_tuning_diagram():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')  # Hide axes

    # Define positions for the flowchart components
    positions = {
        "input": (0.5, 0.9),
        "mobilenet": (0.5, 0.75),
        "custom_conv": (0.5, 0.6),
        "global_avg_pooling": (0.5, 0.45),
        "dense_128": (0.5, 0.3),
        "dropout": (0.5, 0.2),
        "output": (0.5, 0.1),
    }

    labels = {
        "input": "Input Image\n(224x224x3)",
        "mobilenet": "MobileNet Backbone\n(Fine-Tuned)",
        "custom_conv": "Custom Conv Layer\n512 Filters, 3x3, ReLU",
        "global_avg_pooling": "Global Average Pooling",
        "dense_128": "Dense Layer (128 units)\nActivation: ReLU",
        "dropout": "Dropout (0.5)",
        "output": "Output Layer\n(6 Classes, Softmax)",
    }

    # Draw rectangles for each component
    for key, (x, y) in positions.items():
        ax.add_patch(Rectangle((x - 0.15, y - 0.05), 0.3, 0.1, edgecolor='black', facecolor='lightblue', lw=2))
        ax.text(x, y, labels[key], ha='center', va='center', fontsize=10, wrap=True)

    # Define and draw arrows connecting the components
    arrow_params = dict(arrowstyle='->', color='black', lw=2)
    arrow_positions = [
        ("input", "mobilenet"),
        ("mobilenet", "custom_conv"),
        ("custom_conv", "global_avg_pooling"),
        ("global_avg_pooling", "dense_128"),
        ("dense_128", "dropout"),
        ("dropout", "output"),
    ]

    for start, end in arrow_positions:
        x1, y1 = positions[start]
        x2, y2 = positions[end]
        ax.add_patch(FancyArrowPatch((x1, y1 - 0.05), (x2, y2 + 0.05), **arrow_params))

    # Add a title
    plt.title("MobileNet-Based Model Architecture with Fine-Tuning", fontsize=14)
    plt.show()

draw_mobilenet_fine_tuning_diagram()


# See mobilenet model for grad-cam later
import tensorflow as tf
model = tf.keras.models.load_model("mobilenet_multiclass_model.h5")
model.summary()

for layer in model.layers:
    print(f"Layer Name: {layer.name}, Output Shape: {getattr(layer, 'output_shape', 'Not Available')}")

