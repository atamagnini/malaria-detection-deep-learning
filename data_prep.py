# Individualize cells -save in files

import os
import json
from PIL import Image

# Load the JSON file
with open('updated_combined.json', 'r') as f:
    data = json.load(f)

images_dir = 'images'  
output_dir = 'individual_cells' 
os.makedirs(output_dir, exist_ok=True)

for entry in data:
    image_path = os.path.join(images_dir, entry['image']['new_path'])
    image = Image.open(image_path)

    # Loop through each object in the 'objects' list
    for idx, obj in enumerate(entry['objects']):
        # Get the bounding box coordinates
        bbox = obj['bounding_box']
        min_r, min_c = bbox['minimum']['r'], bbox['minimum']['c']
        max_r, max_c = bbox['maximum']['r'], bbox['maximum']['c']

        cropped_image = image.crop((min_c, min_r, max_c, max_r))

        category = obj['category']
        output_filename = f"{category}_{entry['image']['new_path']}_obj{idx + 1}.png"
        output_path = os.path.join(output_dir, output_filename)
        
        cropped_image.save(output_path)

        print(f"Saved: {output_path}")

print("All object images have been saved.")
# Clean individual_cells folder

## Remove 'difficult' cell types
import os
input_dir = 'individual_cells'

for filename in os.listdir(input_dir):
    if filename.startswith('difficult_'):
        file_path = os.path.join(input_dir, filename)
        os.remove(file_path)
        print(f"Deleted: {file_path}")

print("Deletion complete.")
## rename files
import os
input_dir = 'individual_cells'
uninfected_cell_types = ['red blood cell', 'leukocyte']

uninfected_counter = 1
infected_counter = 1

for filename in os.listdir(input_dir):
    if filename.endswith('.png'):
        parts = filename.split('_')
        cell_type = parts[0]  # The first part of the filename (before the first underscore)
        condition = 'uninfected' if cell_type in uninfected_cell_types else 'infected'
        
        if condition == 'uninfected':
            new_filename = f"{cell_type}_{condition}_{str(uninfected_counter).zfill(3)}.png"
            uninfected_counter += 1
        else:
            new_filename = f"{cell_type}_{condition}_{str(infected_counter).zfill(3)}.png"
            infected_counter += 1

        old_path = os.path.join(input_dir, filename)
        new_path = os.path.join(input_dir, new_filename)

        # Rename the file
        os.rename(old_path, new_path)

        print(f"Renamed: {old_path} -> {new_path}")

print("Renaming complete.")

## do the counts of each type of cell
import os

# Define the directory where the images are stored
input_dir = 'individual_cells'

# Initialize counters
total_images = 0
uninfected_count = 0
infected_count = 0
category_counts = {}

# Loop through all files in the directory
for filename in os.listdir(input_dir):
    # Only process files with .png extension
    if filename.endswith('.png'):
        total_images += 1
        
        # Extract the category from the filename (the part before 'infected' or 'uninfected')
        parts = filename.split('_')
        category = parts[0]

        # Check if the filename contains 'uninfected' or 'infected' to determine the category
        if 'uninfected' in filename.lower():
            uninfected_count += 1
        else:
            infected_count += 1
        
        # Update category counts
        if category not in category_counts:
            category_counts[category] = 0
        category_counts[category] += 1

# Print the results
print("Total images containing cells: 1328")
print(f"Total cells: {total_images}")
print(f"Total uninfected cells: {uninfected_count}")
print(f"Total infected cells: {infected_count}")
print("Category counts:")
for category, count in category_counts.items():
    print(f"  {category}: {count}")

# Count of infected vs uninfected cells

import matplotlib.pyplot as plt

# Data for the bar plot
categories = ['Uninfected', 'Infected']
counts = [uninfected_count, infected_count]
total_count = uninfected_count + infected_count

# Calculate the percentages
percentages = [(count / total_count) * 100 for count in counts]

# Create the bar plot
plt.figure(figsize=(8, 6))
bars = plt.bar(categories, counts, color=['green', 'red'])

# Add labels and title
plt.xlabel('Cell Type')
plt.ylabel('Count')
plt.title('Count of Infected vs Uninfected Cells')

# Display the percentage labels on top of the bars
for bar, percentage in zip(bars, percentages):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), 
             f'{percentage:.1f}%', 
             ha='center', va='bottom', fontsize=12)

# Show the plot
plt.show()


# COunt of different infected cells

import matplotlib.pyplot as plt
import os

# Define the directory where the images are stored
input_dir = 'individual_cells'

# Initialize counters for each category
gametocyte_count = 0
ring_count = 0
schizont_count = 0
trophozoite_count = 0

# Loop through all files in the directory
for filename in os.listdir(input_dir):
    # Only process files with .png extension
    if filename.endswith('.png'):
        # Check for the presence of specific categories in the filename
        if 'gametocyte' in filename.lower():
            gametocyte_count += 1
        elif 'ring' in filename.lower():
            ring_count += 1
        elif 'schizont' in filename.lower():
            schizont_count += 1
        elif 'trophozoite' in filename.lower():
            trophozoite_count += 1

# Data for the bar plot
categories = ['Gametocyte', 'Ring', 'Schizont', 'Trophozoite']
counts = [gametocyte_count, ring_count, schizont_count, trophozoite_count]
total_count = sum(counts)

# Calculate the percentages
percentages = [(count / total_count) * 100 for count in counts]

# Create the bar plot
plt.figure(figsize=(8, 6))
bars = plt.bar(categories, counts, color=['orange', 'blue', 'green', 'purple'])

# Add labels and title
plt.xlabel('Cell Type')
plt.ylabel('Count')
plt.title('Count of Different Infected Cells')

# Display the percentage labels on top of the bars
for bar, percentage in zip(bars, percentages):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
             f'{percentage:.1f}%', 
             ha='center', va='bottom', fontsize=12)

# Show the plot
plt.tight_layout()
plt.show()

# Downsampling imbalanced data

import os
import shutil

source_folder = 'individual_cells'
destination_folder = 'balanced_cells'

os.makedirs(destination_folder, exist_ok=True)

red_blood_cell_images = []

for image_name in os.listdir(source_folder):
    if 'red blood cell' in image_name:
        red_blood_cell_images.append(image_name)

selected_images = red_blood_cell_images[:2349]

for image in selected_images:
    source_path = os.path.join(source_folder, image)
    destination_path = os.path.join(destination_folder, image)
    shutil.copy(source_path, destination_path)

for image_name in os.listdir(source_folder):
    if 'red blood cell' not in image_name:
        source_path = os.path.join(source_folder, image_name)
        destination_path = os.path.join(destination_folder, image_name)
        shutil.copy(source_path, destination_path)

print("Filtered images saved in:", destination_folder)

## do the counts of each type of cell
import os
input_dir = 'balanced_cells'

total_images = 0
uninfected_count = 0
infected_count = 0
category_counts = {}

for filename in os.listdir(input_dir):
    if filename.endswith('.png'):
        total_images += 1
        
        parts = filename.split('_')
        category = parts[0]

        if 'uninfected' in filename.lower():
            uninfected_count += 1
        else:
            infected_count += 1
        
        # Update category counts
        if category not in category_counts:
            category_counts[category] = 0
        category_counts[category] += 1

# Print the results
print(f"Total cells: {total_images}")
print(f"Total uninfected cells: {uninfected_count}")
print(f"Total infected cells: {infected_count}")
print("Category counts:")
for category, count in category_counts.items():
    print(f"  {category}: {count}")

# Count of infected vs uninfected cells

input_dir = 'balanced_cells'
import matplotlib.pyplot as plt

categories = ['Uninfected', 'Infected']
counts = [uninfected_count, infected_count]
total_count = uninfected_count + infected_count

percentages = [(count / total_count) * 100 for count in counts]

plt.figure(figsize=(8, 6))
bars = plt.bar(categories, counts, color=['green', 'red'])

plt.xlabel('Cell Type')
plt.ylabel('Count')
plt.title('Count of Infected vs Uninfected Cells')

for bar, percentage in zip(bars, percentages):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), 
             f'{percentage:.1f}%', 
             ha='center', va='bottom', fontsize=12)

# Show the plot
plt.show()


# COunt of different infected cells

import matplotlib.pyplot as plt
import os

# Define the directory where the images are stored
input_dir = 'individual_cells'

# Initialize counters for each category
gametocyte_count = 0
ring_count = 0
schizont_count = 0
trophozoite_count = 0

# Loop through all files in the directory
for filename in os.listdir(input_dir):
    # Only process files with .png extension
    if filename.endswith('.png'):
        # Check for the presence of specific categories in the filename
        if 'gametocyte' in filename.lower():
            gametocyte_count += 1
        elif 'ring' in filename.lower():
            ring_count += 1
        elif 'schizont' in filename.lower():
            schizont_count += 1
        elif 'trophozoite' in filename.lower():
            trophozoite_count += 1

# Data for the bar plot
categories = ['Gametocyte', 'Ring', 'Schizont', 'Trophozoite']
counts = [gametocyte_count, ring_count, schizont_count, trophozoite_count]
total_count = sum(counts)

# Calculate the percentages
percentages = [(count / total_count) * 100 for count in counts]

# Create the bar plot
plt.figure(figsize=(8, 6))
bars = plt.bar(categories, counts, color=['orange', 'blue', 'green', 'purple'])

# Add labels and title
plt.xlabel('Cell Type')
plt.ylabel('Count')
plt.title('Count of Different Infected Cells')

# Display the percentage labels on top of the bars
for bar, percentage in zip(bars, percentages):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
             f'{percentage:.1f}%', 
             ha='center', va='bottom', fontsize=12)

# Show the plot
plt.tight_layout()
plt.show()

# Count of all types of cells

import matplotlib.pyplot as plt
import os

input_dir = 'balanced_cells'

# Initialize counters for each category
gametocyte_count = 0
leukocyte_count = 0
red_blood_cell_count = 0
ring_count = 0
schizont_count = 0
trophozoite_count = 0

# Loop through all files in the directory
for filename in os.listdir(input_dir):
    # Only process files with .png extension
    if filename.endswith('.png'):
        # Check for the presence of specific categories in the filename
        if 'gametocyte' in filename.lower():
            gametocyte_count += 1
        elif 'leukocyte' in filename.lower():
            leukocyte_count += 1
        elif 'red blood cell' in filename.lower() or 'rbc' in filename.lower():
            red_blood_cell_count += 1
        elif 'ring' in filename.lower():
            ring_count += 1
        elif 'schizont' in filename.lower():
            schizont_count += 1
        elif 'trophozoite' in filename.lower():
            trophozoite_count += 1

# Data for the bar plot
categories = ['Gametocyte', 'Leukocyte', 'Red Blood Cell', 'Ring', 'Schizont', 'Trophozoite']
counts = [gametocyte_count, leukocyte_count, red_blood_cell_count, ring_count, schizont_count, trophozoite_count]
total_count = sum(counts)

# Calculate the percentages
percentages = [(count / total_count) * 100 for count in counts]

# Create the bar plot
plt.figure(figsize=(10, 6))
bars = plt.bar(categories, counts, color=['orange', 'cyan', 'red', 'blue', 'green', 'purple'])

# Add labels and title
plt.xlabel('Cell Type')
plt.ylabel('Count')
plt.title('Count of All Types of Cells')

# Display the percentage labels on top of the bars
for bar, percentage in zip(bars, percentages):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
             f'{percentage:.1f}%', 
             ha='center', va='bottom', fontsize=12)

# Show the plot
plt.tight_layout()
plt.show()

# Cell types

import matplotlib.pyplot as plt
from PIL import Image
import os

uninfected_cells = [
    'balanced_cells/red blood cell_uninfected_10210.png',
    'balanced_cells/leukocyte_uninfected_091.png'
]
infected_cells = [
    'balanced_cells/gametocyte_infected_040.png',
    'balanced_cells/ring_infected_391.png',
    'balanced_cells/schizont_infected_766.png',
    'balanced_cells/trophozoite_infected_1325.png'
]

uninfected_labels = ['Uninfected cell (Red Blood Cell)', 'Uninfected cell (Leukocyte)']
infected_labels = ['Infected cell (Gametocyte)', 'Infected cell (Ring)', 
                   'Infected cell (Schizont)', 'Infected cell (Trophozoite)']

def show_images_with_labels(image_paths, labels, rows, cols):
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()  # Flatten to easily index
    
    for i, (img_path, label) in enumerate(zip(image_paths, labels)):
        img = Image.open(img_path)
        
        axes[i].imshow(img)
        axes[i].set_title(label, fontsize=10)
        axes[i].axis('off') 
    
    for j in range(len(image_paths), len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()

show_images_with_labels(uninfected_cells, uninfected_labels, 1, len(uninfected_cells))

show_images_with_labels(infected_cells, infected_labels, 1, len(infected_cells))

# Splitting images in folders

import os
import shutil
import random

# Define the source directory and the target directories
source_dir = 'balanced_cells'
train_dir = 'train_multiclass'
test_dir = 'test_multiclass'
val_dir = 'val_multiclass'

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# List all PNG images in the source directory
images = [f for f in os.listdir(source_dir) if f.endswith('.png')]

# Shuffle the list of images
random.shuffle(images)

# Calculate the number of images for each split
total_images = len(images)
train_split = int(total_images * 0.7)
test_split = int(total_images * 0.2)
val_split = total_images - train_split - test_split

# Function to copy images to the appropriate folder
def copy_images_to_folder(image_list, destination_folder):
    for image in image_list:
        src_path = os.path.join(source_dir, image)
        dest_path = os.path.join(destination_folder, image)
        shutil.copy(src_path, dest_path)

# Split the images into train, test, and validation sets
train_images = images[:train_split]
test_images = images[train_split:train_split+test_split]
val_images = images[train_split+test_split:]

# Copy the images to the respective folders
copy_images_to_folder(train_images, train_dir)
copy_images_to_folder(test_images, test_dir)
copy_images_to_folder(val_images, val_dir)

print(f"Total images: {total_images}")
print(f"Training images: {len(train_images)}")
print(f"Testing images: {len(test_images)}")
print(f"Validation images: {len(val_images)}")

# CSV labels for multiclass classification

import os
import csv

# Define the directories where the images are located
train_dir = 'train_multiclass'
test_dir = 'test_multiclass'
val_dir = 'val_multiclass'

# Define the mapping of class names to labels
class_mapping = {
    'gametocyte': 0,
    'leukocyte': 1,
    'red blood cell': 2,
    'ring': 3,
    'schizont': 4,
    'trophozoite': 5
}

# Function to create CSV for each folder
def create_csv_for_folder(folder_path, csv_filename):
    # List to store image paths and labels
    image_data = []
    
    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            # Define the full image path
            img_path = os.path.join(folder_path, filename)
            
            # Determine the label based on the filename
            label = None
            for class_name, class_label in class_mapping.items():
                if class_name in filename.lower():
                    label = class_label
                    break
            
            # If no class is found, skip the file (optional error handling)
            if label is None:
                print(f"Warning: No class found for file {filename}. Skipping...")
                continue
            
            # Append the image path and label to the image_data list
            image_data.append([img_path, label])
    
    # Write the data to a CSV file
    csv_path = os.path.join(folder_path, csv_filename)
    with open(csv_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_path', 'label'])  # Write header
        writer.writerows(image_data)  # Write the image paths and labels

# Create CSVs for the train, test, and val folders
create_csv_for_folder(train_dir, 'train_labels.csv')
create_csv_for_folder(test_dir, 'test_labels.csv')
create_csv_for_folder(val_dir, 'val_labels.csv')

print("CSV files created for train_multiclass, test_multiclass, and val_multiclass.")