
        
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
from sklearn.model_selection import train_test_split
# import cv2

# Load  data
train = pd.read_csv('histopathologic-cancer-detection/train_labels.csv') 
train_images_path = 'histopathologic-cancer-detection/train'

print(f"Training samples: {len(train)}")
print(f"Class distribution:\n{train['label'].value_counts()}")
print(f"Class proportions:\n{train['label'].value_counts(normalize=True)}")

# Check few samples
print("\nFirst 5 samples:")
print(train.head())
print("\nLast 5 samples:")
print(train.tail())

print(f"\nDataset info:")
print(f"Total labels: {len(train)}")
print(f"Unique labels: {train['label'].nunique()}")
# print(f"Image directory: {train_images_path}")

# Display sample images from each class
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

# Get samples from each class
cancer_samples = train[train['label'] == 1]['id'].head(5)
no_cancer_samples = train[train['label'] == 0]['id'].head(5)

# Display cancer samples
for i, img_id in enumerate(cancer_samples):
    img_path = os.path.join(train_images_path, f'{img_id}.tif')
    if os.path.exists(img_path):
        img = Image.open(img_path)
        axes[0, i].imshow(img)
        axes[0, i].set_title(f'Cancer')
        axes[0, i].axis('off')


# Display non-cancer samples
for i, img_id in enumerate(no_cancer_samples):
    img_path = os.path.join(train_images_path, f'{img_id}.tif')
    if os.path.exists(img_path):
        img = Image.open(img_path)
        axes[1, i].imshow(img)
        axes[1, i].set_title(f'No Cancer')
        axes[1, i].axis('off')

plt.tight_layout()
plt.show()

# Analyze pixel intensity distributions
def analyze_image_stats(sample_size=1000):
    cancer_imgs = []
    normal_imgs = []
    cancer_rgb_means = {'R': [], 'G': [], 'B': []}
    normal_rgb_means = {'R': [], 'G': [], 'B': []}
    
    # Sample images for analysis
    cancer_samples = train[train['label'] == 1].sample(min(sample_size//2, len(train[train['label'] == 1])))
    normal_samples = train[train['label'] == 0].sample(min(sample_size//2, len(train[train['label'] == 0])))
    
    for idx, row in cancer_samples.iterrows():
        img_path = os.path.join(train_images_path, f"{row['id']}.tif")
        if os.path.exists(img_path):
            img = np.array(Image.open(img_path))
            cancer_imgs.append(img.mean())
            cancer_rgb_means['R'].append(img[:,:,0].mean())
            cancer_rgb_means['G'].append(img[:,:,1].mean())
            cancer_rgb_means['B'].append(img[:,:,2].mean())
    
    for idx, row in normal_samples.iterrows():
        img_path = os.path.join(train_images_path, f"{row['id']}.tif")
        if os.path.exists(img_path):
            img = np.array(Image.open(img_path))
            normal_imgs.append(img.mean())
            normal_rgb_means['R'].append(img[:,:,0].mean())
            normal_rgb_means['G'].append(img[:,:,1].mean())
            normal_rgb_means['B'].append(img[:,:,2].mean())
    
    # Plot overall intensity distributions
    plt.figure(figsize=(15, 10))
    
    # Overall intensity
    plt.subplot(2, 2, 1)
    plt.hist(cancer_imgs, alpha=0.7, label='Cancer', bins=50, color='red')
    plt.hist(normal_imgs, alpha=0.7, label='No Cancer', bins=50, color='blue')
    plt.xlabel('Mean Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Distribution of Mean Pixel Intensities')
    
    # RGB channel analysis
    colors = ['red', 'green', 'blue']
    channels = ['R', 'G', 'B']
    
    for i, (channel, color) in enumerate(zip(channels, colors)):
        plt.subplot(2, 2, i+2)
        plt.hist(cancer_rgb_means[channel], alpha=0.7, label=f'Cancer {channel}', 
                bins=30, color=color, edgecolor='black')
        plt.hist(normal_rgb_means[channel], alpha=0.5, label=f'Normal {channel}', 
                bins=30, color='gray', edgecolor='black')
        plt.xlabel(f'{channel} Channel Mean Intensity')
        plt.ylabel('Frequency')
        plt.legend()
        plt.title(f'{channel} Channel Distribution')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'cancer_stats': {
            'mean': np.mean(cancer_imgs),
            'std': np.std(cancer_imgs),
            'rgb': {ch: np.mean(cancer_rgb_means[ch]) for ch in channels}
        },
        'normal_stats': {
            'mean': np.mean(normal_imgs),
            'std': np.std(normal_imgs),
            'rgb': {ch: np.mean(normal_rgb_means[ch]) for ch in channels}
        }
    }

# Run the analysis
stats = analyze_image_stats(sample_size=2000)
print("\nImage Statistics Summary:")
print("Cancer images - Mean intensity:", f"{stats['cancer_stats']['mean']:.2f}")
print("Normal images - Mean intensity:", f"{stats['normal_stats']['mean']:.2f}")

def check_image_quality(sample_size=5000):
    corrupted_images = []
    valid_images = []
    size_issues = []
    
    # Sample images to check
    sample_data = train.sample(min(sample_size, len(train)))
    
    print(f"Checking {len(sample_data)} images for quality issues...")
    
    for idx, row in sample_data.iterrows():
        img_path = os.path.join(train_images_path, f"{row['id']}.tif")
        
        try:
            if not os.path.exists(img_path):
                corrupted_images.append({'id': row['id'], 'issue': 'file_not_found'})
                continue
                
            img = Image.open(img_path)
            
            # Check image size
            if img.size != (96, 96):
                size_issues.append({'id': row['id'], 'size': img.size})
            
            # Check if image can be converted to array
            img_array = np.array(img)
            if img_array.shape != (96, 96, 3):
                corrupted_images.append({'id': row['id'], 'issue': 'wrong_shape', 'shape': img_array.shape})
            else:
                valid_images.append(row['id'])
                
        except Exception as e:
            corrupted_images.append({'id': row['id'], 'issue': str(e)})
    
    print(f"\nQuality Check Results:")
    print(f"✓ Valid images: {len(valid_images)}")
    print(f"✗ Corrupted images: {len(corrupted_images)}")
    print(f"⚠ Size issues: {len(size_issues)}")
    
    if corrupted_images:
        print("\nCorrupted images sample:")
        for item in corrupted_images[:5]:
            print(f"  {item}")
    
    if size_issues:
        print("\nSize issues sample:")
        for item in size_issues[:5]:
            print(f"  {item}")
    
    return corrupted_images, valid_images, size_issues

# Check for duplicate images (by hash) - optional for large datasets
def find_duplicate_images(sample_size=10000):
    import hashlib
    
    image_hashes = {}
    duplicates = []
    
    sample_data = train.sample(min(sample_size, len(train)))
    
    print(f"Checking {len(sample_data)} images for duplicates...")
    
    for idx, row in sample_data.iterrows():
        img_path = os.path.join(train_images_path, f"{row['id']}.tif")
        
        if os.path.exists(img_path):
            try:
                with open(img_path, 'rb') as f:
                    img_hash = hashlib.md5(f.read()).hexdigest()
                
                if img_hash in image_hashes:
                    duplicates.append((row['id'], image_hashes[img_hash]))
                else:
                    image_hashes[img_hash] = row['id']
            except Exception as e:
                print(f"Error processing {row['id']}: {e}")
    
    print(f"Found {len(duplicates)} duplicate pairs")
    return duplicates

# Create train/validation split
def create_train_val_split(test_size=0.2, random_state=42):
    # Stratified split to maintain class balance
    train_df, val_df = train_test_split(
        train, 
        test_size=test_size, 
        stratify=train['label'], 
        random_state=random_state
    )
    
    print(f"Training set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    
    print(f"\nTraining set class distribution:")
    print(train_df['label'].value_counts(normalize=True))
    
    print(f"\nValidation set class distribution:")
    print(val_df['label'].value_counts(normalize=True))
    
    return train_df, val_df

# Run quality checks and create splits
corrupted, valid, size_issues = check_image_quality(sample_size=5000)
train_df, val_df = create_train_val_split()
