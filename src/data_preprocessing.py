import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class BrainTumorDataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.classes = ['glioma', 'meningioma', 'pituitary', 'no_tumor']
        self.img_size = (224, 224)
    
    def load_data(self):
        """Load and preprocess MRI images"""
        X, y = [], []
        
        for idx, class_name in enumerate(self.classes):
            class_path = os.path.join(self.data_dir, class_name)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, self.img_size)
                img = img / 255.0  # Normalize
                X.append(img)
                y.append(idx)
        
        X = np.array(X).reshape(-1, *self.img_size, 1)
        y = tf.keras.utils.to_categorical(y, num_classes=4)
        
        return train_test_split(X, y, test_size=0.2, random_state=42)

def visualize_samples(X_train, y_train):
    """Visualize sample MRI scans"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    class_names = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']
    
    for i, class_name in enumerate(class_names):
        idx = np.where(np.argmax(y_train, axis=1) == i)[0][0]
        axes[0, i].imshow(X_train[idx].squeeze(), cmap='gray')
        axes[0, i].set_title(f'{class_name}\n({len(np.where(np.argmax(y_train, axis=1) == i)[0])} samples)')
        axes[0, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('mri_samples.png', dpi=300, bbox_inches='tight')
    plt.show()
