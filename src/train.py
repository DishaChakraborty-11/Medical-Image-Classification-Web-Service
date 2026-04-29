import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from data_preprocessing import BrainTumorDataLoader
from model import create_brain_tumor_model

def train_model():
    # Load data
    loader = BrainTumorDataLoader('data/Training/')
    X_train, X_val, y_train, y_val = loader.load_data()
    
    # Data augmentation
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )
    
    # Create and train model
    model = create_brain_tumor_model()
    
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        validation_data=(X_val, y_val),
        epochs=50,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
    )
    
    # Save model
    model.save('models/brain_tumor_classifier.h5')
    
    # Evaluation
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_val, axis=1)
    
    print(classification_report(y_true, y_pred_classes, 
                              target_names=['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']))
    
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_true, y_pred_classes), 
                annot=True, fmt='d', cmap='Blues',
                xticklabels=['Glioma', 'Meningioma', 'Pituitary', 'No Tumor'],
                yticklabels=['Glioma', 'Meningioma', 'Pituitary', 'No Tumor'])
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return model, history

if __name__ == "__main__":
    model, history = train_model()
