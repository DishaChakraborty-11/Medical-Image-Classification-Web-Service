import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0

def create_brain_tumor_model(input_shape=(224, 224, 1), num_classes=4):
    """Custom CNN + Transfer Learning Hybrid Model"""
    
    # EfficientNetB0 pretrained on ImageNet (grayscale to RGB conversion)
    base_model = EfficientNetB0(
        input_shape=(*input_shape[:2], 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Custom CNN Head
    inputs = layers.Input(shape=input_shape)
    
    # Convert grayscale to RGB for EfficientNet
    x = layers.Concatenate()([inputs, inputs, inputs])
    
    # Feature extraction
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    
    # Classification head
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model

def create_simple_cnn(input_shape=(224, 224, 1), num_classes=4):
    """Lightweight CNN for comparison"""
    model = tf.keras.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(256, 3, activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
