"""
Brain Tumor Prediction CLI & API
Usage: python src/predict.py path/to/mri_image.jpg
"""

import os
import sys
import argparse
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import json
from pathlib import Path

# Import medical insights
from medical_insights import TUMOR_INFO

# Model path
MODEL_PATH = "models/brain_tumor_classifier.h5"
CLASSES = ['Glioma', 'Meningioma', 'Pituitary Tumor', 'No Tumor']

class BrainTumorPredictor:
    def __init__(self):
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load trained model"""
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"❌ Model not found at {MODEL_PATH}\n"
                f"📥 Run: python src/train.py first!"
            )
        
        self.model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully!")
    
    def preprocess_image(self, image_path):
        """Preprocess MRI image for prediction"""
        # Load and convert to grayscale
        img = Image.open(image_path).convert('L')
        
        # Resize to model input size
        img = cv2.resize(np.array(img), (224, 224))
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        
        # Add dimensions: (1, 224, 224, 1) -> (1, 224, 224, 3) for EfficientNet
        img = np.expand_dims(img, axis=(0, -1))
        img = np.repeat(img, 3, axis=-1)
        
        return img
    
    def predict(self, image_path):
        """Make prediction and return results"""
        processed_img = self.preprocess_image(image_path)
        
        # Predict
        predictions = self.model.predict(processed_img, verbose=0)[0]
        top_class_idx = np.argmax(predictions)
        confidence = predictions[top_class_idx] * 100
        
        result = {
            'primary': CLASSES[top_class_idx],
            'confidence': float(confidence),
            'all_probabilities': {
                cls: float(prob * 100) 
                for cls, prob in zip(CLASSES, predictions)
            }
        }
        
        return result
    
    def print_medical_insights(self, tumor_type):
        """Print clinical information"""
        tumor_key = tumor_type.lower().replace(' ', '_')
        if tumor_key in TUMOR_INFO:
            info = TUMOR_INFO[tumor_key]
            print(f"\n{'='*60}")
            print(f"🏥 {tumor_type.upper()} - CLINICAL INSIGHTS")
            print(f"{'='*60}")
            print("\n📋 EARLY SYMPTOMS:")
            for i, symptom in enumerate(info['early_symptoms'], 1):
                print(f"   {i:2d}. {symptom}")
            print("\n🔬 DIAGNOSTIC METHODS:")
            for i, method in enumerate(info['diagnosis'], 1):
                print(f"   {i:2d}. {method}")
            print(f"\n📊 PROGNOSIS: {info['prognosis']}")
        else:
            print("\nℹ️  Consult medical literature for detailed information.")

def main():
    parser = argparse.ArgumentParser(description="🧠 Brain Tumor Classifier")
    parser.add_argument("image", nargs='?', help="Path to MRI image")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--medical", action="store_true", help="Show medical info")
    
    args = parser.parse_args()
    
    if not args.image:
        print("❌ Usage: python src/predict.py your_mri_image.jpg")
        print("📋 Examples:")
        print("   python src/predict.py test.jpg")
        print("   python src/predict.py test.jpg --medical")
        print("   python src/predict.py test.jpg --json")
        sys.exit(1)
    
    if not os.path.exists(args.image):
        print(f"❌ Image not found: {args.image}")
        sys.exit(1)
    
    try:
        # Initialize predictor
        predictor = BrainTumorPredictor()
        
        # Make prediction
        result = predictor.predict(args.image)
        
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("\n🧠" + "="*50)
            print("BRAIN TUMOR CLASSIFICATION RESULT")
            print("="*50)
            print(f"📸 Image: {Path(args.image).name}")
            print(f"🎯 Primary: {result['primary']}")
            print(f"📊 Confidence: {result['confidence']:.1f}%")
            print("\n📈 ALL PROBABILITIES:")
            for tumor, prob in result['all_probabilities'].items():
                marker = "🔥" if abs(prob - max(result['all_probabilities'].values())) < 1 else "  "
                print(f"   {marker} {tumor:<18}: {prob:5.1f}%")
            
            # Medical insights
            if args.medical:
                predictor.print_medical_insights(result['primary'])
        
        print("\n✅ Prediction complete!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
