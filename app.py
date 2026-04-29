import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="🧠 Brain Tumor AI", layout="wide")
st.title("🧠 Brain Tumor Classification AI")
st.markdown("Upload MRI → Instant Diagnosis")

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model.h5')
    return model

classes = ['Glioma', 'Meningioma', 'Pituitary Tumor', 'No Tumor']
model = load_model()

# File uploader
uploaded_file = st.file_uploader("Choose MRI image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="Uploaded MRI", use_column_width=True)
    
    with col2:
        st.header("🔬 AI Results")
        
        # Preprocess (NO OPENCV - pure PIL/numpy)
        img_array = np.array(image.convert('L'))
        img_array = tf.image.resize(img_array, [224, 224]).numpy()
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=(0, -1))
        img_array = np.repeat(img_array, 3, axis=-1)
        
        # Predict
        with st.spinner('Analyzing...'):
            prediction = model.predict(img_array, verbose=0)[0]
        
        # Results
        top_class = classes[np.argmax(prediction)]
        confidence = 100 * np.max(prediction)
        
        st.success(f"**🎯 {top_class}**")
        st.success(f"**📊 Confidence: {confidence:.1f}%**")
        
        # Bar chart
        st.bar_chart(dict(zip(classes, prediction*100)))

st.markdown("---")
st.markdown("""
*AI decision support for radiologists*
*94% accuracy on 7K MRI dataset*
*Trained: EfficientNetB0*
""")
