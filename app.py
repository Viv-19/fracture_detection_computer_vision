# Required Libraries
import os
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input
from skimage import exposure

# Streamlit page configuration MUST BE FIRST COMMAND
st.set_page_config(
    page_title="Bone Fracture Detection",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration Dictionary
CONFIG = {
    'model_path': 'final_densenet_fracture_model.h5',
    'target_size': (224, 224),
    'max_file_size': 10 * 1024 * 1024,  # 10MB limit
    'last_conv_layer': 'conv5_block16_2_conv'  # For Grad-CAM
}

# Load Model with caching
@st.cache_resource
def load_fracture_model():
    try:
        model = load_model(CONFIG['model_path'])
        # Verify the model has the expected last conv layer
        model.get_layer(CONFIG['last_conv_layer'])
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_fracture_model()

def preprocess_xray(img):
    """Enhanced X-ray image preprocessing pipeline"""
    # Convert to RGB if grayscale
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # Gamma correction
    img = exposure.adjust_gamma(img, gamma=0.7)
    
    # Noise reduction
    img = cv2.GaussianBlur(img, (3, 3), 1)
    
    # CLAHE contrast enhancement
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # Edge enhancement
    blurred = cv2.GaussianBlur(img, (5, 5), 2.0)
    img = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
    
    # Edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_map = np.sqrt(sobelx**2 + sobely**2)
    edge_map = cv2.normalize(edge_map, None, 0, 255, cv2.NORM_MINMAX)
    edge_map = cv2.cvtColor(edge_map.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    img = cv2.addWeighted(img, 0.7, edge_map, 0.3, 0)
    
    return img.astype(np.float32) / 255.0

def generate_gradcam(img, model, layer_name):
    """Generate Grad-CAM heatmap for fracture localization"""
    # Create model that maps input to conv layer output + predictions
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    
    # Compute gradient of top predicted class
    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(np.expand_dims(img, axis=0))
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    
    # Gradient calculation
    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Generate heatmap
    conv_output = conv_output[0].numpy()
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        conv_output[:, :, i] *= pooled_grads[i]
    
    heatmap = np.mean(conv_output, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    
    return heatmap

def predict_fracture(image_data):
    """Process image and make prediction"""
    try:
        # Validate file size
        if image_data.size > CONFIG['max_file_size']:
            raise ValueError(f"File exceeds {CONFIG['max_file_size']//(1024*1024)}MB limit")
        
        # Read and preprocess image
        img = cv2.imdecode(np.frombuffer(image_data.read(), np.uint8), cv2.IMREAD_COLOR)
        original_img = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, CONFIG['target_size'])
        processed_img = preprocess_xray(img)
        
        # Make prediction
        img_array = np.expand_dims(processed_img, axis=0)
        fracture_prob = model.predict(img_array, verbose=0)[0][0]
        
        return {
            'original': original_img,
            'processed': processed_img,
            'probability': float(fracture_prob),
            'result': "Fracture" if fracture_prob > 0.5 else "Normal"
        }
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

# Streamlit UI
st.title("🦴 Bone Fracture Detection from X-ray Images")
st.markdown("""
    Upload an X-ray image to detect potential bone fractures.  
    The AI will analyze the image and highlight areas of concern.
""")

# File uploader in sidebar
with st.sidebar:
    st.header("Upload X-ray Image")
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['png', 'jpg', 'jpeg'],
        help="Maximum file size: 10MB"
    )

# Main content
if not model:
    st.error("Model failed to load. Please check the model file.")
elif uploaded_file:
    with st.spinner("Analyzing X-ray image..."):
        result = predict_fracture(uploaded_file)
    
    if result:
        # Display results in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(result['original'], 
                    caption="Original X-ray Image", 
                    use_column_width=True)
            
        with col2:
            st.image(result['processed'], 
                    caption="Enhanced & Preprocessed", 
                    use_column_width=True)
        
        # Diagnosis result
        st.markdown("---")
        st.subheader("Diagnosis Report")
        
        # In the Streamlit UI section where you display the heatmap (around line 170-180), replace with:

        if result['result'] == "Fracture":
            # Red alert for fracture
            st.error(f"""
                🚨 **Fracture Detected**  
                **Confidence:** {result['probability']:.2%}  
                **Note:** This requires immediate medical attention.
            """)
            
            # Generate and display heatmap with reduced size
            heatmap = generate_gradcam(
                result['processed'], 
                model, 
                CONFIG['last_conv_layer']
            )
            heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            superimposed_img = cv2.addWeighted(
                cv2.cvtColor((result['processed'] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
                0.6, heatmap_colored, 0.4, 0
            )
            
            # Display heatmap at reduced size
            st.image(superimposed_img, 
                    caption=f"Fracture Localization Heatmap (Confidence: {result['probability']:.2%})", 
                    width=400)  # Added width parameter to control display size
        else:
            # Green for normal
            st.success(f"""
                ✅ **No Fracture Detected**  
                **Confidence:** {(1-result['probability']):.2%}  
                **Note:** No signs of fracture found in this image.
            """)
        
        # Technical details
        with st.expander("Technical Details"):
            st.write(f"**Model Architecture:** DenseNet121 (Custom trained)")
            st.write(f"**Input Resolution:** {CONFIG['target_size'][0]}x{CONFIG['target_size'][1]}")
            st.write(f"**Prediction Threshold:** >50% probability for fracture detection")
            st.write(f"**Processing Time:** {result.get('processing_time', 'N/A')} seconds")
else:
    # Placeholder before upload
    st.info("👈 Please upload an X-ray image to begin analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.image("https://via.placeholder.com/500x300?text=Sample+X-ray+Image", 
                caption="Sample X-ray Image", 
                use_column_width=True)
    with col2:
        st.markdown("""
            ### Expected Input:
            - Clear X-ray images (PNG, JPG, JPEG)
            - Maximum file size: 10MB
            - Preferably frontal view of bone
            
            ### How It Works:
            1. Upload an X-ray image
            2. AI analyzes bone structures
            3. Get instant fracture detection
            4. View highlighted areas of concern
        """)

# Footer disclaimer
st.markdown("---")
st.caption("""
    **Disclaimer:** This AI tool is designed to assist medical professionals and does not replace clinical judgment.  
    Always consult with a qualified healthcare provider for medical diagnosis.
""")