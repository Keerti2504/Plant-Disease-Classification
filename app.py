import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from class_names import CLASS_NAMES

# --------------------------------------------------
# App config
# --------------------------------------------------
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Plant Disease Detection")
st.write("Upload a leaf image from the PlantVillage dataset")

# --------------------------------------------------
# Device (CPU only for Streamlit Cloud)
# --------------------------------------------------
device = torch.device("cpu")

# --------------------------------------------------
# Model loader
# --------------------------------------------------
@st.cache_resource
def load_model(model_name):
    if model_name == "ResNet50":
        model = models.resnet50(pretrained=False)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, len(CLASS_NAMES))
        )
        model_path = "model/resnet50.pth"

    else:  # DenseNet121
        model = models.densenet121(pretrained=False)
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, len(CLASS_NAMES))
        )
        model_path = "model/densenet121.pth"

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


# --------------------------------------------------
# Image preprocessing (MUST match training)
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------------------------------------
# UI controls
# --------------------------------------------------
model_choice = st.selectbox(
    "Select model architecture",
    ["ResNet50", "DenseNet121"]
)

model = load_model(model_choice)

uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probs, 1)

    predicted_class = CLASS_NAMES[predicted_idx.item()]
    confidence_score = confidence.item() * 100

    st.success(f"🧠 Prediction: **{predicted_class}**")
    st.info(f"📊 Confidence: **{confidence_score:.2f}%**")

    # Optional: show top-3 predictions
    st.subheader("Top 3 Predictions")
    top_probs, top_idxs = torch.topk(probs, 3)

    for i in range(3):
        st.write(
            f"{i+1}. {CLASS_NAMES[top_idxs[0][i].item()]} "
            f"— {top_probs[0][i].item()*100:.2f}%"
        )

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption(
    "Model trained on PlantVillage dataset using transfer learning "
    "(ResNet50 & DenseNet121)."
)
