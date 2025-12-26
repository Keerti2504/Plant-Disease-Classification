# 🌿 Plant Disease Classification Web App

A deep learning–based web application for detecting plant leaf diseases from images using **PyTorch** and **Streamlit**.  
The app supports multiple CNN architectures and runs fully in the browser via Streamlit Cloud.

🔗 **Live App**:  
https://keerti2504-plant-disease-classification-app-nk0hhr.streamlit.app/

---

## 📌 Features

- 📷 Upload a plant leaf image (JPG / PNG)
- 🧠 Choose between **ResNet50** and **DenseNet121**
- 🌱 Classifies **15 plant disease classes**
- 📊 Displays prediction confidence
- ☁️ Deployed on **Streamlit Cloud**

---

## 🗂 Dataset

**PlantVillage Dataset**

Supported classes include:

- Pepper Bell: Healthy, Bacterial Spot  
- Potato: Healthy, Early Blight, Late Blight  
- Tomato:
  - Healthy
  - Bacterial Spot
  - Early Blight
  - Late Blight
  - Leaf Mold
  - Septoria Leaf Spot
  - Spider Mites
  - Target Spot
  - Mosaic Virus
  - Yellow Leaf Curl Virus

Total classes: **15**

---

## 🏗 Model Architecture

Two pretrained CNN backbones were fine-tuned:

| Model        | Description |
|--------------|------------|
| ResNet50     | Deep residual network with skip connections |
| DenseNet121  | Dense connectivity for feature reuse |

- Pretrained on **ImageNet**
- Final layers customized for PlantVillage classes
- Models saved as `.pth` files

---

## 🧪 Training Details

- Image size: `224 × 224`
- Optimizer: `Adam`
- Loss: `CrossEntropyLoss`
- Data augmentation:
  - Random horizontal flip
  - Random rotation
- Train / Validation / Test split: `80 / 10 / 10`

---

## 🚀 Running Locally

### 1️⃣ Clone the repository
```bash
git clone https://github.com/keerti2504/plant-disease-streamlit.git
cd plant-disease-streamlit
```
### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
### 3️⃣ Run the app
```bash
python -m streamlit run app.py
```

---
## 📁 Project Structure
```bash
plant-disease-streamlit/
├── app.py                  # Streamlit application
├── class_names.py          # Class labels
├── requirements.txt        # Dependencies
├── model/
│   ├── best_resnet50.pth
│   └── best_densenet121.pth
└── README.md
```

---
## 🌐 Deployment
The app is deployed using Streamlit Cloud directly from the GitHub repository.
