# Agriculture-Land-Classification-CNN
# IBM AI Engineering – Computer Vision (CNN + ViT) Project

A completed set of computer vision labs from the **IBM AI Engineering Professional Certificate**, demonstrating end-to-end image classification workflows using **Convolutional Neural Networks (CNNs)** and **Vision Transformers (ViTs)** in both **TensorFlow/Keras** and **PyTorch**.

---

## Features
- Download and prepare a satellite image classification dataset (agriculture vs non-agriculture)  
- Compare memory-based vs generator-based data loading  
- Perform image augmentation with Keras  
- Build, train, and save a CNN classifier in **Keras**  
- Build, train, and save a CNN classifier in **PyTorch**  
- Implement and train Vision Transformers (ViT) in **Keras** and **PyTorch**  
- Evaluate models using metrics and confusion matrices  
- Run a capstone-style evaluation for CNN–ViT integration (Keras + PyTorch)  

---

## Tech Stack
- Python  
- TensorFlow / Keras  
- PyTorch + Torchvision  
- NumPy  
- Matplotlib  
- Scikit-learn  
- Pillow  
- tqdm  
- einops

## Setup and Run Instructions
# 1. Clone the repository
git clone https://github.com/yourusername/ibm-ai-engineering-project.git
cd ibm-ai-engineering-project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download + extract the dataset
# This pulls the dataset used in the IBM/Skills Network labs.
python src/data_download.py

# 4. (Optional) Run the "memory vs generator" loading comparison
python src/memory_vs_generator.py

# 5. (Optional) Preview Keras data augmentation
python src/data_augmentation_keras.py

# 6. Train CNN models
python src/train_cnn_keras.py
python src/train_cnn_pytorch.py

# 7. Train Vision Transformer models
python src/train_vit_keras.py
python src/train_vit_pytorch.py

# 8. Run capstone-style evaluation for CNN–ViT integration
# This script downloads pretrained capstone models and evaluates them.
python src/evaluate_hybrid_models.py
