# EARLY WARNING EYE DISEASE DETECTION USING DEEP LEARNING

## Project Overview

This project focuses on early-stage detection of **eye diseases** using deep learning models to aid preventive healthcare. The goal is aligned with the **United Nations Sustainable Development Goal 3: Good Health and Well-Being**. Our classifier detects **four eye conditions**:

- **Normal**
- **Cataract**
- **Glaucoma**
- **Retinal Disease**

We compare and evaluate multiple deep learning models — including **ResNet18**, **EfficientNetV2**, and **Swin Transformer (ViT)** — for accuracy, efficiency, and sustainability via fine tuning and pruning.


## Model Comparison: CNNs vs Transformers

We examine trade-offs between traditional CNNs and modern vision transformers:

| Model            | Accuracy      | Size (Post-Pruning) | Notes                          |
|------------------|---------------|----------------------|---------------------------------|
| ResNet18         | Good        | Lightweight       | Easily Fine Tuned + pruned       |
| EfficientNetV2   | Better      | Moderately large | Easily Fine Tuned + pruned   |
| Swin Transformer | Best        | Efficient         | Top accuracy & robust to pruning |


## 2. Setup Instructions

To run this app locally on your system, follow these steps:

### Clone and Setup Virtual Environment

#### Windows
```bash
git clone https://github.com/AkshataSingh/Early-Eye-disease-detection
git clone https://github.com/link
cd Early-warinng-Cataract-detection
python -m venv venv
venv\Scripts\activate
```

#### macOS/Linux
```bash
git clone https://github.com/AkshataSingh/Early-Eye-disease-detection
git clone https://github.com/link
cd Early-warinng-Cataract-detection
python3 -m venv venv
source venv/bin/activate
```

#### Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage Instructions

TTo reproduce the entire workflow, run the following Jupyter notebooks **in order**:

1. `1. Data_loading_and_preprocessing.ipynb`  
   ⤷ Prepares datasets and performs train/val/test splits.

2. `2. Data_Exploratory_Analysis.ipynb`  
   ⤷ Visualizes class distributions, sample images, and basic statistics.

3. `3a. AI_Modelling_ResNet18.ipynb`  
   ⤷ Trains and prunes ResNet18 for classification.

4. `3b. AI_Modelling_EfficientNetV2.ipynb`  
   ⤷ Trains and prunes EfficientNetV2-S for classification.

5. `3c. AI_Modelling_Swin_T_ViT_output.ipynb`  
   ⤷ Trains and prunes Swin Transformer V2 for classification. 

6. `4. Assessment_&_Evaluation.ipynb`  
   ⤷ Compares test accuracy, parameter count, and efficiency metrics across all models.


---

## Future Enhancements
The following improvements are recommended for future iterations:

- **Larger & Diverse Datasets:** Include multi-source datasets to improve generalizability.
- **Explainability Tools:** Integrate Grad-CAM, SHAP, or similar to visualize model decisions.
- **Edge Device Deployment:** Explore deploying compressed models on mobile or embedded hardware.
- **Hybrid Models:** Combine CNNs and Transformers to optimize both performance and efficiency.

> Favoring Swin-T V2 demonstrates that modern Transformer architectures can outperform traditional CNNs, even after aggressive pruning, aligning with the principles of sustainable and interpretable AI for healthcare.
