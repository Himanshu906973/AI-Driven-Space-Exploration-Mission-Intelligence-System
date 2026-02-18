# 🚀 AI-Driven Space Exploration Mission Intelligence System
## Enhanced GEMSNet – Graph Neural Network Based Multi-Task Learning Framework

---

# 📌 COMPLETE PROJECT DOCUMENTATION

---

## 1️⃣ Introduction

The **AI-Driven Space Exploration Mission Intelligence System** is an advanced Graph Neural Network (GNN)-based framework designed to analyze global space mission data and predict mission sustainability.

The system uses an enhanced multi-task deep learning model called **EnhancedGEMSNet**, which captures relational dependencies between missions using graph structures instead of treating them independently.

This intelligent system predicts:

- Mission Success Probability
- Mission Duration
- Environmental Risk
- Mission Sustainability Index (MSI)
- Strategic Recommendations

This project demonstrates the application of AI and Graph Deep Learning in aerospace mission intelligence and sustainability analysis.

---

## 2️⃣ Problem Statement

Space missions depend on multiple interconnected parameters such as:

- Country
- Mission Type
- Satellite Type
- Launch Site
- Technology Used
- Budget
- Environmental Impact
- Duration
- Historical Success Rate

Traditional ML models fail to capture relationships between missions.

To overcome this limitation, missions are modeled as a graph:

- Missions → Nodes
- Shared attributes (Country, Technology, Mission Type) → Edges

This allows relational learning and improved prediction performance.

---

## 3️⃣ Proposed Solution – EnhancedGEMSNet

EnhancedGEMSNet is a Multi-Task Graph Convolutional Network consisting of:

- 3 Graph Convolutional Layers (GCN)
- Batch Normalization
- Dropout Regularization
- Self-Attention Mechanism
- Multi-Task Output Heads

### Multi-Task Predictions:

1. Success Prediction (Binary Classification)
2. Duration Prediction (Regression)
3. Environmental Risk Prediction (Regression)

---

## 4️⃣ Model Architecture

Input Features (8-dimensional encoded mission features)
↓
GCN Layer 1 (128 units)
↓
GCN Layer 2 (128 units)
↓
GCN Layer 3 (64 units)
↓
Self-Attention Layer
↓
Multi-Task Heads

Outputs:
- Success Head → Sigmoid Activation
- Duration Head → Linear Output
- Environmental Risk Head → Sigmoid Activation

---

## 5️⃣ Dataset Information

Dataset File:
Global_Space_Exploration_Dataset.csv

### Target Variable

Mission Success is defined as:

- Success Rate ≥ 70% → Successful (1)
- Success Rate < 70% → Failed (0)

---

## 6️⃣ Data Preprocessing

- Removed duplicate records
- Handled missing values
  - Mode for categorical features
  - Median for numerical features
- Label Encoding for categorical columns
- MinMax Scaling for numerical features
- Created Environmental Risk mapping
- Generated binary target variable

---

## 7️⃣ Class Imbalance Handling

To handle imbalanced data:

SMOTE (Synthetic Minority Oversampling Technique) was applied.

This balances:
- Successful Missions
- Failed Missions

---

## 8️⃣ Graph Construction

Graph Components:

- Nodes → Individual Missions
- Edges created when missions share:
  - Same Country
  - Same Technology Used
  - Same Mission Type

Graph learning enables relational intelligence.

---

## 9️⃣ Data Splitting Strategy

- 70% Training
- 15% Validation
- 15% Testing
- Stratified sampling used

---

## 🔟 Training Configuration

Loss Functions:
- Focal Loss (α = 0.75, γ = 2.0)
- Mean Squared Error (MSE) for regression tasks

Optimizer:
- AdamW (Learning Rate = 0.001, Weight Decay = 5e-4)

Learning Rate Scheduler:
- ReduceLROnPlateau
- Cosine Annealing

Additional Techniques:
- Early Stopping (Patience = 30)
- Gradient Clipping
- Dropout Regularization

---

## 1️⃣1️⃣ Evaluation Metrics

Model performance evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- AUC-ROC
- Confusion Matrix
- Balanced Accuracy

---

## 1️⃣2️⃣ Mission Sustainability Index (MSI)

Final Sustainability Formula:

MSI = Success Probability × (1 − Environmental Risk)

### Risk Categories:

- MSI ≥ 0.7 → Low Risk
- 0.5 ≤ MSI < 0.7 → Medium Risk
- 0.3 ≤ MSI < 0.5 → High Risk
- MSI < 0.3 → Critical Risk

---

## 1️⃣3️⃣ New Mission Prediction

The system includes a prediction function:

predict_new_mission_enhanced(new_mission_data)

Returns:

- Success Probability
- Predicted Duration
- Environmental Risk
- Mission Sustainability Index
- Risk Category
- Recommendation

---

## 1️⃣4️⃣ Generated Output Files

- gemsnet_complete_model.pth
- gemsnet_all_predictions.csv
- top_sustainable_missions.csv
- high_risk_missions.csv
- training_history.csv
- evaluation_metrics.csv
- Confusion Matrix Image
- ROC Curve Image
- Feature Importance Plot
- Correlation Heatmap

---

## 1️⃣5️⃣ Technologies Used

- Python
- PyTorch
- PyTorch Geometric
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Imbalanced-learn

---

## 1️⃣6️⃣ Installation & Execution

### Step 1: Install Dependencies

pip install torch torchvision torchaudio  
pip install torch-geometric  
pip install scikit-learn pandas numpy matplotlib seaborn imbalanced-learn  

### Step 2: Place Dataset

Place:
Global_Space_Exploration_Dataset.csv

inside the project directory.

### Step 3: Run Notebook

AI_Driven_Space_Exploration_Mission_Intelligence_System.ipynb

---

## 1️⃣7️⃣ Future Enhancements

- Web Application Deployment (Flask / FastAPI)
- Cloud Deployment (AWS / Azure)
- Upgrade to Graph Attention Network (GAT)
- Real-time Satellite Data Integration
- Model Explainability using SHAP

---

## 1️⃣8️⃣ Author

Himanshu Bhoi  
B.Tech – Computer Science & Engineering   

---

## 1️⃣9️⃣ License

This project is developed for academic and research purposes.
