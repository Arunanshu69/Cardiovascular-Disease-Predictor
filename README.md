# Diffusion-Based GA-PSO Optimized XGBoost Framework for Early Heart Disease Prediction

## Overview

This project implements a novel machine learning framework for early heart disease prediction that combines:
- **TabDDPM (Tabular Denoising Diffusion Probabilistic Model)** for synthetic data generation and class balancing
- **Genetic Algorithm (GA)** for global hyperparameter optimization
- **Particle Swarm Optimization (PSO)** for local hyperparameter refinement
- **XGBoost** as the final classification model

The system aims to improve prediction accuracy, AUC, recall, and generalization compared to traditional methods such as SMOTE-XGBoost, GAN-XGBoost, and standard XGBoost.

## Datasets

- **PKIOHD Dataset**: Primary dataset for training and evaluation
- **Framingham Dataset**: External validation dataset for generalization testing

## Project Structure

```
HeartDiseasePrediction/
│
├── data/
│   ├── pkiohd.csv
│   ├── framingham.csv
│
├── notebooks/
│   ├── EDA.ipynb
│
├── src/
│   ├── preprocessing/
│   │   ├── data_cleaning.py
│   │   ├── outlier_removal.py
│   │   ├── normalization.py
│   │
│   ├── diffusion/
│   │   ├── train_tabddpm.py
│   │   ├── generate_samples.py
│   │
│   ├── feature_selection/
│   │   ├── correlation_analysis.py
│   │   ├── feature_importance.py
│   │
│   ├── optimization/
│   │   ├── genetic_algorithm.py
│   │   ├── particle_swarm.py
│   │   ├── ga_pso_hybrid.py
│   │
│   ├── models/
│   │   ├── xgboost_model.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── visualization.py
│   │
│   ├── explainability/
│   │   ├── shap_analysis.py
│
├── results/
│
├── results_abalation/
|
├── app/
│   ├── streamlit_app.py
│
├── main.py
├──plot_ga_convergence.py
├──plot_pso_convergence.py
├── requirements.txt
├── README.md
```

## Pipeline Workflow

1. **Data Preprocessing**
   - Data cleaning and missing value handling
   - Outlier removal using Z-Score and IQR methods
   - Feature scaling and normalization

2. **Diffusion-Based Synthetic Data Generation**
   - Train TabDDPM on minority heart disease class
   - Generate synthetic samples to balance dataset
   - Compare distributions between real and synthetic data

3. **Feature Selection**
   - Correlation analysis using Pearson correlation
   - XGBoost feature importance ranking
   - Automatic selection of top predictive features

4. **GA Hyperparameter Optimization**
   - Optimize XGBoost hyperparameters globally
   - Population size: 50, Generations: 30
   - Tournament selection, crossover, mutation
   - Fitness function: maximize ROC-AUC

5. **PSO Refinement**
   - Use best GA solutions as initialization
   - Local refinement around GA-discovered search space
   - Parameters: n_particles=30, iterations=50
   - Objective: maximize ROC-AUC

6. **XGBoost Training**
   - Train optimized XGBoost with GA-PSO parameters
   - Cross-validation for robust evaluation

7. **Evaluation**
   - Metrics: Accuracy, Precision, Recall, F1-score, Specificity, ROC-AUC
   - Visualizations: ROC Curve, Confusion Matrix, Precision-Recall Curve

8. **External Validation**
   - Validate on Framingham dataset
   - Performance comparison and generalization report

9. **Explainable AI**
   - SHAP analysis for model interpretability
   - SHAP Summary Plot, Force Plot, Feature Importance

10. **Streamlit Dashboard**
    - Interactive web application for predictions
    - Input: Age, Sex, Blood Pressure, Cholesterol, Glucose, BMI, Smoking, Physical Activity
    - Output: Risk Score, Probability, Prediction, SHAP Explanation

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Cardiovascular-Disease-Predictor
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Run the complete pipeline:
```bash
python main.py
```

### Run the Streamlit dashboard:
```bash
streamlit run app/streamlit_app.py
```

### Run EDA notebook:
```bash
jupyter notebook notebooks/EDA.ipynb
```

## Key Features

- **Advanced Data Balancing**: Uses TabDDPM diffusion model for high-quality synthetic data generation
- **Hybrid Optimization**: Combines GA (global search) and PSO (local refinement) for optimal hyperparameters
- **Comprehensive Evaluation**: Multiple metrics and visualizations for thorough model assessment
- **External Validation**: Tests generalization on independent Framingham dataset
- **Explainability**: SHAP analysis for transparent predictions
- **Interactive Dashboard**: User-friendly Streamlit interface for real-time predictions

## Hyperparameters Optimized

- max_depth
- learning_rate
- n_estimators
- subsample
- colsample_bytree
- gamma
- min_child_weight

## Performance Metrics

The framework is evaluated on:
- Accuracy
- Precision
- Recall
- F1-score
- Specificity
- ROC-AUC

## Comparison with Traditional Methods

This framework is designed to outperform:
- SMOTE-XGBoost
- GAN-XGBoost
- Standard XGBoost

## Requirements

- Python 3.11+
- PyTorch
- XGBoost
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
- SHAP
- DEAP (for GA)
- pyswarms (for PSO)
- Streamlit

## License

[Specify your license here]

## Citation

If you use this code in your research, please cite:
```
[Add citation information]
```

## Contact

[Add contact information]
