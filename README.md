# 🎯 Bias Mitigation in Income Prediction Models
### Responsible AI Project: Algorithmic Fairness in Census Data Analysis

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live-brightgreen)](https://yourusername.github.io/bias-mitigation-income-prediction/)

## 📋 Project Overview

This project implements a comprehensive **Responsible AI** approach to income prediction using the UCI Adult/Census dataset. We analyze and mitigate algorithmic biases in Random Forest models that predict whether an individual's annual income exceeds $50,000.

### 🔍 Key Findings
- **Baseline Model Performance**: 81.02% accuracy, 68.42% F1-score, 91.65% AUC-ROC
- **Critical Biases Identified**: 
  - Gender disparity: 2.78:1 (Male vs Female)
  - Racial disparity: 2.30:1 (Asian-Pacific Islander vs Amer-Indian-Eskimo)
  - Country origin discrimination: 12.04:1 (India vs Guatemala)
- **Best Mitigation Strategy**: Group Balancing achieved 38.4% reduction in gender bias

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
pip install -r requirements.txt
```

### Installation
```bash
git clone https://github.com/adrianfulla/ResAI-Proyecto1.git
cd ResAI-Proyecto1
pip install -r requirements.txt
```

### Run Analysis
```bash
# 1. Exploratory Data Analysis
python scripts/ExpAnalisis.py

# 2. Baseline Model Training & Bias Detection
python scripts/BiasMitigationSystem.py
```

## 📁 Repository Structure
```
├── data/                          # Processed datasets
│   ├── census_income_clean.csv    # Cleaned dataset
│   ├── X_processed.csv            # Features
│   ├── y_processed.csv            # Labels
│   └── processing_metadata.json   # Data processing info
├── scripts/                       # Python analysis scripts
│   ├── ExpAnalisis.py            # Exploratory Data Analysis
│   └── BiasMitigationSystem.py   # Bias mitigation implementation
├── results/                      # Generated outputs
│   ├── baseline_model_results.json
│   ├── mitigation_results_comparison.json
│   └── visualizations/          # Generated plots
├── README.md                     # This file
└── requirements.txt              # Python dependencies
```

## 🔧 Methodology

### 1. Exploratory Data Analysis
- **Dataset**: UCI Adult/Census Income (48,842 records, 14 features)
- **Target**: Binary income classification (≤$50K vs >$50K)
- **Bias Detection**: Systematic analysis across demographic groups

### 2. Baseline Model Development
- **Algorithm**: Random Forest (200 estimators, max_depth=15)
- **Features**: 13 predictors selected based on correlation analysis
- **Performance**: High technical performance with significant algorithmic bias

### 3. Bias Mitigation Strategies
We implemented four bias mitigation techniques:

| Strategy | Approach | Gender Bias Reduction | Performance Impact |
|----------|----------|----------------------|-------------------|
| **Group Balancing** | Pre-processing data balancing | 38.4% | Minimal (-0.60% F1) |
| **Fairness Constraints** | Ensemble with equity constraints | -1.8% | Moderate (-0.94% F1) |
| **Group Calibration** | Post-processing threshold optimization | 4.3% | Positive (+1.70% F1) |
| **Adversarial Debiasing** | Protected attribute removal | 10.3% | Slight (+0.68% F1) |

## 📈 Key Results

### Bias Analysis Summary
```
📊 BASELINE MODEL BIASES:
Gender:     Male (47.3%) vs Female (14.2%) = 3.33:1 disparity
Race:       Asian-Pac-Islander (39.5%) vs Black (17.7%) = 2.23:1 disparity  
Country:    India (67.7%) vs Guatemala (0.0%) = ∞:1 disparity (complete discrimination)

🎯 BEST MITIGATION (Group Balancing):
Gender:     2.05:1 disparity (38.4% improvement)
Performance: 80.99% accuracy, 67.81% F1-score (minimal degradation)
```

### Feature Importance Rankings
1. **relationship** (22.76%) - Family structure dominant factor
2. **marital-status** (21.48%) - Marital status determinant  
3. **education-num** (13.96%) - Years of education
4. **age** (11.27%) - Work experience/maturity
5. **capital-gain** (11.12%) - Investment income

## ⚠️ Critical Findings & Recommendations

### 🚨 Deployment Risks Identified
- **Legal Risk**: Systematic discrimination violates equal opportunity laws
- **Reputational Risk**: Model perpetuates historical census biases (1994 data)
- **Operational Risk**: Intersectional biases not fully addressed

### 💡 Recommendations
1. **Implement Advanced Techniques**: Adversarial training, synthetic data generation
2. **Update Dataset**: Use post-1994 data with equitable representation

### Areas for Improvement
- Advanced adversarial debiasing implementation
- Intersectional bias analysis (gender × race × country)
- Additional fairness metrics (equalized odds, individual fairness)
- Real-time bias monitoring dashboard


## 🔗 Links

- **Dataset**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/adult)
- **University**: [Universidad del Valle de Guatemala](https://www.uvg.edu.gt/)

## 📧 Contact

**Adrian Fulladolsa Palma**  
Universidad del Valle de Guatemala  
Email: afulladolsa@uvg.edu.gt  
Student ID: 21592

---

*This project is part of the Responsible AI course at Universidad del Valle de Guatemala, emphasizing ethical considerations in machine learning deployment.*