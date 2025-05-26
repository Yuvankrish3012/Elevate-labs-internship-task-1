# Elevate-labs-internship-task-1

# Titanic Data Cleaning & Preprocessing 
![Titanic](https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/RMS_Titanic_3.jpg/1200px-RMS_Titanic_3.jpg)

This repository contains my solution for **Task 1: Data Cleaning & Preprocessing** of the Titanic dataset as part of my AI/ML internship with Elevate Labs.

## 📋 Project Overview
This project demonstrates fundamental data preprocessing techniques applied to the classic Titanic dataset to prepare it for machine learning models. The notebook covers:
- Missing value treatment
- Feature engineering
- Categorical encoding
- Outlier handling
- Feature scaling

## 🛠️ Technical Implementation
### Libraries Used
- Python 3.x
- Pandas (Data manipulation)
- NumPy (Numerical operations)
- Matplotlib & Seaborn (Visualizations)
- Scikit-learn (StandardScaler)

### Key Steps Performed
1. **Data Loading & Initial Exploration**
   - Checked dataset structure
   - Identified missing values

2. **Missing Value Treatment**
   - Filled missing `Age` with median
   - Filled missing `Embarked` with mode
   - Dropped `Cabin` column (too many missing values)

3. **Feature Engineering**
   - Created `FamilySize` (SibSp + Parch + 1)
   - Created `IsAlone` flag
   - Extracted `Title` from names (Mr, Mrs, etc.)

4. **Data Transformation**
   - Encoded categorical variables (Sex, Embarked, Title)
   - Scaled numerical features (Age, Fare, FamilySize)

5. **Outlier Handling**
   - Detected outliers using boxplots
   - Capped extreme `Fare` values

## 🚀 How to Use This Repository
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ElevateLabs-Internship-Task1.git
