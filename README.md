# Credit Card Fraud Detection with Decision Trees and SVM

This project demonstrates the detection of fraudulent credit card transactions using machine learning models such as Decision Trees and Support Vector Machines (SVM). The dataset used is highly imbalanced, with only 0.172% of transactions being fraudulent.

## Dataset

The dataset is sourced from Kaggle: [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud). It contains 284,807 transactions, with 31 features:
- Features V1 to V28 are anonymized numerical features obtained via PCA transformation.
- The `Time` feature represents the seconds elapsed between the transaction and the first transaction in the dataset.
- The `Amount` feature represents the transaction amount.
- The `Class` feature is the target variable, where:
  - `0` indicates a legitimate transaction.
  - `1` indicates a fraudulent transaction.

## Project Workflow

1. **Dataset Analysis**:
   - Visualized the distribution of the target variable (`Class`).
   - Analyzed feature correlations with the target variable.

2. **Data Preprocessing**:
   - Standardized features using `StandardScaler`.
   - Normalized the feature matrix using the $L_1$ norm.
   - Excluded the `Time` feature from modeling.

3. **Train/Test Split**:
   - Split the dataset into training and testing sets (70% train, 30% test).

4. **Model Training**:
   - Built a Decision Tree Classifier using Scikit-Learn.
   - Built a Support Vector Machine (SVM) model using Scikit-Learn.

5. **Model Evaluation**:
   - Evaluated models using the ROC-AUC score.
   - Identified the top 6 most correlated features for potential feature selection.

## Results

- **Decision Tree Classifier**:
  - Trained with class weights to handle the imbalanced dataset.
  - Evaluated using the ROC-AUC score.

- **Support Vector Machine**:
  - Configured with balanced class weights and hinge loss.
  - Evaluated using the ROC-AUC score.

## Key Insights

- The dataset is highly imbalanced, requiring careful handling during training and evaluation.
- Feature correlation analysis revealed that some features are more predictive of fraud than others.
- Both Decision Trees and SVM models can be effectively used for fraud detection, with appropriate handling of class imbalance.

## How to Run

1. Clone the repository.
2. Install the required Python libraries:
   ```bash
   pip install pandas matplotlib scikit-learn
   ```
3. Run the Jupyter Notebook to execute the analysis and train the models.

## References

- Kaggle Dataset: [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Scikit-Learn Documentation: [https://scikit-learn.org](https://scikit-learn.org)

## License

This project is for educational purposes and is licensed under the MIT License.
