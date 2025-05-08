import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class MedicalDiagnosticAssistant:
    """
    A framework for developing AI-assisted medical diagnostic tools
    with a focus on cardiology and endocrinology applications.
    """
    
    def __init__(self, specialty="general"):
        """Initialize the diagnostic assistant with specialty focus."""
        self.specialty = specialty
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        
    def load_data(self, filepath):
        """
        Load patient data from CSV, Excel, or database.
        
        Parameters:
        filepath (str): Path to the data file
        
        Returns:
        DataFrame: Processed patient data
        """
        if filepath.endswith('.csv'):
            data = pd.read_csv(filepath)
        elif filepath.endswith(('.xls', '.xlsx')):
            data = pd.read_excel(filepath)
        else:
            raise ValueError("Unsupported file format. Please use CSV or Excel.")
            
        print(f"Data loaded: {data.shape[0]} patients with {data.shape[1]} features")
        return data
    
    def preprocess_data(self, data, target_column, features_to_exclude=None):
        """
        Preprocess data for modeling.
        
        Parameters:
        data (DataFrame): Patient data
        target_column (str): Name of the diagnosis column
        features_to_exclude (list): Columns to exclude from analysis
        
        Returns:
        tuple: X (features), y (target)
        """
        # Handle missing values
        print(f"Missing values before processing:\n{data.isnull().sum().sum()}")
        
        # For numerical columns, fill with median
        num_cols = data.select_dtypes(include=['float64', 'int64']).columns
        for col in num_cols:
            if col != target_column:
                data[col].fillna(data[col].median(), inplace=True)
        
        # For categorical columns, fill with mode
        cat_cols = data.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if col != target_column:
                data[col].fillna(data[col].mode()[0], inplace=True)
                
        print(f"Missing values after processing:\n{data.isnull().sum().sum()}")
        
        # Exclude non-relevant columns
        if features_to_exclude:
            data = data.drop(columns=features_to_exclude)
        
        # One-hot encode categorical variables
        data = pd.get_dummies(data, drop_first=True)
        
        # Split features and target
        if target_column in data.columns:
            y = data[target_column]
            X = data.drop(columns=[target_column])
        else:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Scale numerical features
        X = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)
        
        return X, y
    
    def train_model(self, X, y, test_size=0.2, random_state=42):
        """
        Train a diagnostic model.
        
        Parameters:
        X (DataFrame): Features
        y (Series): Target diagnosis
        test_size (float): Proportion for test set
        random_state (int): Random seed
        
        Returns:
        float: Model accuracy
        """
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Initialize and train the model
        self.model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        self.model.fit(X_train, y_train)
        
        # Calculate feature importance
        self.feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model trained with accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        return accuracy
    
    def predict_diagnosis(self, patient_data):
        """
        Predict diagnosis for new patient data.
        
        Parameters:
        patient_data (DataFrame): New patient features
        
        Returns:
        array: Predicted diagnosis
        """
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Ensure patient data has the same features as training data
        missing_cols = set(self.feature_importance['Feature']) - set(patient_data.columns)
        for col in missing_cols:
            patient_data[col] = 0
            
        # Keep only the columns used during training
        patient_data = patient_data[self.feature_importance['Feature']]
        
        # Scale the data
        patient_data = pd.DataFrame(
            self.scaler.transform(patient_data),
            columns=patient_data.columns
        )
        
        # Make prediction
        prediction = self.model.predict(patient_data)
        prediction_proba = self.model.predict_proba(patient_data)
        
        return prediction, prediction_proba
    
    def identify_key_factors(self, top_n=10):
        """
        Identify the most important diagnostic factors.
        
        Parameters:
        top_n (int): Number of top factors to show
        
        Returns:
        DataFrame: Top diagnostic factors
        """
        if self.feature_importance is None:
            raise ValueError("Feature importance not calculated. Train the model first.")
            
        top_features = self.feature_importance.head(top_n)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=top_features)
        plt.title(f'Top {top_n} Diagnostic Factors')
        plt.tight_layout()
        plt.show()
        
        return top_features
    
    def suggest_additional_tests(self, patient_data, prediction_proba, threshold=0.8):
        """
        Suggest additional tests based on prediction uncertainty.
        
        Parameters:
        patient_data (DataFrame): Patient data
        prediction_proba (array): Prediction probabilities
        threshold (float): Certainty threshold
        
        Returns:
        list: Suggested additional tests
        """
        max_proba = np.max(prediction_proba, axis=1)[0]
        
        if max_proba >= threshold:
            return []
        
        missing_important_features = []
        for feature in self.feature_importance['Feature'][:20]:  # Top 20 important features
            if feature in patient_data and pd.isna(patient_data[feature]).any():
                missing_important_features.append(feature)
        
        if self.specialty == "cardiology":
            if max_proba < 0.5:
                return ["Echocardiogram", "Stress Test", "24-hour Holter Monitor"]
            elif max_proba < threshold:
                return ["ECG", "Cardiac Biomarkers"]
        elif self.specialty == "endocrinology":
            if max_proba < 0.5:
                return ["Comprehensive Hormone Panel", "Glucose Tolerance Test"]
            elif max_proba < threshold:
                return ["Basic Metabolic Panel", "HbA1c Test"]
        
        return ["General Blood Work", "Follow-up consultation"]


# Example Usage:

def cardiac_diagnostic_example():
    """Example implementation for cardiac diagnosis"""
    # Initialize the assistant
    cardiac_assistant = MedicalDiagnosticAssistant(specialty="cardiology")
    
    # In real-world, you would load actual patient data
    # For demonstration, we'll create synthetic data
    np.random.seed(42)
    n_samples = 1000
    
    # Create synthetic cardiac patient data
    data = {
        'age': np.random.randint(30, 80, n_samples),
        'sex': np.random.choice(['M', 'F'], n_samples),
        'chest_pain_type': np.random.choice(['typical', 'atypical', 'non-anginal', 'asymptomatic'], n_samples),
        'resting_bp': np.random.randint(100, 200, n_samples),
        'cholesterol': np.random.randint(120, 300, n_samples),
        'fasting_bs': np.random.choice([0, 1], n_samples),
        'resting_ecg': np.random.choice(['normal', 'ST-T abnormality', 'LVH'], n_samples),
        'max_hr': np.random.randint(100, 200, n_samples),
        'exercise_angina': np.random.choice(['Y', 'N'], n_samples),
        'st_depression': np.random.uniform(0, 4, n_samples),
        'st_slope': np.random.choice(['upsloping', 'flat', 'downsloping'], n_samples),
        'vessels_colored': np.random.choice([0, 1, 2, 3], n_samples),
        'thalassemia': np.random.choice(['normal', 'fixed defect', 'reversible defect'], n_samples),
        'heart_disease': np.random.choice([0, 1], n_samples)
    }
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Simulate preprocessing and training
    X, y = cardiac_assistant.preprocess_data(df, 'heart_disease')
    accuracy = cardiac_assistant.train_model(X, y)
    
    # Identify key diagnostic factors
    top_features = cardiac_assistant.identify_key_factors()
    
    # Example new patient
    new_patient = pd.DataFrame({
        'age': [65],
        'sex': ['M'],
        'chest_pain_type': ['atypical'],
        'resting_bp': [160],
        'cholesterol': [250],
        'fasting_bs': [1],
        'resting_ecg': ['ST-T abnormality'],
        'max_hr': [142],
        'exercise_angina': ['Y'],
        'st_depression': [2.5],
        'st_slope': ['downsloping'],
        'vessels_colored': [1],
        'thalassemia': ['reversible defect']
    })
    
    # Preprocess new patient data
    new_patient_processed = pd.get_dummies(new_patient, drop_first=True)
    
    # Make prediction
    prediction, prediction_proba = cardiac_assistant.predict_diagnosis(new_patient_processed)
    
    # Suggest additional tests if needed
    suggested_tests = cardiac_assistant.suggest_additional_tests(
        new_patient_processed, prediction_proba, threshold=0.85
    )
    
    print(f"\nPatient Diagnosis: {'Heart Disease' if prediction[0] == 1 else 'No Heart Disease'}")
    print(f"Prediction Confidence: {np.max(prediction_proba) * 100:.2f}%")
    if suggested_tests:
        print("Suggested Additional Tests:")
        for test in suggested_tests:
            print(f"- {test}")

    return cardiac_assistant


if __name__ == "__main__":
    assistant = cardiac_diagnostic_example()