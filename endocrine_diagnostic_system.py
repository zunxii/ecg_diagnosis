import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
import joblib
import datetime

class EndocrineDiagnosticSystem:
    """
    A comprehensive diagnostic assistant system for endocrine disorders
    that integrates lab results, patient data, and clinical findings.
    """
    
    def __init__(self):
        """Initialize the endocrine diagnostic system"""
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.reference_ranges = self._load_reference_ranges()
        
    def _load_reference_ranges(self):
        """
        Load reference ranges for common endocrine lab tests
        Returns a dictionary with test names and their normal ranges
        """
        # These would typically be loaded from a database
        # Here we'll define some common ones
        return {
            "TSH": {"unit": "mIU/L", "low": 0.4, "high": 4.0},
            "Free T4": {"unit": "ng/dL", "low": 0.8, "high": 1.8},
            "Free T3": {"unit": "pg/mL", "low": 2.3, "high": 4.2},
            "Glucose (fasting)": {"unit": "mg/dL", "low": 70, "high": 100},
            "HbA1c": {"unit": "%", "low": 4.0, "high": 5.6},
            "Insulin": {"unit": "μIU/mL", "low": 2.6, "high": 24.9},
            "Cortisol (morning)": {"unit": "μg/dL", "low": 5, "high": 23},
            "ACTH": {"unit": "pg/mL", "low": 7.2, "high": 63.3},
            "Prolactin": {"unit": "ng/mL", "low": 2, "high": 20},
            "Testosterone (male)": {"unit": "ng/dL", "low": 280, "high": 1100},
            "Testosterone (female)": {"unit": "ng/dL", "low": 15, "high": 70},
            "Estradiol (female, follicular)": {"unit": "pg/mL", "low": 12.5, "high": 166},
            "FSH (female, follicular)": {"unit": "mIU/mL", "low": 3.5, "high": 12.5},
            "LH (female, follicular)": {"unit": "mIU/mL", "low": 2.4, "high": 12.6},
            "PTH": {"unit": "pg/mL", "low": 15, "high": 65},
            "Calcium": {"unit": "mg/dL", "low": 8.6, "high": 10.2},
            "Vitamin D (25-OH)": {"unit": "ng/mL", "low": 30, "high": 100}
        }
    
    def load_patient_data(self, filepath):
        """
        Load patient data for analysis
        
        Parameters:
        filepath (str): Path to data file (CSV, Excel)
        
        Returns:
        DataFrame: Patient data
        """
        if filepath.endswith('.csv'):
            data = pd.read_csv(filepath)
        elif filepath.endswith(('.xlsx', '.xls')):
            data = pd.read_excel(filepath)
        else:
            raise ValueError("Unsupported file format. Please use CSV or Excel.")
            
        print(f"Loaded {data.shape[0]} patient records with {data.shape[1]} features")
        return data
    
    def analyze_lab_results(self, lab_data):
        """
        Analyze lab results against reference ranges
        
        Parameters:
        lab_data (DataFrame): Lab test results with test names as columns
        
        Returns:
        DataFrame: Analysis with flags for abnormal values
        """
        analysis = lab_data.copy()
        abnormal_flags = pd.DataFrame(index=lab_data.index)
        
        for test, range_info in self.reference_ranges.items():
            if test in lab_data.columns:
                # Add flags for abnormal values
                abnormal_flags[f"{test}_flag"] = "Normal"
                abnormal_flags.loc[lab_data[test] < range_info["low"], f"{test}_flag"] = "Low"
                abnormal_flags.loc[lab_data[test] > range_info["high"], f"{test}_flag"] = "High"
                
                # Calculate percent deviation from normal range
                mid_normal = (range_info["low"] + range_info["high"]) / 2
                analysis[f"{test}_deviation"] = ((lab_data[test] - mid_normal) / mid_normal) * 100
        
        result = pd.concat([analysis, abnormal_flags], axis=1)
        
        # Count abnormalities per patient
        abnormal_count = abnormal_flags.apply(lambda x: (x != "Normal").sum(), axis=1)
        result["abnormal_test_count"] = abnormal_count
        
        return result
    
    def preprocess_data(self, data, target_column, categorical_columns=None, drop_columns=None):
        """
        Preprocess data for machine learning
        
        Parameters:
        data (DataFrame): Patient data
        target_column (str): Name of the diagnosis column
        categorical_columns (list): Categorical columns to encode
        drop_columns (list): Columns to exclude
        
        Returns:
        tuple: X (features), y (target), feature_names
        """
        df = data.copy()
        
        # Handle missing values
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            if col != target_column:
                df[col].fillna(df[col].median(), inplace=True)
        
        for col in df.select_dtypes(include=['object']).columns:
            if col != target_column:
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        # Drop specified columns
        if drop_columns:
            df = df.drop(columns=drop_columns)
        
        # Extract target variable
        if target_column in df.columns:
            y = df[target_column]
            X = df.drop(columns=[target_column])
        else:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Handle categorical features
        if categorical_columns is None:
            categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
        
        X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
        
        # Save feature names before scaling
        feature_names = X.columns.tolist()
        
        # Scale numerical features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)
        
        return X_scaled_df, y, feature_names
    
    def handle_imbalanced_data(self, X, y):
        """
        Handle imbalanced datasets using SMOTE
        
        Parameters:
        X (DataFrame): Features
        y (Series): Target
        
        Returns:
        tuple: Balanced X and y
        """
        print("Class distribution before balancing:")
        print(y.value_counts())
        
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        print("Class distribution after balancing:")
        print(pd.Series(y_resampled).value_counts())
        
        return X_resampled, y_resampled
    
    def select_features(self, X, y, n_features=None):
        """
        Select important features for the model
        
        Parameters:
        X (DataFrame): Features
        y (Series): Target
        n_features (int): Number of features to select
        
        Returns:
        DataFrame: Selected features
        """
        # Use Random Forest for feature selection
        selector = RandomForestClassifier(n_estimators=100, random_state=42)
        selector.fit(X, y)
        
        # Get feature importance
        importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': selector.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        self.feature_importance = importance
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance.head(15))
        plt.title('Top 15 Important Features')
        plt.tight_layout()
        plt.show()
        
        # Select top features if specified
        if n_features:
            selection_model = SelectFromModel(selector, threshold=-np.inf, max_features=n_features)
            selection_model.fit(X, y)
            selected_features = X.columns[selection_model.get_support()]
            return X[selected_features]
        
        return X
    
    def train_model(self, X, y, model_type='random_forest', tune_hyperparameters=False):
        """
        Train a model for endocrine disorder diagnosis
        
        Parameters:
        X (DataFrame): Features
        y (Series): Target
        model_type (str): Type of model ('random_forest' or 'gradient_boosting')
        tune_hyperparameters (bool): Whether to tune hyperparameters
        
        Returns:
        object: Trained model
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize model
        if model_type == 'random_forest':
            model = RandomForestClassifier(random_state=42)
            
            # Hyperparameter grid for tuning
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        
        elif model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(random_state=42)
            
            # Hyperparameter grid for tuning
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5],
                'min_samples_split': [2, 5]
            }
        
        else:
            raise ValueError("Unsupported model type. Use 'random_forest' or 'gradient_boosting'")
        
        # Tune hyperparameters if specified
        if tune_hyperparameters:
            print("Tuning hyperparameters...")
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        
        print("\nModel Evaluation:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        # For binary classification, plot ROC curve
        if len(np.unique(y)) == 2:
            y_prob = model.predict_proba(X_test)[:,1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                     label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.show()
        
        self.model = model
        return model
    
    def diagnose_patient(self, patient_data):
        """
        Diagnose a patient based on their data
        
        Parameters:
        patient_data (DataFrame): Patient features
        
        Returns:
        dict: Diagnosis results
        """
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Ensure patient data has the right format
        if self.feature_importance is not None:
            # Keep only features used in training
            missing_cols = set(self.feature_importance['Feature']) - set(patient_data.columns)
            for col in missing_cols:
                patient_data[col] = 0
            
            patient_data = patient_data[self.feature_importance['Feature']]
        
        # Scale the data
        patient_data_scaled = self.scaler.transform(patient_data)
        
        # Get prediction and probabilities
        prediction = self.model.predict(patient_data_scaled)
        probabilities = self.model.predict_proba(patient_data_scaled)
        
        # Get class names
        classes = self.model.classes_
        
        # Organize results
        results = {
            'diagnosis': prediction[0],
            'probabilities': {classes[i]: prob for i, prob in enumerate(probabilities[0])}
        }
        
        # Add explanation based on feature importance
        if self.feature_importance is not None:
            top_features = self.feature_importance.head(5)['Feature'].tolist()
            results['key_factors'] = top_features
        
        return results
    
    def generate_report(self, patient_data, diagnosis_results, patient_info=None):
        """
        Generate a diagnostic report
        
        Parameters:
        patient_data (DataFrame): Patient lab and clinical data
        diagnosis_results (dict): Results from diagnose_patient
        patient_info (dict): Basic patient information
        
        Returns:
        str: Diagnostic report
        """
        # Create report header
        report = []
        report.append("=" * 50)
        report.append("ENDOCRINE DISORDER DIAGNOSTIC REPORT")
        report.append("=" * 50)
        
        # Add timestamp
        now = datetime.datetime.now()
        report.append(f"Generated on: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Add patient info if provided
        if patient_info:
            report.append("PATIENT INFORMATION")
            report.append("-" * 50)
            for key, value in patient_info.items():
                report.append(f"{key}: {value}")
            report.append("")
        
        # Add diagnosis
        report.append("DIAGNOSTIC ASSESSMENT")
        report.append("-" * 50)
        report.append(f"Primary diagnosis: {diagnosis_results['diagnosis']}")
        report.append("")
        
        # Add probabilities
        report.append("Differential diagnosis probabilities:")
        for diagnosis, prob in diagnosis_results['probabilities'].items():
            report.append(f"- {diagnosis}: {prob:.2%}")
        report.append("")
        
        # Add key factors
        if 'key_factors' in diagnosis_results:
            report.append("KEY DIAGNOSTIC FACTORS")
            report.append("-" * 50)
            for i, factor in enumerate(diagnosis_results['key_factors'], 1):
                if factor in patient_data.columns:
                    value = patient_data[factor].values[0]
                    report.append(f"{i}. {factor}: {value}")
            report.append("")
        
        # Add abnormal lab values
        report.append("LABORATORY FINDINGS")
        report.append("-" * 50)
        
        abnormal_labs = []
        for test, range_info in self.reference_ranges.items():
            if test in patient_data.columns:
                value = patient_data[test].values[0]
                status = "Normal"
                
                if value < range_info["low"]:
                    status = "Low"
                    abnormal_labs.append((test, value, status))
                elif value > range_info["high"]:
                    status = "High"
                    abnormal_labs.append((test, value, status))
                
                if status != "Normal":
                    report.append(f"{test}: {value} {range_info['unit']} ({status}) " +
                                 f"[Reference: {range_info['low']}-{range_info['high']} {range_info['unit']}]")
        
        if not abnormal_labs:
            report.append("All laboratory values within normal limits.")
        report.append("")
        
        # Add recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 50)
        
        # Disease-specific recommendations (simplified example)
        diagnosis = diagnosis_results['diagnosis']
        
        if "Diabetes" in diagnosis:
            report.append("1. Glucose monitoring and HbA1c every 3 months")
            report.append("2. Consider oral hypoglycemic agents or insulin therapy")
            report.append("3. Dietary and lifestyle modifications")
            report.append("4. Screen for complications: nephropathy, retinopathy, neuropathy")
        
        elif "Hypothyroidism" in diagnosis:
            report.append("1. Thyroid hormone replacement therapy")
            report.append("2. TSH and Free T4 monitoring every 6-8 weeks until stabilized")
            report.append("3. Annual thyroid function tests once stable")
        
        elif "Hyperthyroidism" in diagnosis:
            report.append("1. Consider antithyroid medications, radioactive iodine, or surgery")
            report.append("2. Beta-blockers for symptom management if indicated")
            report.append("3. TSH, Free T4, and Free T3 monitoring every 4-6 weeks")
        
        elif "Cushing" in diagnosis:
            report.append("1. Refer to endocrinologist for specialized management")
            report.append("2. Additional testing to determine etiology (pituitary, adrenal, ectopic)")
            report.append("3. Consider specific therapy based on etiology")
        
        elif "Adrenal Insufficiency" in diagnosis:
            report.append("1. Glucocorticoid replacement therapy")
            report.append("2. Mineralocorticoid replacement if indicated")
            report.append("3. Patient education on stress dosing and emergency management")
            report.append("4. Medical alert bracelet")
        
        else:
            report.append("1. Follow-up with endocrinology for further evaluation")
            report.append("2. Consider additional specialized testing")
            report.append("3. Reassess in 3-6 months")
        
        report.append("")
        report.append("NOTE: This report is generated by an AI diagnostic assistant and")
        report.append("should be reviewed by a healthcare professional before making")
        report.append("clinical decisions.")
        report.append("=" * 50)
        
        return "\n".join(report)
    
    def save_model(self, filepath):
        """Save the trained model and associated data"""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_importance': self.feature_importance
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model and associated data"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_importance = model_data['feature_importance']
        
        print(f"Model loaded from {filepath}")


# Example Usage:

def demo_endocrine_system():
    """Demonstrate the endocrine diagnostic system with synthetic data"""
    # Create synthetic patient data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate diabetes dataset
    def generate_diabetes_data():
        data = {
            'Age': np.random.randint(20, 80, n_samples),
            'Sex': np.random.choice(['M', 'F'], n_samples),
            'BMI': np.random.normal(28, 6, n_samples),
            'Glucose (fasting)': [],
            'HbA1c': [],
            'Insulin': [],
            'Family_History_Diabetes': np.random.choice([0, 1], n_samples),
            'Diagnosis': []
        }
        
        # Generate correlated glucose, HbA1c, and insulin values
        for i in range(n_samples):
            # Diabetes patients
            if i < 350:
                data['Glucose (fasting)'].append(np.random.normal(150, 20))
                data['HbA1c'].append(np.random.normal(7.5, 1.0))
                data['Insulin'].append(np.random.normal(15, 10))
                data['Diagnosis'].append('Type 2 Diabetes')
            
            # Prediabetic patients
            elif i < 650:
                data['Glucose (fasting)'].append(np.random.normal(115, 10))
                data['HbA1c'].append(np.random.normal(6.3, 0.3))
                data['Insulin'].append(np.random.normal(20, 8))
                data['Diagnosis'].append('Prediabetes')
            
            # Normal patients
            else:
                data['Glucose (fasting)'].append(np.random.normal(85, 10))
                data['HbA1c'].append(np.random.normal(5.2, 0.3))
                data['Insulin'].append(np.random.normal(10, 5))
                data['Diagnosis'].append('Normal')
                
        # Add some comorbidity indicators
        data['Hypertension'] = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        data['Dyslipidemia'] = np.random.choice([0, 1], n_samples, p=[0.65, 0.35])
        data['Retinopathy'] = np.zeros(n_samples)
        data['Neuropathy'] = np.zeros(n_samples)
        data['Nephropathy'] = np.zeros(n_samples)
        
        # Add more complications for diabetic patients
        for i in range(350):
            if np.random.random() < 0.3:
                data['Retinopathy'][i] = 1
            if np.random.random() < 0.25:
                data['Neuropathy'][i] = 1
            if np.random.random() < 0.2:
                data['Nephropathy'][i] = 1
                
        return pd.DataFrame(data)
    
    # Create the diagnostic system
    endocrine_system = EndocrineDiagnosticSystem()
    
    # Generate and analyze diabetes data
    diabetes_data = generate_diabetes_data()
    
    # Analyze lab results
    analyzed_data = endocrine_system.analyze_lab_results(diabetes_data)
    
    # Print sample of analyzed data
    print("Sample of analyzed patient data:")
    print(analyzed_data.head())
    
    # Preprocess data for modeling
    X, y, feature_names = endocrine_system.preprocess_data(
        analyzed_data, 
        target_column='Diagnosis', 
        drop_columns=['Retinopathy', 'Neuropathy', 'Nephropathy']  # We'll consider these as outcomes
    )
    
    # Balance the dataset
    X_balanced, y_balanced = endocrine_system.handle_imbalanced_data(X, y)
    
    # Select important features
    X_selected = endocrine_system.select_features(X_balanced, y_balanced, n_features=10)
    
    # Train model
    endocrine_system.train_model(X_selected, y_balanced, tune_hyperparameters=True)
    
    # Create a test patient
    test_patient = pd.DataFrame({
        'Age': [55],
        'Sex': ['M'],
        'BMI': [32.5],
        'Glucose (fasting)': [135],
        'HbA1c': [6.7],
        'Insulin': [18],
        'Family_History_Diabetes': [1],
        'Hypertension': [1],
        'Dyslipidemia': [1]
    })
    
    # Analyze test patient lab results
    test_patient_analyzed = endocrine_system.analyze_lab_results(test_patient)
    
    # Make diagnosis
    diagnosis = endocrine_system.diagnose_patient(test_patient_analyzed)
    
    # Generate report
    patient_info = {
        'Patient ID': 'P12345',
        'Name': 'John Doe',
        'Age': 55,
        'Sex': 'Male'
    }
    
    report = endocrine_system.generate_report(test_patient_analyzed, diagnosis, patient_info)
    
    print("\n" + report)
    
    return endocrine_system

if __name__ == "__main__":
    system = demo_endocrine_system()