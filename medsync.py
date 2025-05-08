import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input, concatenate
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
import joblib
import datetime
import json
import logging
import time
from flask import Flask, request, jsonify

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("medsync.log"), logging.StreamHandler()]
)
logger = logging.getLogger("MedSync")

class MedSyncDiagnosticSystem:
    """
    MedSync: An integrated diagnostic platform that combines cardiology and endocrinology
    diagnostic capabilities into a unified medical decision support system.
    """
    
    def __init__(self):
        """Initialize the MedSync diagnostic system"""
        self.ecg_analyzer = ECGAnalysisModule()
        self.endocrine_analyzer = EndocrineAnalysisModule()
        self.integrated_analyzer = IntegratedAnalysisModule()
        self.version = "1.0.0"
        self.models_loaded = False
        logger.info("MedSync Diagnostic System initialized")
    
    def load_models(self, ecg_model_path=None, endocrine_model_path=None, integrated_model_path=None):
        """Load pretrained models for all analysis modules"""
        try:
            if ecg_model_path:
                self.ecg_analyzer.load_model(ecg_model_path)
            
            if endocrine_model_path:
                self.endocrine_analyzer.load_model(endocrine_model_path)
                
            if integrated_model_path:
                self.integrated_analyzer.load_model(integrated_model_path)
                
            self.models_loaded = True
            logger.info("All models loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def process_patient_data(self, patient_id, data_sources):
        """
        Process complete patient data from multiple sources
        
        Parameters:
        patient_id (str): Unique patient identifier
        data_sources (dict): Dictionary containing paths to different data sources
                            {
                                'ecg_data': path_to_ecg,
                                'lab_results': path_to_lab_results,
                                'patient_history': path_to_history,
                                'demographics': demographics_dict
                            }
        
        Returns:
        dict: Complete diagnostic report
        """
        logger.info(f"Processing data for patient {patient_id}")
        start_time = time.time()
        
        results = {
            'patient_id': patient_id,
            'timestamp': datetime.datetime.now().isoformat(),
            'diagnoses': {},
            'integrated_insights': {},
            'recommendations': []
        }
        
        # Process ECG data if available
        if 'ecg_data' in data_sources and data_sources['ecg_data']:
            try:
                ecg_signal, metadata = self.ecg_analyzer.load_ecg_data(data_sources['ecg_data'])
                ecg_results = self.ecg_analyzer.analyze_ecg(ecg_signal)
                results['diagnoses']['cardiac'] = ecg_results
                logger.info(f"ECG analysis complete: {ecg_results['condition']}")
            except Exception as e:
                logger.error(f"ECG analysis failed: {e}")
                results['diagnoses']['cardiac'] = {'error': str(e)}
        
        # Process endocrine lab data if available
        if 'lab_results' in data_sources and data_sources['lab_results']:
            try:
                lab_data = self.endocrine_analyzer.load_patient_data(data_sources['lab_results'])
                analyzed_labs = self.endocrine_analyzer.analyze_lab_results(lab_data)
                
                # Add demographics if available
                if 'demographics' in data_sources and data_sources['demographics']:
                    for key, value in data_sources['demographics'].items():
                        analyzed_labs[key] = value
                
                endocrine_results = self.endocrine_analyzer.diagnose_patient(analyzed_labs)
                results['diagnoses']['endocrine'] = endocrine_results
                
                # Generate detailed report
                if 'demographics' in data_sources:
                    report = self.endocrine_analyzer.generate_report(analyzed_labs, endocrine_results, data_sources['demographics'])
                    results['diagnoses']['endocrine']['detailed_report'] = report
                
                logger.info(f"Endocrine analysis complete: {endocrine_results['diagnosis']}")
            except Exception as e:
                logger.error(f"Endocrine analysis failed: {e}")
                results['diagnoses']['endocrine'] = {'error': str(e)}
        
        # Perform integrated analysis if we have both sets of results
        if 'cardiac' in results['diagnoses'] and 'endocrine' in results['diagnoses'] and \
           'error' not in results['diagnoses']['cardiac'] and 'error' not in results['diagnoses']['endocrine']:
            try:
                integrated_results = self.integrated_analyzer.analyze_combined_data(
                    results['diagnoses']['cardiac'],
                    results['diagnoses']['endocrine']
                )
                results['integrated_insights'] = integrated_results
                logger.info("Integrated analysis complete")
            except Exception as e:
                logger.error(f"Integrated analysis failed: {e}")
                results['integrated_insights'] = {'error': str(e)}
        
        # Generate clinical recommendations
        results['recommendations'] = self._generate_recommendations(results)
        
        # Add processing metadata
        processing_time = time.time() - start_time
        results['metadata'] = {
            'processing_time_seconds': processing_time,
            'medsync_version': self.version
        }
        
        logger.info(f"Complete patient analysis finished in {processing_time:.2f} seconds")
        return results
    
    def _generate_recommendations(self, results):
        """
        Generate clinical recommendations based on all analyses
        
        Parameters:
        results (dict): Combined diagnostic results
        
        Returns:
        list: Clinical recommendations
        """
        recommendations = []
        
        # Add cardiac-specific recommendations
        if 'cardiac' in results['diagnoses'] and 'error' not in results['diagnoses']['cardiac']:
            cardiac = results['diagnoses']['cardiac']
            
            if cardiac['condition'] == "Atrial Fibrillation":
                recommendations.append({
                    'priority': 'high',
                    'domain': 'cardiac',
                    'recommendation': 'Initiate anticoagulation assessment using CHA₂DS₂-VASc score',
                    'evidence': f"Detected {cardiac['condition']} with {cardiac['confidence']:.0%} confidence"
                })
                recommendations.append({
                    'priority': 'high',
                    'domain': 'cardiac',
                    'recommendation': 'Consider rate control medication',
                    'evidence': f"Detected {cardiac['condition']}"
                })
            
            elif cardiac['condition'] == "Left Bundle Branch Block":
                recommendations.append({
                    'priority': 'medium',
                    'domain': 'cardiac',
                    'recommendation': 'Evaluate for underlying structural heart disease with echocardiogram',
                    'evidence': f"Detected {cardiac['condition']}"
                })
        
        # Add endocrine-specific recommendations
        if 'endocrine' in results['diagnoses'] and 'error' not in results['diagnoses']['endocrine']:
            endocrine = results['diagnoses']['endocrine']
            
            if "Diabetes" in endocrine['diagnosis']:
                recommendations.append({
                    'priority': 'high',
                    'domain': 'endocrine',
                    'recommendation': 'Initiate glucose monitoring protocol',
                    'evidence': f"Diagnosed {endocrine['diagnosis']}"
                })
            
            elif "Hypothyroidism" in endocrine['diagnosis']:
                recommendations.append({
                    'priority': 'medium',
                    'domain': 'endocrine',
                    'recommendation': 'Initiate thyroid hormone replacement therapy',
                    'evidence': f"Diagnosed {endocrine['diagnosis']}"
                })
        
        # Add integrated recommendations
        if 'integrated_insights' in results and 'error' not in results['integrated_insights']:
            if 'cardiac_endocrine_interactions' in results['integrated_insights']:
                for interaction in results['integrated_insights']['cardiac_endocrine_interactions']:
                    recommendations.append({
                        'priority': interaction['severity'],
                        'domain': 'integrated',
                        'recommendation': interaction['recommendation'],
                        'evidence': interaction['finding']
                    })
        
        # Sort recommendations by priority
        priority_map = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        recommendations.sort(key=lambda x: priority_map.get(x['priority'], 4))
        
        return recommendations
    
    def save_models(self, output_dir):
        """Save all models to the specified directory"""
        os.makedirs(output_dir, exist_ok=True)
        
        self.ecg_analyzer.save_model(os.path.join(output_dir, "ecg_model.h5"))
        self.endocrine_analyzer.save_model(os.path.join(output_dir, "endocrine_model.joblib"))
        self.integrated_analyzer.save_model(os.path.join(output_dir, "integrated_model.joblib"))
        
        logger.info(f"All models saved to {output_dir}")
        return True
    
    def export_report_to_ehr(self, results, ehr_system, api_credentials):
        """
        Export diagnostic results to hospital EHR system
        
        Parameters:
        results (dict): Diagnostic results
        ehr_system (str): EHR system identifier (e.g., 'epic', 'cerner')
        api_credentials (dict): API credentials for EHR access
        
        Returns:
        dict: Export status
        """
        # This is a placeholder function - actual implementation would
        # depend on the specific EHR API requirements
        logger.info(f"Exporting results to {ehr_system} EHR system")
        
        # In a real implementation, this would handle the EHR-specific API calls
        
        return {
            'status': 'success',
            'ehr_system': ehr_system,
            'timestamp': datetime.datetime.now().isoformat(),
            'message': 'Results successfully exported to EHR'
        }


class ECGAnalysisModule:
    """
    Module for ECG signal analysis and cardiac condition detection
    """
    
    def __init__(self):
        """Initialize the ECG analysis module"""
        self.model = None
        self.classes = None
        self.history = None
        self.input_shape = None
        
        # Define class mapping for predictions
        self.class_mapping = {
            0: "Normal Sinus Rhythm",
            1: "Atrial Fibrillation",
            2: "First-degree AV Block",
            3: "Left Bundle Branch Block",
            4: "Right Bundle Branch Block"
        }
        
        logger.info("ECG Analysis Module initialized")
    
    def load_ecg_data(self, file_path, sample_rate=500):
        """
        Load ECG signal data from various file formats
        
        Parameters:
        file_path (str): Path to ECG data file
        sample_rate (int): ECG sample rate in Hz
        
        Returns:
        tuple: ECG data array and metadata
        """
        # Different importers based on file extension
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
            # Assuming first column is time and second is ECG
            ecg_signal = data.iloc[:, 1].values
            
        elif file_path.endswith('.npy'):
            ecg_signal = np.load(file_path)
            
        elif file_path.endswith('.txt'):
            ecg_signal = np.loadtxt(file_path)
            
        else:
            raise ValueError("Unsupported file format. Use CSV, NPY, or TXT.")
        
        # Create time axis
        time_axis = np.arange(0, len(ecg_signal) / sample_rate, 1/sample_rate)
        
        logger.info(f"Loaded ECG signal with {len(ecg_signal)} data points")
        return ecg_signal, {'sample_rate': sample_rate, 'time_axis': time_axis}
    
    def preprocess_ecg(self, ecg_signal, window_size=1000, step=500):
        """
        Preprocess ECG signal and segment into windows
        
        Parameters:
        ecg_signal (array): Raw ECG signal
        window_size (int): Size of each window in samples
        step (int): Step size between windows
        
        Returns:
        array: Processed and segmented ECG data
        """
        # Normalize the signal
        ecg_normalized = (ecg_signal - np.mean(ecg_signal)) / np.std(ecg_signal)
        
        # Apply bandpass filter to remove noise (simulated)
        # In a real implementation, use scipy.signal.butter and filtfilt
        
        # Segment the signal into windows
        segments = []
        for i in range(0, len(ecg_normalized) - window_size, step):
            segment = ecg_normalized[i:i + window_size]
            segments.append(segment)
            
        segments = np.array(segments)
        logger.info(f"Preprocessed ECG signal into {len(segments)} segments")
        return segments
    
    def build_model(self, input_shape):
        """
        Build a 1D CNN model for ECG classification
        
        Parameters:
        input_shape (tuple): Shape of input data
        
        Returns:
        Model: Keras model
        """
        self.input_shape = input_shape
        
        model = Sequential([
            # First convolutional block
            Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            
            # Second convolutional block
            Conv1D(filters=128, kernel_size=5, activation='relu'),
            MaxPooling1D(pool_size=2),
            
            # Third convolutional block
            Conv1D(filters=256, kernel_size=5, activation='relu'),
            MaxPooling1D(pool_size=2),
            
            # Flatten the output and feed into dense layer
            Flatten(),
            
            # Dense layers
            Dense(units=128, activation='relu'),
            Dropout(0.5),
            Dense(units=64, activation='relu'),
            Dropout(0.3),
            
            # Output layer - 5 classes for different cardiac conditions
            Dense(units=5, activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        logger.info("ECG analysis model built successfully")
        return model
    
    def train_model(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.2):
        """
        Train the ECG classification model
        
        Parameters:
        X_train (array): Training ECG segments
        y_train (array): Training labels
        epochs (int): Number of training epochs
        batch_size (int): Batch size
        validation_split (float): Fraction of data for validation
        
        Returns:
        History: Training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model first.")
            
        # Set up early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping]
        )
        
        logger.info("ECG model training completed")
        return self.history
    
    def analyze_ecg(self, ecg_signal, window_size=1000):
        """
        Analyze a new ECG signal and provide diagnostic suggestions
        
        Parameters:
        ecg_signal (array): ECG signal to analyze
        window_size (int): Window size for segmentation
        
        Returns:
        dict: Analysis results and diagnostic suggestions
        """
        if self.model is None:
            raise ValueError("Model not trained. Train the model first.")
            
        # Preprocess the signal
        processed_signal = self.preprocess_ecg(ecg_signal, window_size=window_size, step=window_size)
        
        if len(processed_signal) == 0:
            raise ValueError("Signal too short for analysis with current window size.")
            
        # Reshape for model input - add channel dimension
        processed_signal = processed_signal.reshape(processed_signal.shape[0], processed_signal.shape[1], 1)
        
        # Make predictions
        predictions = self.model.predict(processed_signal)
        
        # Aggregate predictions across windows
        avg_predictions = np.mean(predictions, axis=0)
        predicted_class = np.argmax(avg_predictions)
        confidence = avg_predictions[predicted_class]
        
        condition = self.class_mapping.get(predicted_class, "Unknown")
        
        # Generate diagnostic suggestions
        if condition == "Normal Sinus Rhythm":
            suggestion = "ECG appears normal. No immediate intervention required."
            urgency = "low"
        elif condition == "Atrial Fibrillation":
            suggestion = "Atrial fibrillation detected. Consider anticoagulation therapy assessment and rate control."
            urgency = "high"
        elif condition == "First-degree AV Block":
            suggestion = "First-degree AV block detected. Monitor for progression to higher-degree blocks."
            urgency = "medium"
        elif condition == "Left Bundle Branch Block":
            suggestion = "Left bundle branch block detected. Evaluate for underlying structural heart disease."
            urgency = "medium"
        elif condition == "Right Bundle Branch Block":
            suggestion = "Right bundle branch block detected. May be normal variant or indicate pulmonary disease."
            urgency = "medium"
        else:
            suggestion = "Unclear diagnosis. Consider manual review by a cardiologist."
            urgency = "medium"
        
        logger.info(f"ECG analysis complete: {condition} (confidence: {confidence:.2f})")
        
        return {
            'condition': condition,
            'confidence': float(confidence),
            'all_probabilities': {self.class_mapping[i]: float(avg_predictions[i]) for i in range(len(avg_predictions))},
            'suggestion': suggestion,
            'urgency': urgency,
            'diagnostic_features': self._extract_diagnostic_features(ecg_signal)
        }
    
    def _extract_diagnostic_features(self, ecg_signal):
        """
        Extract key diagnostic features from the ECG signal
        
        Parameters:
        ecg_signal (array): ECG signal
        
        Returns:
        dict: Key diagnostic features
        """
        # This is a simplified implementation - in a real system, this would
        # include advanced ECG feature extraction
        
        # Calculate heart rate (simplified)
        threshold = np.mean(ecg_signal) + 1.5 * np.std(ecg_signal)
        r_peaks = np.where((ecg_signal > threshold) & 
                          (ecg_signal > np.roll(ecg_signal, 1)) & 
                          (ecg_signal > np.roll(ecg_signal, -1)))[0]
        
        # Average RR interval in samples
        if len(r_peaks) >= 2:
            rr_intervals = np.diff(r_peaks)
            avg_rr = np.mean(rr_intervals)
            # Assuming 500 Hz sampling rate
            heart_rate = 60 * 500 / avg_rr
            # Heart rate variability (standard deviation of RR intervals)
            hrv = np.std(rr_intervals) / 500  # in seconds
        else:
            heart_rate = None
            hrv = None
        
        return {
            'heart_rate': float(heart_rate) if heart_rate is not None else None,
            'heart_rate_variability': float(hrv) if hrv is not None else None,
            'r_peak_count': len(r_peaks),
            'signal_quality': 'good' if len(r_peaks) > 5 else 'poor'
        }
    
    def save_model(self, filepath):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        self.model.save(filepath)
        logger.info(f"ECG model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        self.model = tf.keras.models.load_model(filepath)
        logger.info(f"ECG model loaded from {filepath}")
        return self.model


class EndocrineAnalysisModule:
    """
    Module for analyzing endocrine lab results and diagnosing endocrine disorders
    """
    
    def __init__(self):
        """Initialize the endocrine analysis module"""
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.reference_ranges = self._load_reference_ranges()
        logger.info("Endocrine Analysis Module initialized")
    
    def _load_reference_ranges(self):
        """
        Load reference ranges for common endocrine lab tests
        Returns a dictionary with test names and their normal ranges
        """
        # These would typically be loaded from a database
        return {
            "TSH": {"unit": "mIU/L", "low": 0.4, "high": 4.0},
            "Free T4": {"unit": "ng/dL", "low": 0.8, "high": 1.8},
            "Free T3": {"unit": "pg/mL", "low": 2.3, "high": 4.2},
            "Glucose (fasting)": {"unit": "mg/dL", "low": 70, "high": 100},
            "HbA1c": {"unit": "%", "low": 4.0, "high": 5.6},
            "Insulin": {"unit": "μIU/mL", "low": 2.6, "high": 24.9},
            "Cortisol (morning)": {"unit": "μg/dL", "low": 5, "high": 23},
            "ACTH": {"unit": "pg/mL", "low": 7.2, "high": 63},
            "Prolactin": {"unit": "ng/mL", "low": 4, "high": 15.2},
            "Testosterone (male)": {"unit": "ng/dL", "low": 280, "high": 1100},
            "Testosterone (female)": {"unit": "ng/dL", "low": 15, "high": 70},
            "Estradiol (male)": {"unit": "pg/mL", "low": 10, "high": 40},
            "Estradiol (female, follicular)": {"unit": "pg/mL", "low": 30, "high": 120},
            "LH (male)": {"unit": "mIU/mL", "low": 1.5, "high": 9.3},
            "LH (female, follicular)": {"unit": "mIU/mL", "low": 1.9, "high": 12.5},
            "FSH (male)": {"unit": "mIU/mL", "low": 1.6, "high": 8},
            "FSH (female, follicular)": {"unit": "mIU/mL", "low": 2.5, "high": 10.2}
        }
    
    def load_patient_data(self, file_path):
        """
        Load patient lab data from various file formats
        
        Parameters:
        file_path (str): Path to lab data file
        
        Returns:
        DataFrame: Patient lab data
        """
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            data = pd.read_excel(file_path)
        elif file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                data = pd.DataFrame(json.load(f))
        else:
            raise ValueError("Unsupported file format. Use CSV, Excel, or JSON.")
        
        logger.info(f"Loaded patient lab data with {len(data)} records")
        return data
    
    def analyze_lab_results(self, lab_data):
        """
        Analyze laboratory results and flag abnormal values
        
        Parameters:
        lab_data (DataFrame): Patient lab data
        
        Returns:
        dict: Analyzed lab results with flags
        """
        results = {}
        
        # Process each lab test
        for test in lab_data.columns:
            if test in self.reference_ranges:
                value = lab_data[test].iloc[-1]  # Get most recent value
                unit = self.reference_ranges[test]["unit"]
                low = self.reference_ranges[test]["low"]
                high = self.reference_ranges[test]["high"]
                
                # Determine if value is abnormal
                if value < low:
                    flag = "low"
                    significance = "high" if value < low * 0.5 else "moderate"
                elif value > high:
                    flag = "high"
                    significance = "high" if value > high * 1.5 else "moderate"
                else:
                    flag = "normal"
                    significance = "none"
                
                results[test] = {
                    "value": value,
                    "unit": unit,
                    "reference_range": f"{low}-{high}",
                    "flag": flag,
                    "significance": significance
                }
                
                # For time-series data, calculate trends
                if len(lab_data) > 1:
                    previous_values = lab_data[test].values
                    results[test]["trend"] = self._calculate_trend(previous_values)
        
        logger.info(f"Analysis complete: {sum(1 for v in results.values() if v['flag'] != 'normal')} abnormal values")
        return results
    
    def _calculate_trend(self, values):
        """
        Calculate trend from a series of values
        
        Parameters:
        values (array): Series of lab values over time
        
        Returns:
        str: Trend description
        """
        # Simple linear regression to determine trend
        if len(values) < 3:
            return "insufficient data"
            
        x = np.arange(len(values))
        y = values
        
        # Calculate slope using least squares
        slope = (np.mean(x * y) - np.mean(x) * np.mean(y)) / (np.mean(x**2) - np.mean(x)**2)
        
        # Last 3 values
        recent = values[-3:]
        
        if slope > 0.05:
            return "increasing"
        elif slope < -0.05:
            return "decreasing"
        else:
            return "stable"
    
    def build_model(self, features=None):
        """
        Build a machine learning model for endocrine disorder prediction
        
        Parameters:
        features (list): List of feature names to use
        
        Returns:
        Model: Machine learning model
        """
        # In a real system, this would be trained on historical data
        # Here we'll use a RandomForest as an example
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        self.model = model
        logger.info("Endocrine analysis model built successfully")
        return model
    
    def train_model(self, X_train, y_train):
        """
        Train the endocrine disorder prediction model
        
        Parameters:
        X_train (array): Training features
        y_train (array): Training labels
        
        Returns:
        Model: Trained model
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model first.")
        
        # Scale the data
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Handle imbalanced classes using SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)
        
        # Train the model
        self.model.fit(X_resampled, y_resampled)
        
        # Store feature importance for interpretation
        self.feature_importance = dict(zip(
            X_train.columns, 
            self.model.feature_importances_
        ))
        
        logger.info("Endocrine model training completed")
        return self.model
    
    def diagnose_patient(self, analyzed_labs):
        """
        Diagnose patient based on analyzed lab results
        
        Parameters:
        analyzed_labs (dict): Analyzed lab results
        
        Returns:
        dict: Diagnostic assessment
        """
        # Rule-based diagnostic approach
        diagnosis = []
        flags = {}
        explanation = []
        risk_factors = []
        
        # Check for diabetes markers
        if 'Glucose (fasting)' in analyzed_labs and analyzed_labs['Glucose (fasting)']['flag'] == 'high':
            if analyzed_labs['Glucose (fasting)']['value'] >= 126:
                diagnosis.append("Type 2 Diabetes Mellitus")
                explanation.append(f"Fasting glucose {analyzed_labs['Glucose (fasting)']['value']} mg/dL (diagnostic threshold: ≥126)")
                flags['diabetes'] = True
            elif analyzed_labs['Glucose (fasting)']['value'] >= 100:
                diagnosis.append("Prediabetes")
                explanation.append(f"Fasting glucose {analyzed_labs['Glucose (fasting)']['value']} mg/dL (prediabetic range: 100-125)")
                flags['prediabetes'] = True
        
        if 'HbA1c' in analyzed_labs:
            if analyzed_labs['HbA1c']['value'] >= 6.5:
                if 'diabetes' not in flags:
                    diagnosis.append("Type 2 Diabetes