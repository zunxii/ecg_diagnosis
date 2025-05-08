import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns

class ECGAnalysisSystem:
    """
    A deep learning system for analyzing ECG data to detect cardiac abnormalities
    and assist in diagnostic decision-making.
    """
    
    def __init__(self):
        """Initialize the ECG analysis system"""
        self.model = None
        self.classes = None
        self.history = None
        self.input_shape = None
    
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
            
        return np.array(segments)
    
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
            
            # Output layer - number of units depends on classification task
            Dense(units=5, activation='softmax')  # 5 classes in this example
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
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
        
        return self.history
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the model performance
        
        Parameters:
        X_test (array): Test ECG segments
        y_test (array): Test labels
        
        Returns:
        dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Train the model first.")
            
        # Evaluate the model
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test)
        
        # Get predictions
        y_pred_prob = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        # Print classification report
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Plot ROC curve (for binary classification)
        if len(np.unique(y_test)) == 2:
            fpr, tpr, _ = roc_curve(y_test, y_pred_prob[:, 1])
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.show()
        
        # Plot training history
        if self.history is not None:
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.plot(self.history.history['accuracy'], label='Training Accuracy')
            plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(self.history.history['loss'], label='Training Loss')
            plt.plot(self.history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.tight_layout()
            plt.show()
        
        return {
            'accuracy': test_accuracy,
            'loss': test_loss,
            'confusion_matrix': cm,
            'classification_report': classification_report(y_test, y_pred)
        }
    
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
            
        # Make predictions
        predictions = self.model.predict(processed_signal)
        
        # Aggregate predictions across windows
        avg_predictions = np.mean(predictions, axis=0)
        predicted_class = np.argmax(avg_predictions)
        confidence = avg_predictions[predicted_class]
        
        # Map class index to ECG condition (example mapping)
        class_mapping = {
            0: "Normal Sinus Rhythm",
            1: "Atrial Fibrillation",
            2: "First-degree AV Block",
            3: "Left Bundle Branch Block",
            4: "Right Bundle Branch Block"
        }
        
        condition = class_mapping.get(predicted_class, "Unknown")
        
        # Generate diagnostic suggestions
        if condition == "Normal Sinus Rhythm":
            suggestion = "ECG appears normal. No immediate intervention required."
        elif condition == "Atrial Fibrillation":
            suggestion = "Atrial fibrillation detected. Consider anticoagulation therapy assessment and rate control."
        elif condition == "First-degree AV Block":
            suggestion = "First-degree AV block detected. Monitor for progression to higher-degree blocks."
        elif condition == "Left Bundle Branch Block":
            suggestion = "Left bundle branch block detected. Evaluate for underlying structural heart disease."
        elif condition == "Right Bundle Branch Block":
            suggestion = "Right bundle branch block detected. May be normal variant or indicate pulmonary disease."
        else:
            suggestion = "Unclear diagnosis. Consider manual review by a cardiologist."
        
        # Visualize the ECG with analysis
        self.visualize_analysis(ecg_signal, condition, confidence)
        
        return {
            'condition': condition,
            'confidence': float(confidence),
            'all_probabilities': {class_mapping[i]: float(avg_predictions[i]) for i in range(len(avg_predictions))},
            'suggestion': suggestion
        }
    
    def visualize_analysis(self, ecg_signal, condition, confidence):
        """
        Visualize ECG with analysis results
        
        Parameters:
        ecg_signal (array): Original ECG signal
        condition (str): Detected condition
        confidence (float): Prediction confidence
        """
        # Create time axis (assuming 500 Hz sample rate)
        time = np.arange(0, len(ecg_signal) / 500, 1/500)
        
        plt.figure(figsize=(15, 6))
        plt.plot(time, ecg_signal, 'b-')
        plt.title(f'ECG Analysis: {condition} (Confidence: {confidence:.2f})')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        
        # Add markers for R-peaks (simplified example)
        # In a real implementation, use a QRS detection algorithm
        threshold = np.mean(ecg_signal) + 1.5 * np.std(ecg_signal)
        r_peaks = np.where((ecg_signal > threshold) & 
                          (ecg_signal > np.roll(ecg_signal, 1)) & 
                          (ecg_signal > np.roll(ecg_signal, -1)))[0]
        
        plt.scatter(time[r_peaks], ecg_signal[r_peaks], color='red', marker='x', s=100, label='R-peaks')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
        return self.model


# Example Usage:

def demo_ecg_analysis():
    """Demonstrate the ECG analysis system with synthetic data"""
    # Create synthetic ECG data
    np.random.seed(42)
    
    # Create synthetic normal sinus rhythm
    def generate_ecg(num_beats=15, sample_rate=500, noise_level=0.05):
        # Parameters for PQRST complex
        p_amplitude = 0.25
        q_amplitude = -0.5
        r_amplitude = 1.5
        s_amplitude = -0.5
        t_amplitude = 0.35
        
        # Time intervals
        beat_duration = 0.8  # seconds
        p_duration = 0.08
        pr_interval = 0.16
        qrs_duration = 0.08
        qt_interval = 0.36
        
        # Convert to samples
        beat_samples = int(beat_duration * sample_rate)
        total_samples = beat_samples * num_beats
        
        # Initialize ECG signal
        ecg = np.zeros(total_samples)
        
        # For each beat
        for i in range(num_beats):
            # Beat start sample
            beat_start = i * beat_samples
            
            # P wave
            p_center = beat_start + int(0.1 * sample_rate)
            p_width = int(p_duration * sample_rate)
            p_start = p_center - p_width // 2
            p_end = p_center + p_width // 2
            t = np.linspace(-np.pi/2, np.pi/2, p_end - p_start)
            ecg[p_start:p_end] = p_amplitude * np.sin(t)
            
            # QRS complex
            q_pos = beat_start + int(pr_interval * sample_rate)
            r_pos = q_pos + int(0.04 * sample_rate)
            s_pos = r_pos + int(0.04 * sample_rate)
            
            # Q wave
            ecg[q_pos:r_pos] = np.linspace(0, q_amplitude, r_pos - q_pos)
            
            # R wave
            ecg[r_pos-3:r_pos+3] = r_amplitude
            
            # S wave
            ecg[s_pos-2:s_pos+2] = s_amplitude
            ecg[s_pos+2:s_pos+10] = np.linspace(s_amplitude, 0, 8)
            
            # T wave
            t_center = beat_start + int(qt_interval * sample_rate)
            t_width = int(0.16 * sample_rate)
            t_start = t_center - t_width // 2
            t_end = t_center + t_width // 2
            t = np.linspace(-np.pi/2, np.pi/2, t_end - t_start)
            ecg[t_start:t_end] = t_amplitude * np.sin(t)
        
        # Add noise
        noise = np.random.normal(0, noise_level, total_samples)
        noisy_ecg = ecg + noise
        
        return noisy_ecg
    
    # Generate different types of ECG for demo
    normal_ecg = generate_ecg(num_beats=15, noise_level=0.03)
    
    # Create synthetic dataset
    # In real-world, this would be actual patient ECG data
    num_samples = 1000
    window_size = 500
    
    # Initialize arrays
    X = np.zeros((num_samples, window_size))
    y = np.zeros(num_samples)
    
    # Generate different ECG patterns
    for i in range(num_samples):
        if i < 200:  # Normal
            signal = generate_ecg(num_beats=5, noise_level=0.03)
            label = 0
        elif i < 400:  # AF - more irregular
            signal = generate_ecg(num_beats=6, noise_level=0.1)
            # Make irregular
            signal = signal + 0.2 * np.sin(np.linspace(0, 20*np.pi, len(signal)))
            label = 1
        elif i < 600:  # First-degree AV block - longer PR interval
            signal = generate_ecg(num_beats=4, noise_level=0.05)
            # Stretch PR interval
            label = 2
        elif i < 800:  # LBBB
            signal = generate_ecg(num_beats=5, noise_level=0.04)
            # Modify QRS
            label = 3
        else:  # RBBB
            signal = generate_ecg(num_beats=5, noise_level=0.04)
            # Different QRS
            label = 4
            
        # Extract window
        if len(signal) >= window_size:
            start_idx = np.random.randint(0, len(signal) - window_size)
            X[i] = signal[start_idx:start_idx + window_size]
            y[i] = label
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the model
    ecg_analyzer = ECGAnalysisSystem()
    ecg_analyzer.build_model(input_shape=(window_size, 1))
    
    # Reshape data for CNN (add channel dimension)
    X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # Train with reduced epochs for demo
    ecg_analyzer.train_model(X_train_reshaped, y_train, epochs=10, batch_size=32)
    
    # Evaluate
    metrics = ecg_analyzer.evaluate_model(X_test_reshaped, y_test)
    
    # Analyze a new ECG
    new_ecg = generate_ecg(num_beats=10, noise_level=0.04)
    new_ecg_reshaped = new_ecg.reshape(1, len(new_ecg), 1)
    
    # For simplicity in demo, we'll just analyze a segment
    if len(new_ecg) > window_size:
        analysis_result = ecg_analyzer.analyze_ecg(new_ecg[:window_size])
        print("\nECG Analysis Results:")
        print(f"Detected Condition: {analysis_result['condition']}")
        print(f"Confidence: {analysis_result['confidence']:.2f}")
        print(f"Suggestion: {analysis_result['suggestion']}")
    
    return ecg_analyzer

if __name__ == "__main__":
    analyzer = demo_ecg_analysis()