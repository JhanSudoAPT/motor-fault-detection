import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import yaml

class ModelTrainer:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Create directories
        self.models_dir = self.config['paths']['models_dir']
        self.processed_dir = self.config['paths']['processed_dir']
        self.output_dir = self.config['paths']['output_dir']
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Set parameters
        self.seed = self.config['project']['seed']
        self.training_params = self.config['training']
        self.expected_features = self.config['features']['expected']
        self.snr_levels = self.config['noise']['snr_levels']
        
        # Set random seeds for reproducibility
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
    
    def load_and_prepare_data(self):
        """Load and prepare data for training"""
        # Load clean dataset
        data_path = os.path.join(self.processed_dir, "dataset.csv")
        df = pd.read_csv(data_path)
        
        # Verify we have all expected features
        missing = set(self.expected_features) - set(df.columns)
        if missing:
            raise ValueError(f"Missing features in dataset: {missing}")
        
        X = df[self.expected_features]  # Ensure correct feature order
        y = df['fault'].values
        
        # Encode labels
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        joblib.dump(le, os.path.join(self.models_dir, 'label_encoder.pkl'))
        
        # Create reproducible splits
        total_idx = np.arange(len(df))
        idx_trainval, idx_test = train_test_split(
            total_idx, test_size=self.training_params['test_size'], 
            random_state=self.seed, stratify=y_enc)
        
        idx_train, idx_val = train_test_split(
            idx_trainval, test_size=self.training_params['val_size'], 
            random_state=self.seed, stratify=y_enc[idx_trainval])
        
        # Save splits
        joblib.dump({'train': idx_train, 'val': idx_val, 'test': idx_test},
                    os.path.join(self.models_dir, 'data_splits.pkl'))
        
        # Create arrays
        X_arr = X.values
        y_arr = y_enc
        X_train = X_arr[idx_train]
        X_val = X_arr[idx_val]
        X_test = X_arr[idx_test]
        y_train = y_arr[idx_train]
        y_val = y_arr[idx_val]
        y_test = y_arr[idx_test]
        
        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        
        # Save scaler
        joblib.dump(scaler, os.path.join(self.models_dir, 'scaler.pkl'))
        pd.DataFrame({
            'feature': self.expected_features,
            'mean': scaler.mean_,
            'std': scaler.scale_
        }).to_csv(os.path.join(self.models_dir, 'scaler_params.csv'), index=False)
        
        return (X_train, X_val, X_test, y_train, y_val, y_test, le.classes_)
    
    def build_model(self, input_dim, output_dim):
        """Build neural network model with architecture n → 2n → n → m"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(input_dim, activation='tanh', input_shape=(input_dim,)),
            tf.keras.layers.Dense(2 * input_dim, activation='relu'),
            tf.keras.layers.Dense(input_dim, activation='relu'),
            tf.keras.layers.Dense(output_dim, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.training_params['learning_rate']),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self):
        """Train the model"""
        # Load and prepare data
        X_train, X_val, X_test, y_train, y_val, y_test, classes = self.load_and_prepare_data()
        
        # Build model
        n_features = X_train.shape[1]
        n_classes = len(classes)
        model = self.build_model(n_features, n_classes)
        model.summary()
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.training_params['epochs'],
            batch_size=self.training_params['batch_size'],
            verbose=2,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    patience=5, monitor='val_loss', restore_best_weights=True),
                tf.keras.callbacks.ModelCheckpoint(
                    os.path.join(self.models_dir, 'best_model.keras'),
                    monitor='val_accuracy',
                    save_best_only=True,
                    mode='max'
                )
            ]
        )
        
        # Save training history
        pd.DataFrame(history.history).to_csv(
            os.path.join(self.models_dir, 'training_history.csv'), index=False)
        
        # Load best model
        best_model = tf.keras.models.load_model(os.path.join(self.models_dir, 'best_model.keras'))
        
        # Evaluate on clean test data
        test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)
        print(f"Test accuracy (clean data): {test_acc:.4f}")
        
        # Evaluate on noisy data
        noise_results = self.evaluate_on_noisy_data(best_model, X_test, y_test)
        noise_results['clean'] = test_acc
        
        # Create performance visualization
        self.create_performance_plot(noise_results, classes)
        
        # Save final model
        final_model_path = os.path.join(self.models_dir, 'final_model.keras')
        best_model.save(final_model_path)
        
        # Export to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
        tflite_model = converter.convert()
        
        tflite_path = os.path.join(self.models_dir, 'final_model.tflite')
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"Models saved:")
        print(f"- Keras: {final_model_path}")
        print(f"- TFLite: {tflite_path}")
        
        # Save results
        pd.DataFrame.from_dict(noise_results, orient='index', columns=['accuracy']).to_csv(
            os.path.join(self.models_dir, 'final_results.csv'))
        
        return best_model, noise_results
    
    def evaluate_on_noisy_data(self, model, X_test_clean, y_test_clean):
        """Evaluate model on noisy data at different SNR levels"""
        results = {}
        scaler = joblib.load(os.path.join(self.models_dir, 'scaler.pkl'))
        le = joblib.load(os.path.join(self.models_dir, 'label_encoder.pkl'))
        
        for snr in self.snr_levels:
            snr_folder = f"SNR{snr}"
            csv_path = os.path.join(self.output_dir, snr_folder, 'csv', 'dataset.csv')
            
            if not os.path.exists(csv_path):
                print(f"Dataset not found for {snr_folder}")
                continue
            
            # Load noisy dataset
            df_n = pd.read_csv(csv_path)
            
            # Verify features
            if not set(self.expected_features).issubset(df_n.columns):
                print(f"Error: Dataset {snr_folder} missing expected features")
                continue
            
            Xn = df_n[self.expected_features].values  # Same feature order
            yn = le.transform(df_n['fault'].values)
            
            # Verify dimensions
            if Xn.shape[1] != X_test_clean.shape[1]:
                print(f"Dimensionality error in {snr_folder}: {Xn.shape[1]} vs {X_test_clean.shape[1]}")
                continue
            
            # Scale and evaluate
            Xn = scaler.transform(Xn)
            
            # Use the same test indices as clean data
            splits = joblib.load(os.path.join(self.models_dir, 'data_splits.pkl'))
            Xn_test = Xn[splits['test']]
            yn_test = yn[splits['test']]
            
            _, acc_n = model.evaluate(Xn_test, yn_test, verbose=0)
            results[snr_folder] = acc_n
            print(f"Accuracy {snr_folder}: {acc_n:.4f}")
        
        return results
    
    def create_performance_plot(self, results, classes):
        """Create performance visualization across noise levels"""
        # Prepare data
        snrs = ['clean'] + [f'SNR{snr}' for snr in self.snr_levels]
        accuracies = [results[k] for k in snrs]
        
        # Create plot
        plt.figure(figsize=(12, 7))
        
        # Scatter plot with color gradient
        scatter = plt.scatter(
            snrs,
            accuracies,
            s=200,
            c=accuracies,
            cmap='viridis',
            edgecolors='w',
            linewidth=2,
            alpha=0.8,
            zorder=3
        )
        
        # Connection line
        plt.plot(snrs, accuracies,
                 color='#2a9d8f',
                 linestyle='--',
                 linewidth=2,
                 marker='',
                 alpha=0.7,
                 zorder=2)
        
        # Customize axes and style
        plt.ylim(0, 1.05)
        plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
        plt.xlabel('Noise Level', fontsize=12, fontweight='bold')
        plt.title('Model Performance Across Different Noise Conditions',
                  fontsize=14, fontweight='bold', pad=20)
        
        # Add accuracy values
        for i, acc in enumerate(accuracies):
            plt.annotate(f'{acc:.4f}',
                         (i, acc),
                         xytext=(0, 10),
                         textcoords='offset points',
                         ha='center',
                         fontsize=10,
                         fontweight='bold')
        
        # Style improvements
        plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=1)
        plt.gca().set_facecolor('#f8f9fa')
        plt.gca().spines[['top','right']].set_visible(False)
        plt.xticks(rotation=15, ha='right', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.models_dir, 'noise_performance.png'), dpi=300)
        plt.close()
        
        print("Performance plot saved")

if __name__ == "__main__":
    trainer = ModelTrainer()
    model, results = trainer.train_model()