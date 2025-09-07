import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from scipy.signal import get_window, welch
from scipy.stats import kurtosis, skew, entropy as scipy_entropy
from sklearn.preprocessing import StandardScaler
import yaml

class DataProcessor:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Create directories
        self.dirs = {
            'input': self.config['paths']['input_dir'],
            'processed': self.config['paths']['processed_dir'],
            'output': self.config['paths']['output_dir']
        }
        
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        # Set parameters
        self.signal_params = self.config['signal_processing']
        self.noise_params = self.config['noise']
        np.random.seed(self.config['project']['seed'])
    
    def calculate_features(self, signal):
        """Calculate 24 time and frequency domain features from a signal segment"""
        signal = np.asarray(signal).astype(float)
        if signal.size == 0:
            raise ValueError("Empty signal segment")
        
        # Signal processing parameters
        fs = self.signal_params['sampling_rate']
        desired_N = self.signal_params['fft_size']
        overlap = self.signal_params['overlap']
        use_welch = self.signal_params['use_welch']
        eps = self.signal_params['epsilon']
        
        N = min(desired_N, len(signal))
        noverlap = int(N * overlap)
        hop = N - noverlap if N > 0 else 0
        window = get_window('hann', N, fftbins=True)

        # Time-domain features
        abs_signal = np.abs(signal)
        squared_signal = signal ** 2
        meanv = np.mean(signal)
        std = np.std(signal)
        var = np.var(signal)
        rms = np.sqrt(np.mean(squared_signal))

        # Temporal entropy
        hist, _ = np.histogram(signal, bins=50)
        p = hist.astype(float) / (hist.sum() + eps)
        entropy_time = float(scipy_entropy(p + eps))

        # Spectral estimation
        if use_welch and len(signal) >= N:
            f, Pxx = welch(signal - meanv, fs=fs, window=window, nperseg=N, noverlap=noverlap)
            spec = Pxx.astype(float)
            freqs = f
        else:
            fft_accum = np.zeros(N // 2 + 1, dtype=float)
            count = 0
            if len(signal) >= N:
                for i in range(0, len(signal) - N + 1, hop):
                    seg = signal[i:i+N].astype(float)
                    seg = seg - np.mean(seg)
                    seg_w = seg * window
                    X = np.abs(np.fft.rfft(seg_w, n=N))**2
                    fft_accum += X
                    count += 1
            else:
                seg = np.pad(signal, (0, N - len(signal)), mode='constant').astype(float)
                seg = seg - np.mean(seg)
                seg_w = seg * window
                X = np.abs(np.fft.rfft(seg_w, n=N))**2
                fft_accum += X
                count = 1

            spec = (fft_accum / (count + eps)).astype(float)
            freqs = np.fft.rfftfreq(N, d=1.0/fs)

        # Spectral features
        total_energy = np.sum(spec) + eps
        spec_rel = spec / total_energy

        mask_low = freqs < 0.25 * freqs.max()
        fft_energy_low = float(np.sum(spec_rel[mask_low]))
        fft_energy_high = float(np.sum(spec_rel[~mask_low]))
        fft_peak_rel = float(np.max(spec_rel))
        centroid = float(np.sum(freqs * spec) / (total_energy))
        spread = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * spec) / (total_energy)))
        fft_entropy = float(scipy_entropy(spec_rel + eps))
        rmsf = float(np.sqrt(np.sum((freqs ** 2) * spec) / total_energy))

        # Higher moments
        n5m = float(np.mean(np.abs(signal - meanv) ** 5))
        n6m = float(np.mean(np.abs(signal - meanv) ** 6))

        return {
            'fft_energy_low': fft_energy_low,
            'fft_rvf': spread,
            'meanv': float(meanv),
            'smr': float((np.mean(np.sqrt(abs_signal))) ** 2),
            'fft_mean': float(np.mean(spec_rel)),
            'rms': float(rms),
            'sd': float(std),
            'var': float(var),
            'fft_energy_high': fft_energy_high,
            'fft_std': float(np.std(spec_rel)),
            'trough': float(np.min(signal)),
            'form': float(rms / (np.mean(abs_signal) + eps)),
            'fft_fc': centroid,
            'peak': float(np.max(abs_signal)),
            'kurto': float(kurtosis(signal)),
            'n6m': n6m,
            'fft_entropy': fft_entropy,
            'entropy': entropy_time,
            'fft_peak': fft_peak_rel,
            'impulse': float(np.max(abs_signal) / (np.mean(abs_signal) + eps)),
            'fft_rmsf': rmsf,
            'crest': float(np.max(abs_signal) / rms if rms > 0 else 0),
            'skewn': float(skew(signal)),
            'n5m': n5m
        }
    
    def add_noise(self, signal, snr_db, seed=None):
        """Add Gaussian noise to signal with specified SNR"""
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random
        signal_power = np.mean(signal ** 2)
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = signal_power / (snr_linear + 1e-15)
        noise = rng.normal(0, np.sqrt(noise_power), size=signal.shape)
        return signal + noise
    
    def process_clean_data(self):
        """Process clean signals and extract features"""
        rows = []
        mat_files = [f for f in os.listdir(self.dirs['input']) if f.endswith("_procesada.mat")]
        
        for fname in mat_files:
            path = os.path.join(self.dirs['input'], fname)
            data = loadmat(path)
            signal = data['signal_downsampled'].flatten()
            fault_type = fname.replace("_procesada.mat", "")

            segment_size = len(signal) // self.signal_params['segments_per_file']
            if segment_size == 0:
                print(f"Warning: {fname} too short for {self.signal_params['segments_per_file']} segments")
                continue

            for i in range(self.signal_params['segments_per_file']):
                segment = signal[i * segment_size: (i + 1) * segment_size]
                if len(segment) == segment_size:
                    features = self.calculate_features(segment)
                    features['fault'] = fault_type
                    rows.append(features)

        # Create DataFrame and save
        df = pd.DataFrame(rows)
        output_path = os.path.join(self.dirs['processed'], "dataset.csv")
        df.to_csv(output_path, index=False)
        print(f"Clean dataset saved: {output_path} (samples: {len(df)})")
        
        # Train and save scaler
        if not df.empty:
            cols = [c for c in df.columns if c != 'fault']
            scaler = StandardScaler()
            scaler.fit(df[cols].values)
            
            scaler_path = os.path.join(self.dirs['processed'], "scaler.pkl")
            joblib.dump({'scaler': scaler, 'columns': cols}, scaler_path)
            print(f"Scaler saved: {scaler_path}")
            
            return df, scaler, cols
        
        return None, None, None
    
    def process_noisy_data(self):
        """Process signals with added noise at different SNR levels"""
        # Load scaler if available
        scaler_path = os.path.join(self.dirs['processed'], "scaler.pkl")
        scaler_obj = None
        if os.path.isfile(scaler_path):
            try:
                scaler_obj = joblib.load(scaler_path)
                print(f"Scaler loaded: {scaler_path}")
            except Exception as e:
                print(f"Could not load scaler: {e}")
        
        # Process each SNR level
        for snr in self.noise_params['snr_levels']:
            snr_dir = os.path.join(self.dirs['output'], f"SNR{snr}")
            mats_dir = os.path.join(snr_dir, "mats")
            csv_dir = os.path.join(snr_dir, "csv")
            os.makedirs(mats_dir, exist_ok=True)
            os.makedirs(csv_dir, exist_ok=True)

            rows = []
            mat_files = [f for f in os.listdir(self.dirs['input']) if f.endswith("_procesada.mat")]
            
            for fname in mat_files:
                base_name = fname.replace('_procesada.mat', '')
                path_mat = os.path.join(self.dirs['input'], fname)
                data = loadmat(path_mat)
                signal = data['signal_downsampled'].flatten()

                # Add noise
                seed = abs(hash(base_name)) % (2**32)
                noisy_signal = self.add_noise(signal, snr, seed=seed)

                # Save noisy signal
                mat_out = os.path.join(mats_dir, f"{base_name}_SNR{snr}.mat")
                savemat(mat_out, {'signal_downsampled': noisy_signal})

                # Calculate features
                total_length = len(noisy_signal)
                segment_size = total_length // self.signal_params['segments_per_file']
                if segment_size == 0:
                    print(f"Warning: Signal {fname} too short for SNR{snr}")
                    continue

                for i in range(self.signal_params['segments_per_file']):
                    start = i * segment_size
                    end = start + segment_size
                    segment = noisy_signal[start:end]
                    if len(segment) == segment_size:
                        features = self.calculate_features(segment)
                        features['fault'] = base_name
                        rows.append(features)

            # Save dataset
            df = pd.DataFrame(rows)
            csv_out = os.path.join(csv_dir, "dataset.csv")
            df.to_csv(csv_out, index=False)
            print(f"SNR {snr} dataset saved: {csv_out} (samples: {len(df)})")
            
            # Apply scaler if available
            if scaler_obj is not None and len(df) > 0:
                scaler = scaler_obj.get('scaler')
                cols = scaler_obj.get('columns')
                try:
                    df_scaled = df.copy()
                    df_scaled[cols] = scaler.transform(df[cols].values)
                    df_scaled.to_csv(os.path.join(csv_dir, "dataset_scaled.csv"), index=False)
                    print(f"SNR {snr} scaled dataset saved")
                except Exception as e:
                    print(f"Could not apply scaler to SNR{snr}: {e}")

        # Save metadata
        meta = {
            'sampling_rate': self.signal_params['sampling_rate'],
            'fft_size': self.signal_params['fft_size'],
            'overlap': self.signal_params['overlap'],
            'segments_per_file': self.signal_params['segments_per_file'],
            'snr_levels': self.noise_params['snr_levels'],
            'use_welch': self.signal_params['use_welch']
        }
        
        with open(os.path.join(self.dirs['output'], 'metadata.json'), 'w') as f:
            json.dump(meta, f, indent=2)
        
        print("Noise processing completed.")

if __name__ == "__main__":
    processor = DataProcessor()
    processor.process_clean_data()
    processor.process_noisy_data()