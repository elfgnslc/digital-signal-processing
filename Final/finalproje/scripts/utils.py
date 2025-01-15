import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os
import json
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

def load_data(file_path):
    """Veri yükleme ve hazırlama"""
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Desteklenmeyen dosya formatı")
    
    X = df.iloc[:, :200].values 
    y = df['class'].values
    
    return X, y, df

def filter_signals(signals, fs=1000, lowcut=0.5, highcut=50, order=5):
    """Bant geçiren filtre uygulama"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    b, a = signal.butter(order, [low, high], btype='band')
    
    filtered_signals = np.zeros_like(signals)
    for i in range(signals.shape[0]):
        filtered_signals[i] = signal.filtfilt(b, a, signals[i])
    
    return filtered_signals

def plot_signal_comparison(original, filtered, class_num, save_dir):
    """Orijinal ve filtrelenmiş sinyalleri karşılaştırmalı çizim"""
    plt.figure(figsize=(15, 5))
    
    # Orijinal sinyal
    plt.subplot(1, 2, 1)
    for sig in original:
        plt.plot(sig)
    plt.title(f'{class_num}. Sınıf Unfiltered')
    plt.grid(True)
    
    # Filtrelenmiş sinyal
    plt.subplot(1, 2, 2)
    for sig in filtered:
        plt.plot(sig)
    plt.title(f'{class_num}. Sınıf Filtered')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'signal_comparison_class_{class_num}.png'))
    plt.close()

def apply_fft(signals, sampling_rate=1000):
    """FFT uygulama ve özellik çıkarımı"""
    n_samples = signals.shape[1]
    features = np.zeros((signals.shape[0], 10))
    
    for i in range(signals.shape[0]):
        # FFT hesaplama
        fft_result = fft(signals[i])
        freqs = fftfreq(n_samples, 1/sampling_rate)
        magnitude_spectrum = np.abs(fft_result[:n_samples//2])
        
        # Özellik çıkarımı
        features[i, 0] = np.max(magnitude_spectrum)  # Tepe genliği
        features[i, 1] = freqs[np.argmax(magnitude_spectrum)]  # Tepe frekansı
        features[i, 2] = np.mean(magnitude_spectrum)  # Ortalama genlik
        features[i, 3] = np.std(magnitude_spectrum)  # Standart sapma
        features[i, 4] = np.sum(magnitude_spectrum)  # Toplam güç
        features[i, 5] = np.median(magnitude_spectrum)  # Medyan genlik
        features[i, 6] = np.percentile(magnitude_spectrum, 75)  # 75. yüzdelik
        features[i, 7] = np.var(magnitude_spectrum)  # Varyans
        features[i, 8] = np.sum(magnitude_spectrum**2)  # Enerji
        features[i, 9] = stats.kurtosis(magnitude_spectrum)  # Basıklık
    
    return freqs[:n_samples//2], magnitude_spectrum, features

def plot_signals(signals, title, save_path):
    """Sinyalleri görselleştirme fonksiyonu"""
    plt.figure(figsize=(12, 6))
    for i, signal in enumerate(signals):
        plt.plot(signal, label=f'Sinyal {i+1}')
    plt.title(title)
    plt.xlabel('Örnek Sayısı')
    plt.ylabel('Genlik')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_fft_spectrum(freqs, magnitude, title, save_path):
    """FFT spektrumunu görselleştirme fonksiyonu"""
    plt.figure(figsize=(12, 6))
    plt.plot(freqs, magnitude)
    plt.title(title)
    plt.xlabel('Frekans (Hz)')
    plt.ylabel('Genlik')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def train_and_evaluate_model(X_train, X_test, y_train, y_test, model_params, n_classes):
    """Model eğitimi ve değerlendirmesi"""
    model = MLPClassifier(
        hidden_layer_sizes=model_params['hidden_layer_sizes'],
        activation=model_params['activation'],
        solver=model_params['solver'],
        max_iter=model_params['max_iter'],
        alpha=model_params['alpha'],
        random_state=42
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, zero_division=1)
    return model, conf_matrix, class_report