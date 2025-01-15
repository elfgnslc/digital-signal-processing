import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.utils import load_data, filter_signals, apply_fft, train_and_evaluate_model
from sklearn.model_selection import train_test_split
import json
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from itertools import cycle
from scripts.utils import (
    load_data,
    filter_signals,
    apply_fft,
    plot_signal_comparison,
    plot_signals,
    plot_fft_spectrum,
    train_and_evaluate_model
)
def main():
    print("Program başlatılıyor...")
    
    project_dirs = ['data', 'notebooks', 'scripts', 'results']
    print(f"Oluşturulacak dizinler: {project_dirs}")
    
    for dir_name in project_dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"{dir_name} dizini oluşturuldu")

    data_path = os.path.join('data', 'final_corrected_file.xlsx')
    results_dir = 'results'

    try:
        # 1. Veri Yükleme
        print("Veri yükleniyor...")
        X, y, df = load_data(data_path)
        
        # 2. Ön İşleme ve Filtreleme
        print("Sinyaller filtreleniyor ve ön işleniyor...")
        filtered_signals = filter_signals(X)
        
        # Her sınıf için filtreleme karşılaştırması
        unique_classes = np.unique(y)
        for class_num in unique_classes:
            class_mask = y == class_num
            # Filtreleme karşılaştırması için yeni bir plot oluştur
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.plot(X[class_mask][0], label='Ham Sinyal')
            plt.title(f'Sınıf {class_num} - Ham Sinyal')
            plt.xlabel('Örnek')
            plt.ylabel('Genlik')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(filtered_signals[class_mask][0], label='Filtrelenmiş Sinyal')
            plt.title(f'Sınıf {class_num} - Filtrelenmiş Sinyal')
            plt.xlabel('Örnek')
            plt.ylabel('Genlik')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f'sinyal_karsilastirma_sinif_{class_num}.png'))
            plt.close()
        
        # 3. FFT Analizi ve Özellik Çıkarımı
        print("FFT analizi ve özellik çıkarımı yapılıyor...")
        freqs, magnitude_spectrum, fft_features = apply_fft(filtered_signals)
        
        # Sınıf 5 için ek özellik çıkarımı
        print("Sınıf 5 için ek özellikler hesaplanıyor...")
        additional_features = np.zeros((len(X), 3))
        for i in range(len(X)):
            additional_features[i, 0] = np.max(filtered_signals[i])
            additional_features[i, 1] = np.min(filtered_signals[i])
            additional_features[i, 2] = np.std(filtered_signals[i])
        
        # FFT özelliklerine ek özellikleri ekle
        fft_features = np.hstack((fft_features, additional_features))
        
        # 4. Sınıflandırma
        print("Özellikler normalize ediliyor ve sınıflandırma yapılıyor...")
        
        # Veriyi böl
        X_train, X_test, y_train, y_test = train_test_split(
            fft_features, y, test_size=0.2, random_state=42)
        
        # Model parametrelerini güncelle
        model_params = {
            'hidden_layer_sizes': (100,),
            'max_iter': 500,
            'alpha': 0.0001,
            'activation': 'relu',
            'solver': 'adam'
        }
        
        # Neural Network modelini eğit ve değerlendir
        model, conf_matrix, class_report = train_and_evaluate_model(
            X_train, X_test, y_train, y_test, model_params, 
            n_classes=len(unique_classes))
        
        plt.style.use('default')  
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['axes.grid'] = True
        plt.rcParams['font.size'] = 12

        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Karışıklık Matrisi - Neural Network')
        plt.xlabel('Tahmin Edilen Sınıf')
        plt.ylabel('Gerçek Sınıf')
        plt.savefig(os.path.join(results_dir, 'karisiklik_matrisi_nn.png'))
        plt.close()

        # Filtreleme karşılaştırması
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        ax1.plot(X[0], label='Ham Sinyal')
        ax1.set_title('Filtreleme Öncesi')
        ax1.set_xlabel('Örnek Sayısı')
        ax1.set_ylabel('Genlik')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(filtered_signals[0], label='Filtrelenmiş Sinyal')
        ax2.set_title('Filtreleme Sonrası')
        ax2.set_xlabel('Örnek Sayısı')
        ax2.set_ylabel('Genlik')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'filtreleme_karsilastirmasi.png'))
        plt.close()

        results = {
            'classification_report': class_report,
            'model_parameters': model_params
        }
        with open(os.path.join(results_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=4)

    except Exception as e:
        print(f"Hata oluştu: {str(e)}")
        import traceback
        print("Detaylı hata:")
        print(traceback.format_exc())

if __name__ == "__main__":
    print("Script başlatılıyor")
    main()