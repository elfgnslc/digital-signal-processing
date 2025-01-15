# Sinyal İşleme ve Sınıflandırma Projesi

Bu proje, gürültülü sinyaller üzerinde analiz ve sınıflandırma yapmak için geliştirilmiştir. Proje boyunca sinyal filtreleme, frekans alanında analiz, özellik çıkarımı ve makine öğrenimi tabanlı sınıflandırma adımları gerçekleştirilmiştir.

## Kurulum
1. Python ve gerekli kütüphaneleri yükleyin:
   ```bash
   pip install numpy scipy matplotlib pandas scikit-learn
   ```
2. Proje dizinini klonlayın veya oluşturun:
   ```bash
   git clone <proje-linki>
   cd proje-dizini
   ```
3. Veriyi `data/` klasörüne yerleştirin ve kodları çalıştırmaya başlayın.

## Kullanım
1. **Veri Yükleme ve İnceleme**:
   - `data_loader.py` dosyası kullanılarak veri yüklenir ve kontrol edilir.
2. **Gürültü Filtreleme**:
   - Butterworth filtresi uygulanarak yüksek frekanslı gürültü temizlenir.
3. **Frekans Analizi**:
   - Filtrelenmiş sinyaller üzerinde FFT uygulanarak frekans bileşenleri çıkarılır.
4. **Sınıflandırma**:
   - Rastgele Orman algoritması kullanılarak sinyaller sınıflandırılır.

## Çıktılar
- **Filtrelenmiş Sinyal Grafikleri**
- **FFT Sonuçları**
- **Sınıflandırma Performansı**: Doğruluk, karışıklık matrisi ve diğer metrikler raporlanır.

## Rapor
Proje kapsamında elde edilen bulgular ve görselleştirmeler, `report.pdf` dosyasında detaylı olarak sunulmuştur.

## Gereksinimler
- Python >= 3.8
- Gerekli kütüphaneler:
  - Numpy
  - Scipy
  - Matplotlib
  - Pandas
  - Scikit-learn

## Yazar
Elif Müberra Güneşlice
