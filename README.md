# music-classification

# 🎵 Müzik Enstrümanlarını Sınıflandırma (Music Classification)

Bu proje, **derin öğrenme ve sinir ağları (YSA)** kullanarak müzik enstrümanlarını sınıflandırmak için geliştirilmiştir. 
Librosa ve TensorFlow kullanarak ses dosyalarından **MFCC özellikleri çıkarılıp**, bir yapay sinir ağı (ANN) modeli ile tahmin yapılmaktadır.

---

## 🚀 Özellikler
- 📌 **Ses verisi işleme** (Librosa ile MFCC özellikleri çıkarma)
- 🎼 **Müzik enstrümanlarını sınıflandırma**
- 🧠 **Derin öğrenme modeli ile tahmin yapma**
- 📊 **Veri analizi ve değerlendirme metrikleri**

---

## 📂 Kullanım
1. **Google Drive'ı bağlayın ve veri setini yükleyin.**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   

## 📂 1. Proje Kurulumu ve Kullanımı
🔹 1.1 Gerekli Kütüphaneleri Yükleyin
-İlk olarak, çalıştırmadan önce gerekli kütüphanelerin yüklü olduğundan emin olun.

    pip install numpy pandas librosa matplotlib tqdm tensorflow sklearn imbalanced-learn
  
-Eğer Google Colab kullanıyorsanız:

    !pip install numpy pandas librosa matplotlib tqdm tensorflow sklearn imbalanced-learn
  
🔹 1.2 Google Drive Bağlantısı (Colab Kullanıyorsanız) 

         from google.colab import drive
         drive.mount('/content/drive')

  📌 Burada "data_dir" değerine kendi Drive yolunu girmelisin.
  
      data_dir = input("Lütfen veri setinin bulunduğu Drive klasör yolunu girin (Örn: MyDrive/YSA): ")
  
🔹 1.3 Veri Setini Yükleyin
Eğitim ve test verisini yükleyelim:

    import pandas as pd
  
    train_metadata = pd.read_csv(f'/content/drive/{data_dir}/Metadata_Train.csv')
    test_metadata = pd.read_csv(f'/content/drive/{data_dir}/Metadata_Test.csv')
    
    # Veri setinin ilk 5 satırına göz atalım
    print(train_metadata.head())
    
    # Eksik verileri kontrol edelim
    print(train_metadata.isnull().sum())

  🏗 2. Model Eğitimi
🔹 2.1 Ses Dosyalarını İşleyip Özellik Çıkaralım (MFCC)

    import librosa
    import numpy as np
    
    def extract_features(file_path):
        y, sr = librosa.load(file_path, duration=3)  # 3 saniyelik ses yükleme
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        return np.mean(mfcc.T, axis=0)
    
    # Örnek bir dosyanın MFCC özelliklerini çıkaralım
    file_path = f"/content/drive/{data_dir}/some_audio_file.wav"
    features = extract_features(file_path)
    print(features.shape)  # Çıktı: (40,) olmalı

  🔹 2.2 Model için Veri Setini Hazırlayalım

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    
    # Bağımsız değişkenleri ve hedef değişkeni ayıralım
    X = train_metadata['FileName'].apply(lambda x: extract_features(f'/content/drive/{data_dir}/{x}'))
    X = np.array(X.tolist())
    
    # Hedef değişkeni encode edelim
    encoder = LabelEncoder()
    y = encoder.fit_transform(train_metadata['Class'])
    
    # Eğitim ve test verisini ayıralım
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Eğitim veri boyutu:", X_train.shape)
    print("Test veri boyutu:", X_test.shape)


🔹 2.3 Sinir Ağı Modelini Eğitelim

    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    
    # Modeli oluştur
    model = Sequential([
        Dense(256, activation='relu', input_shape=(40,)),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(len(set(y_train)), activation='softmax')  # Çıkış katmanı
    ])
    
    # Modeli derle
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Modeli eğit
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

    
🎯 3. Modeli Test Edelim ve Doğruluk Oranını Ölçelim

    from sklearn.metrics import accuracy_score
    import numpy as np
    
    # Tahmin yap
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Doğruluk oranı
    accuracy = accuracy_score(y_test, y_pred_classes)
    print(f"Test Doğruluk Oranı: {accuracy * 100:.2f}%")

📂 4. Eğitilmiş Modeli Kaydetme ve Kullanma
Modeli .h5 formatında kaydedelim:

    model.save("/content/music_classification_model.h5")
    
Daha sonra tekrar yüklemek için:

    from tensorflow.keras.models import load_model
    model = load_model("/content/music_classification_model.h5")

📌 5. GitHub Bağlantısı ve Proje Linki
📂 Proje Repository: https://github.com/sule-ozt/music-classification

📌 6. Katkıda Bulunma
Bu projeye katkıda bulunmak isterseniz:

1. Fork yapın 🍴 (Projeyi kendi hesabınıza kopyalayın).
2. Değişiklik yapın ve commit atın 🔨.
3. Pull Request gönderin 🔄 (Değişikliklerinizi bize önerin).
💡 Katkıda bulunmak için önce GitHub hesabınıza Fork yaparak başlayabilirsiniz!



