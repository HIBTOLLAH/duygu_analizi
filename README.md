# RNN ile Türkçe Duygu Analizi

Bu proje, **Recurrent Neural Network (RNN)** kullanarak Türkçe metinler üzerinde duygu analizi (sentiment analysis) gerçekleştirmeyi amaçlamaktadır. Projede, kullanıcıdan alınan bir cümlenin **pozitif** veya **negatif** olup olmadığını belirlemek için derin öğrenme teknikleri kullanılmıştır.

## 🚀 Özellikler
- **Gömme Katmanı (Embedding Layer):** Türkçe kelimeler için Word2Vec tabanlı kelime gömme matrisi oluşturulmuştur.
- **Basit RNN Modeli:** Tek katmanlı SimpleRNN kullanılarak duygu tahmini yapılmaktadır.
- **Veri Ön İşleme:** Keras Tokenizer ile metinler sayısal dizilere çevrilmiş ve `pad_sequences` ile uygun uzunluğa getirilmiştir.
- **Model Eğitimi:** Model, `adam` optimizasyon algoritması ile eğitilmiş olup, `binary_crossentropy` kayıp fonksiyonu kullanılmıştır.
- **Gerçek Zamanlı Tahmin:** Kullanıcının girdiği herhangi bir cümlenin duygusunu sınıflandırabilme.

## 📂 Kullanılan Teknolojiler
- **Python** 🐍
- **Keras & TensorFlow** 🔥
- **Gensim (Word2Vec)** 📖
- **Pandas & NumPy** 📊

## 📌 Kurulum
Aşağıdaki adımları takip ederek projeyi çalıştırabilirsiniz:

1. Gerekli kütüphaneleri yükleyin:

   pip install tensorflow keras gensim numpy pandas scikit-learn

2. `Rnn_model.py` dosyasını çalıştırarak modeli eğitin ve test edin:
  
   python Rnn_model.py
  
3. Model eğitildikten sonra aşağıdaki fonksiyonu kullanarak duygu tahmini yapabilirsiniz:
   
   sentence = "Servis çok kötüydü"
   result = classifier_data(sentence)
   print(result)  # Negatif
   

## 📝 Örnek Kullanım

sentence = "Yemekler çok lezzetliydi"
print(classifier_data(sentence))  # Çıktı: Pozitif

sentence = "Mekan çok gürültülüydü"
print(classifier_data(sentence))  # Çıktı: Negatif


## 📧 İletişim
Eğer proje hakkında geri bildirim vermek isterseniz, benimle iletişime geçebilirsiniz! 😊
hiba2003alhasan@gmail.com
