
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Embedding
from keras_preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Veri seti oluşturuluyor
data = {
    "text": [
        "Yemek çok güzeldi", 
        "Yemek hiç güzel değildi",  
        "Servis çok hızlıydı",  
        "Servis çok yavaş",  
        "Fiyatlar oldukça uygun", 
        "Fiyatlar aşırı pahalı", 
        "Atmosfer çok hoştu",  
        "Ortam çok gürültülüydü", 
        "Yemekler lezzetliydi",  
        "Yemekler soğuk ve tatsızdı", 
        "Personel çok nazikti",  
        "Garsonlar çok ilgisizdi",  
        "Burası harika bir mekan",  
        "Burası kesinlikle tavsiye edilmez", 
        "Tatlılar mükemmeldi",  
        "Tatlılar hiç hoşuma gitmedi",  
        "Yemeklerin sunumu çok güzeldi",  
        "Sunumda bir eksiklik vardı",  
        "Çalışanlar çok yardımcı oldu",  
        "Çalışanlar yetersizdi",  
        "Çok keyifli bir akşam yemeği oldu",  
        "Akşam yemeği tam bir hayal kırıklığıydı",  
        "Restoran çok temizdi",  
        "Restoran kirliydi",  
        "Mekan gerçekten çok hoş",  
        "Mekan çok kötüydü",  
        "Yemek porsiyonları yeterliydi",  
        "Porsiyonlar çok küçüktü",  
        "Yemekler sıcak geldi",  
        "Yemekler soğuk geldi",  
        "Güzel bir deneyimdi",  
        "Deneyim çok kötüydü",  
        "Süper bir deneyim yaşadım",  
        "Yine asla gelmem",  
        "Restoranda harika bir manzara vardı",  
        "Manzara berbat",  
        "Süper bir akşam yemeği deneyimi",  
        "Yemeklerin tadı çok kötüydü",  
        "Yemekler taze ve lezzetliydi",  
        "Yemekler bayattı",  
        "Fiyat kalite oranı çok iyiydi",  
        "Fiyat kalite oranı çok kötüydü",  
        "Yemeklerin tadı harikaydı",  
        "Yemekler oldukça kötüydü",  
        "Restoran çok güzel ve sakin",  
        "Restoran çok kalabalıktı",  
        "Servis çok hızlıydı ve çok beğendim",  
        "Servis yavaş ve kötüydü",  
        "Yemekler çok lezzetliydi",  
        "Yemeklerin tadı çok kötüydü",  
        "Yemekler harikaydı",  
        "Yemekler tatsız ve kötüydü",  
        "Restoran gerçekten çok iyi",  
        "Restoran oldukça kötüydü"
    ], 
    "label": [
        "Pozitif",  
        "Negatif",  
        "Pozitif",  
        "Negatif",  
        "Pozitif",  
        "Negatif",  
        "Pozitif",  
        "Negatif",  
        "Pozitif",  
        "Negatif",  
        "Pozitif",  
        "Negatif",  
        "Pozitif",  
        "Negatif",  
        "Pozitif",  
        "Negatif",  
        "Pozitif",  
        "Negatif",  
        "Pozitif",  
        "Negatif",  
        "Pozitif",  
        "Negatif",  
        "Pozitif",  
        "Negatif",  
        "Pozitif",  
        "Negatif",  
        "Pozitif",  
        "Negatif",  
        "Pozitif",  
        "Negatif",  
        "Pozitif",  
        "Negatif",  
        "Pozitif",  
        "Negatif",  
        "Pozitif",  
        "Negatif",  
        "Pozitif",  
        "Negatif",  
        "Pozitif",  
        "Negatif",  
        "Pozitif",  
        "Negatif",  
        "Pozitif",  
        "Negatif",  
        "Pozitif",  
        "Negatif",  
        "Pozitif",  
        "Negatif",  
        "Pozitif",  
        "Negatif",
        "Pozitif",  # Eklenen 1. değer
        "Negatif",  # Eklenen 2. değer
        "Pozitif",  # Eklenen 3. değer
        "Negatif"   # Eklenen 4. değer
    ]
}

# DataFrame oluşturma
import pandas as pd
df = pd.DataFrame(data)

# Sonuçları kontrol etme
print(df.head())


# Tokenizer kullanarak metinleri sayısal değerlere çevirme
tokenizer=Tokenizer()
tokenizer.fit_on_texts(df["text"])
sequence=tokenizer.texts_to_sequences(df["text"])
word_index=tokenizer.word_index
maxlen=max(len(seq) for seq in sequence)
x=pad_sequences(sequence,maxlen=maxlen)
print(x.shape)

# Etiketleri sayısal değerlere dönüştürme
label_encoder=LabelEncoder()
y=label_encoder.fit_transform(df["label"])

# Eğitim ve test verilerini ayırma
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
senteces=[text.split() for text in df["text"]]

# Word2Vec modeli oluşturma
word2vec_model=Word2Vec(senteces,vector_size=50,window=5,min_count=1)
embedding_dim=50
embedding_matrix=np.zeros((len(word_index)+1,embedding_dim))
for word , i in word_index.items():
    if word in word2vec_model.wv:
        embedding_matrix[i]=word2vec_model.wv[word]
        
# Modelin oluşturulması
model=Sequential()
model.add(Embedding(input_dim=len(word_index)+1, 
                    output_dim=embedding_dim, 
                    weights=[embedding_matrix], 
                    input_length=maxlen,  # Buradaki hata düzeltildi
                    trainable=False))
model.add(SimpleRNN(50,return_sequences=False))
model.add(Dense(1,activation="sigmoid"))

# Modelin derlenmesi
model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"]
              )

# Modelin eğitilmesi
model.fit(x_train,
          y_train,
          epochs=10,
          batch_size=2,
          validation_data=(x_test,y_test))

# Kullanıcıdan alınan cümlenin tahmin edilmesi için fonksiyon
def classifier_data(sentence):
    seq = tokenizer.texts_to_sequences([sentence])  
    padded_seq = pad_sequences(seq, maxlen=maxlen)  
    prediction = model.predict(padded_seq)  
    prediction_class = (prediction > 0.5).astype(int)  
    label = "Pozitif" if prediction_class[0][0] == 1 else "Negatif"  
    return label  

# Test cümlesi
sentence = "Yemek çok kötüydü"
result = classifier_data(sentence)
print(result)

































