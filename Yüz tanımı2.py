import os
import sys
import cv2
import numpy as np
import math
import pyttsx3
from keras.models import model_from_json
import datetime
from deepface import DeepFace
import face_recognition

def yuz_guveni(yuz_mesafe, yuz_eslesme_esigi=0.6):
    aralik_degeri = (1.0 - yuz_eslesme_esigi)
    lineer_deger = (1.0 - yuz_mesafe) / (aralik_degeri * 2.0)

    if yuz_mesafe > yuz_eslesme_esigi:
        return str(round(lineer_deger * 100, 2)) + '%'
    else:
        deger = (lineer_deger + ((1.0 - lineer_deger) * math.pow((lineer_deger - 0.5) * 2, 0.2))) * 100
        return str(round(deger, 2)) + '%'

class YuzTanima:
    def __init__(self):
        self.yuz_konumlari = []
        self.yuz_kodlamalari = []
        self.yuz_isimleri = []
        self.taninan_yuz_kodlamalari = []
        self.taninan_yuz_isimleri = []
        self.aktif_kareyi_isle = True

        self.yuzleri_kodla()

        # Tanınan bir yüz olduğunu kontrol etmek için flag
        self.taninan_yuz_var = False

        self.emotion_labels = {0: 'Kizgin', 1: 'Tiksinti', 2: 'Korku', 3: 'Mutlu', 4: 'Normal', 5: 'Uzgun', 6: 'Sasirmis'}

    def yuzleri_kodla(self):
        for resim in os.listdir('faces'):
            yuz_resmi = face_recognition.load_image_file(os.path.join('faces', resim))
            yuz_kodlamasi = face_recognition.face_encodings(yuz_resmi)[0]
            self.taninan_yuz_kodlamalari.append(yuz_kodlamasi)
            self.taninan_yuz_isimleri.append(resim)
        print("Bulunan Dosyalar:", self.taninan_yuz_isimleri)

    def preprocess_image(self, image):
        feature = np.array(image)
        feature = feature.reshape(1, 48, 48, 1)
        return feature / 255.0

    def tanimlamayi_surdur(self):
        video = cv2.VideoCapture(0)
        
        # Load the model from JSON and weights
        json_file = open("C:/Users/90537/facialemotionmodel.json", "r")
        model_json = json_file.read()
        json_file.close()
        model = model_from_json(model_json)
        model.load_weights("C:/Users/90537/facialemotionmodel.h5")

        # Load the Haarcascade classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        if not video.isOpened():
            sys.exit('Video Açılamadı')

        # Tanınan yüz için bir kere sesli uyarı yapılması için flag
        self.sesli_uyari_yapildi = False

        while True:
            ret, kare = video.read()

            kucuk_kare = cv2.resize(kare, (0, 0), fx=0.25, fy=0.25)
            rgb_kucuk_kare = cv2.cvtColor(kucuk_kare, cv2.COLOR_BGR2RGB)

            self.yuz_konumlari = face_recognition.face_locations(rgb_kucuk_kare)
            self.yuz_kodlamalari = face_recognition.face_encodings(rgb_kucuk_kare, self.yuz_konumlari)
            self.yuz_isimleri = []
            
            gray = cv2.cvtColor(kare, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            
            filename = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            cv2.putText(kare, filename, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            for yuz_kodlamasi in self.yuz_kodlamalari:
                eslesmeler = face_recognition.compare_faces(self.taninan_yuz_kodlamalari, yuz_kodlamasi)
                isim = 'Bilinmiyor'
                guven = 'Bilinmiyor'
                yuz_mesafesi = face_recognition.face_distance(self.taninan_yuz_kodlamalari, yuz_kodlamasi)
                en_iyi_eslesme_indeksi = np.argmin(yuz_mesafesi)

                if eslesmeler[en_iyi_eslesme_indeksi]:
                    isim = self.taninan_yuz_isimleri[en_iyi_eslesme_indeksi]
                    guven = yuz_guveni(yuz_mesafesi[en_iyi_eslesme_indeksi])

                    # Tanınan yüz için bir kere sesli uyarı yapılması
                    if not self.sesli_uyari_yapildi:
                        self.sesli_olarak_soylenen_isim_guven(isim, guven)
                        self.sesli_uyari_yapildi = True

                self.yuz_isimleri.append(f'{isim} ({guven})')

            # Eğer tanınan yüz yoksa, flag'leri sıfırla
            if not self.yuz_konumlari:
                self.taninan_yuz_var = False
                self.sesli_uyari_yapildi = False
            
            for (x, y, w, h), isim, face in zip(self.yuz_konumlari, self.yuz_isimleri, faces):
                x *= 5
                y *= 5
                w *= 5
                h *= 5
                yuz_image = gray[y:y + h, x:x + w]

                # Add a check for empty image before resizing
                if not yuz_image.size:
                    continue

                cv2.rectangle(kare, (x, y), (x + w, y + h), (0, 255, 0), 3)
                yuz_image = cv2.resize(yuz_image, (48, 48))
                processed_image = self.preprocess_image(yuz_image)
                prediction = model.predict(processed_image)
                predicted_label = self.emotion_labels[np.argmax(prediction)]
                result = DeepFace.analyze(kare, actions=["age"], enforce_detection=False)
                yas = result[0]['age']
                metin = isim.replace(".jpg", "")
                cv2.rectangle(kare, (x, y), (x + w, y + h), (0, 255, 255), -1)
                cv2.putText(kare, f"isim:{metin} yas:{yas}", (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (255, 255, 0), 3)
                cv2.putText(kare, f"label:{predicted_label} basarım:{float(prediction[0][np.argmax(prediction)]) * 100:.2f}%", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (255, 255, 0), 3)

            # Calculate fps after displaying the frame
            cv2.imshow("Yuz Tanima", kare)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()

    def sesli_olarak_soylenen_isim_guven(self, isim, guven):
        metin = f"{isim} ismi"
        metin1 = metin.replace('.jpg', '')
        text_to_speech("Merhaba Hoşgeldin" + metin1)

def text_to_speech(text, rate=150, volume=1.0, lang='tr'):
    engine = pyttsx3.init()
    engine.setProperty('rate', rate)
    engine.setProperty('volume', volume)
    engine.say(text)
    engine.runAndWait()

if __name__ == '__main__':
    yt = YuzTanima()
    yt.tanimlamayi_surdur()
