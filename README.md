# 💬 Akıllı Finans Asistanı

Belgeleri anlayan, finansal kavramları yorumlayan ve **RAG (Retrieval-Augmented Generation)** mimarisiyle çalışan bir **Yapay Zekâ Destekli Finans Asistanı**.  
Bu sistem, yalnızca bankacılık sözleşmelerini değil, geliştirici tarafından hazırlanmış **özgün finans veri setlerini** de kullanarak kullanıcı sorularına bağlama dayalı yanıtlar üretir.

---

## 🎯 1. Projenin Amacı

Bu proje, finansal metinleri (örneğin faiz hesaplamaları, kredi ürünleri, yatırım terimleri vb.) doğal dilde işleyerek:

- **Finansal kavramları sadeleştirmek**,  
- Belgelerden bağlama dayalı doğru bilgi çıkarmak,  
- **Kullanıcıya kaynaklı yanıtlar** sunmak (hangi belgeye dayandığını belirtmek),  
- Ve **yatırım tavsiyesi vermeden** bilgilendirici, açıklayıcı bir sistem kurmak amacıyla geliştirilmiştir.

> Bu proje, **Akbank GenAI Bootcamp**'in “Generative AI + RAG” proje gereksinimlerini yerine getirir.

---

## 💾 2. Veri Seti Hakkında Bilgi

### 🔹 2.1 Kaynak Türleri
Uygulama hem **resmî banka dokümanlarını** hem de geliştirici tarafından oluşturulan **özgün eğitim veri setlerini** birlikte kullanır:

#### 📘 a) Banka Belgeleri
- `Bireysel_Bankacilik_Hizmet_Sozlesmesi_2025.pdf`
- `Kredi_Kartlari_Sozlesmesi_2025.pdf`
- `Ozel_Bankacilik_Hizmet_Sozlesmesi_2025.pdf`
- `Ticari_Hizmetler_Sozlesmesi_2025.pdf`  
Bu belgeler, **bankacılık terminolojisi** ve **yasal yükümlülükleri** açıklayan kaynaklardır.

#### 💡 b) Geliştirici Tarafından Oluşturulan Veri Setleri
- `kb_faiz_temelleri.json` → Faiz türleri, bileşik faiz, vadeli/vadesiz hesap açıklamaları:contentReference[oaicite:0]{index=0}  
- `kb_kredi_urunleri.json` → Kredi türleri, PMT formülü, erken kapama ve risk analizi:contentReference[oaicite:1]{index=1}  
- `kb_yatirim_terimleri.json` → Risk-getiri ilişkisi, volatilite, varlık türleri ve çeşitlendirme kavramları:contentReference[oaicite:2]{index=2}

Bu veri setleri, kullanıcıya finans kavramlarını öğretici biçimde açıklamak üzere geliştirici tarafından derlenmiştir.

---

### 🔹 2.2 Veri Toplama Metodolojisi

1. **Kavramsal Seçim:** Finansal konular (faiz, kredi, yatırım) belirlendi.  
2. **Kaynak Taraması:** Türkiye’deki güncel finansal terminoloji, açık kaynak bankacılık dokümanları ve eğitim materyalleri incelendi.  
3. **Yapılandırma:** Her konu için 2–3 paragraf açıklayıcı metin oluşturuldu.  
4. **Etiketleme:** Her metin için `chunk_id`, `source_file`, `title`, `page`, `text` alanları eklendi.  
5. **JSON Formatı:** Veriler `kb_*.json` formatında kaydedildi ve ChromaDB’ye indekslenmeye hazır hale getirildi.

> Bu yöntem sayesinde sistem, yalnızca belgelerden değil, geliştirici tarafından seçilmiş finansal bilgi kümelerinden de öğrenir.

---

## ⚙️ 3. Kurulum ve Çalıştırma

### 🧩 Adımlar

1. Projeyi klonlayın:
```bash
git clone https://github.com/<kullaniciadi>/akilli-finans-asistani.git
cd akilli-finans-asistani
```

2. Sanal ortam oluşturun:
```bash
python -m venv .venv
```

3. Ortamı aktif edin:
```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

4. Gerekli kütüphaneleri yükleyin:
```bash
pip install -r requirements.txt
```

5. `.env` dosyasını oluşturun:
```
GEMINI_API_KEY=YOUR_GEMINI_KEY
OPENAI_API_KEY=YOUR_OPENAI_KEY
```

6. Uygulamayı çalıştırın:
```bash
streamlit run app.py
```

7. Tarayıcıda `http://localhost:8501` adresinde açılacaktır. 

---

## 🧠 4. Çözüm Mimarisi

**Akış:**
1. Kullanıcı bir veya birden fazla belge seçer / yükler.
2. PDF → Metin çıkarımı → Parçalama (chunkify)
3. Embedding → ChromaDB’ye kayıt
4. Kullanıcı sorusu → En benzer `Top-K` parçaların alınması
5. RAG pipeline ile prompt oluşturulması
6. Yanıt Gemini veya OpenAI LLM’den alınır
7. Kaynak referansları ve metin “kart” yapısında görüntülenir.

**Mimari Bileşenler:**

| Katman | Teknoloji | Açıklama |
|--------|------------|-----------|
| Arayüz | Streamlit | UI, ayarlar, metrikler |
| Embedding | SentenceTransformer | all-MiniLM-L6-v2 |
| Vektör DB | ChromaDB | cosine similarity |
| Model | Gemini / OpenAI | LLM motoru |
| Dosya İşleme | PyPDF2 (pypdf) | PDF → metin parçalama |
| Finans Araçları | yfinance, numpy | Hisse, kur, PMT hesaplama |

---

## 🌐 5. Web Arayüzü & Product Kılavuzu

### 🔧 Özellikler
- **Sekmeli UI:** 💬 Sohbet · 📚 Kaynaklar · 🧰 Finans Araçları  
- **Sohbet Sekmesi:** Kullanıcı soru girer, model yanıt üretir, geçmiş listelenir.  
- **Kaynaklar Sekmesi:** Tüm indeksli JSON dosyaları, parça sayıları ve önizleme.  
- **Finans Araçları:**  
  - Hisse fiyatı (yfinance)  
  - Kur çevirici (USD/EUR/TRY)  
  - Kredi/Loan ödeme hesaplayıcı (PMT)

### 🖥️ Ekran Akışı
1. Sayfa açıldığında başlık + açıklama görünür.
2. Sol menüden model, temperature, Top-K, stream ayarları yapılabilir.
3. PDF/CSV yüklenebilir, anında JSON’a çevrilir.
4. “Sor” butonu ile yanıt alınır, kaynaklar referans etiketiyle gösterilir.

---

## 🔗 Deploy Linki
🔹 [Streamlit Cloud Üzerinde Deneyin](https://akilli-finans-asistani.streamlit.app)  
_(Demo adresinizi buraya ekleyin.)_

---

## ⚙️ 6. Sistemin Çalışma Mantığı
1. Ayarlar Menüsü
“Mod”:
Genel Finans Taraması → Yalnızca LLM kullanır.
Belge Tabanlı Tarama → Belgelerden veri çeker (RAG aktif).
“LLM Motoru”: Gemini veya OpenAI arasında seçim yapılabilir.
“Top-K”: Cevap üretiminde kaç belge parçasının kullanılacağını belirler.
“Yaratıcılık (Temperature)”: Yanıtların teknik mi, yoksa yaratıcı mı olacağını ayarlar.

2. Veri Kaynakları Alanı
JSON/CSV veya PDF dosyalarını yükleyebilirsin.
PDF’ler otomatik olarak parçalara ayrılır ve .json olarak kaydedilir.
“Yeniden indeksle” düğmesiyle veritabanını sıfırlayıp yeniden oluşturabilirsin.

3. Sohbet Sekmesi
“Soru Sor” alanına metnini yaz (örneğin: Vadeli mevduat nasıl işler?, Fon birim fiyatı neye göre değişir?, Kalıcı veri saklayıcısı ne demektir?, Taksitli alışverişlerde faiz nasıl işler?, Kredi çekerken efektif faiz oranı ne anlama gelir?).
Sistem, belgeleri tarar, uygun parçaları bulur ve modelden yanıt alır.
Cevaplar kaynak referanslarıyla birlikte görünür.

4. Finans Araçları Sekmesi
Hisse Fiyatı, Kur Çevirici ve Kredi Hesaplayıcı gibi araçlarla canlı finansal veriler alınabilir.

---

## 📚 Kullanılan Kütüphaneler
```
streamlit
google-generativeai
openai
chromadb
sentence-transformers
pypdf
yfinance
python-dotenv
```

---

## ⚠️ Uyarı
Bu uygulama **eğitim amaçlıdır** ve **yatırım tavsiyesi vermez**.  
Yanıtlar yalnızca bilgi verme ve belge özetleme amaçlıdır.

---

## 🗂️ Proje Yapısı

```
.
├─ app.py               # Streamlit uygulaması (UI + RAG + LLM çağrıları)
├─ data/                # JSON/CSV kaynakları, PDF’lerden üretilen JSON’lar
├─ .chroma_store/       # ChromaDB kalıcı dizini
├─ requirements.txt     # Bağımlılıklar
└─ .env                 # API anahtarları (GEMINI_API_KEY / OPENAI_API_KEY)
```

---

## ✍️ Notlar
Bu proje, **Akbank Gençlik Akademisi & Global AI Hub GenAI Bootcamp** kapsamında geliştirilmiştir.  

---
