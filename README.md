# 🍎🥦 Meyve & Sebze Sınıflandırıcı (ResNet-50)

Bu proje, 51 sınıflı meyve ve sebze veri setini sınıflandırmak için önceden eğitilmiş bir **ResNet-50** modelini kullanır. Kullanıcılar, basit bir **Gradio** arayüzü üzerinden bir görsel yükleyerek sınıf tahmini alabilir.

## 🎯 Proje Amacı

Makine öğrenimi teknikleri kullanarak görsel verileri sınıflandırmak ve bu modeli kullanıcı dostu bir arayüzle entegre ederek işlevsel bir yapay zeka uygulaması üretmek.

---

## 🚀 Özellikler

- Görsel yükleme
- Görsel üzerinde ön işleme (boyutlandırma, normalize etme)
- Sınıf tahmini (51 sınıf: meyve ve sebzeler)
- Sonuçların Gradio arayüzü ile gösterilmesi

---

## 🛠️ Kullanılan Teknolojiler

- Python 3.13+
- PyTorch (torch, torchvision)
- Gradio
- Pillow
- ResNet-50 modeli (eğitilmiş hali `.pth` dosyasıyla birlikte)


## Kurulum

1. **Sanal ortam oluşturun**:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

2. **Bağımlılıkları yükleyin**:

    ```bash
    pip install -r requirements.txt    
    ```

    **Yüklenmesi gereken bağımlılıklar**
    
    ```
    torch==2.7.0
    torchvision==0.22.0
    Pillow==11.2.1
    gradio==5.29.0
    ```

3. **Kodu çalıştırın**:
 ```bash
    python main.py
 ```