import torch
from torchvision import models, transforms
from PIL import Image
import gradio as gr

# Cihaz seçimi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modeli yükle
model = models.resnet50(pretrained=False)
num_classes = 51
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load('fruits_vegetables_51.pth', map_location=device))
model.to(device)
model.eval()

# Dönüştürme işlemi
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Sınıf isimleri
class_names = [
    'Amaranth', 'Apple', 'Banana', 'Beetroot', 'Bell pepper',
    'Bitter Gourd', 'Blueberry', 'Bottle Gourd', 'Broccoli', 'Cabbage',
    'Cantaloupe', 'Capsicum', 'Carrot', 'Cauliflower', 'Chilli pepper',
    'Coconut', 'Corn', 'Cucumber', 'Dragon_fruit', 'Eggplant',
    'Fig', 'Garlic', 'Ginger', 'Grapes', 'Jalepeno',
    'Kiwi', 'Lemon', 'Mango', 'Okra', 'Onion',
    'Orange', 'Paprika', 'Pear', 'Peas', 'Pineapple',
    'Pomegranate', 'Potato', 'Pumpkin', 'Raddish', 'Raspberry',
    'Ridge Gourd', 'Soy beans', 'Spinach', 'Spiny Gourd', 'Sponge Gourd',
    'Strawberry', 'Sweetcorn', 'Sweetpotato', 'Tomato', 'Turnip',
    'Watermelon'
]

# Tahmin fonksiyonu
def predict(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return class_names[predicted.item()]

# Gradio arayüzü
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Meyve & Sebze Sınıflandırıcı (ResNet-50)",
    description="Bir meyve veya sebze resmi yükleyin, model ne olduğunu tahmin etsin!"
)

# Arayüzü başlat
interface.launch()