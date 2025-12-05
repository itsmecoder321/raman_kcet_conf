import torch
from torchvision import transforms, models
from PIL import Image
import sys
import os

MODEL_PATH = "../model/best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path, num_classes=5):
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, num_classes)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    class_names = checkpoint.get('class_names', ['Aorta','Flows','Other','V Sign','X Sign'])
    model = model.to(DEVICE)
    model.eval()
    return model, class_names

def predict(img_path):
    model, class_names = load_model(MODEL_PATH, len(class_names:=['Aorta','Flows','Other','V Sign','X Sign']))
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)
        conf, pred = torch.max(probs, 1)
    print(f"Predicted: {class_names[pred.item()]} ({conf.item()*100:.2f}%)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python infer.py path/to/image.jpg")
        sys.exit(1)
    predict(sys.argv[1])
