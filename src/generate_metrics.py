import torch
from torchvision import transforms, models
from PIL import Image
import os


DATASET_PATH = r"path/to/your/dataset/test"  

MODEL_PATH = "../model/best_model.pth" 


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print(f"Loading model from {MODEL_PATH}...")
    
    # --- LOAD MODEL ---
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    class_names = checkpoint.get('class_names', ['Aorta','Flows','Other','V Sign','X Sign'])
    
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(class_names))
    model.load_state_dict(checkpoint['state_dict'])
    model.to(DEVICE).eval()
    
    # --- TRANSFORM ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    true_labels = []
    pred_labels = []

    print("Starting analysis... this might take a minute.")

  
  
    for class_folder in class_names:
        folder_path = os.path.join(DATASET_PATH, class_folder)
        
        if not os.path.exists(folder_path):
            print(f"Warning: Folder not found for class '{class_folder}'. Skipping.")
            continue
            
        images = os.listdir(folder_path)
        print(f"Processing '{class_folder}' ({len(images)} images)...")
        
        for img_name in images:
            img_path = os.path.join(folder_path, img_name)
            
        
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            try:
               
                true_labels.append(class_folder)

              
                image = Image.open(img_path).convert("RGB")
                x = transform(image).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    out = model(x)
                    prob = torch.softmax(out, dim=1)
                    _, pred = torch.max(prob, 1)
                    
                predicted_class = class_names[pred.item()]
                pred_labels.append(predicted_class)
                
            except Exception as e:
                print(f"Error processing {img_name}: {e}")


    print("\n" + "="*50)
    print("DONE! COPY THE LISTS BELOW INTO YOUR STREAMLIT APP")
    print("="*50)
    print(f"\ntrue_labels = {true_labels}")
    print(f"\npred_labels = {pred_labels}")
    print("\n" + "="*50)

if __name__ == "__main__":
    main()