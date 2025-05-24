import cv2
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Load model
model = torch.load('Deep_Learning/Midterm/best_model_Glasses_ResNet18.pt', map_location='cpu')
model.eval()

# Class labels
class_labels = ['Glasses', 'Plain']

# Transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load image from file
image_path = 'Deep_Learning/Midterm/Data/Ed3.jpg'  # <- Replace this with your image filename
frame = cv2.imread(image_path)

if frame is None:
    print("Error: Could not read image.")
else:
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = transform(image)  # transform returns a tensor
    image = image.unsqueeze(0)  # type: ignore # Add batch dimension

    # Inference
    with torch.no_grad():
        output = model(image)
        probs = F.softmax(output, dim=1)

    prob_value, predicted_class = torch.max(probs, dim=1)
    predicted_class = predicted_class.item()
    prob_value = prob_value.item()

    predicted_class_name = class_labels[predicted_class] # type: ignore
    label_text = f"{predicted_class_name} ({prob_value*100:.2f}%)"
    print("Prediction:", label_text)

    # Convert BGR (OpenCV) to RGB (matplotlib)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.putText(frame_rgb, label_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    plt.imshow(frame_rgb)
    plt.title('Prediction')
    plt.axis('off')
    plt.show()
