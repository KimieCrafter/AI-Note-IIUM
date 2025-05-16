import cv2
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

# Load model
model = torch.load('Deep_Learning/Week After (W9+)/Gender Classification/best_model_Gender.pt')
model.eval()
model.to('cpu')

# Class labels
class_labels = ['Male', 'Female']

# Transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Open webcam
capture = cv2.VideoCapture(0)

while True:
    isTrue, frame = capture.read()
    if not isTrue:
        break

    image = frame.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image)  # transform already returns a tensor
    image = image.unsqueeze(0)  # type: ignore # Add batch dimension

    # Inference
    with torch.no_grad():
        output = model(image)
        probs = F.softmax(output, dim=1)  # Convert logits to probabilities

    prob_value, predicted_class = torch.max(probs, dim=1)
    predicted_class = predicted_class.item()
    prob_value = prob_value.item()

    predicted_class_name = class_labels[predicted_class] # type: ignore
    label_text = f"{predicted_class_name} ({prob_value*100:.2f}%)"
    print(label_text)

    # Display on frame
    cv2.putText(frame, label_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
    cv2.imshow('Gender Classification', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
