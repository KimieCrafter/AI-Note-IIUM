import cv2
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image

class GlassesNetRGB(nn.Module):
    def __init__(self):
        super(GlassesNetRGB, self).__init__()
        
        # C1: Conv layer (5x5 kernel), from 3 input channels (RGB) to 6 feature maps
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)  # 256 -> 252

        # S2: Subsampling: Avg pooling (2x2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)  # 252 -> 126

        # C3: Conv layer (5x5), from 6 to 16 feature maps
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)  # 126 -> 122

        # S4: Subsampling: Avg pooling (2x2)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)  # 122 -> 61

        # Feature maps now: 16 x 61 x 61
        self.fc1 = nn.Linear(16 * 61 * 61, 120)  # F5
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(120, 84)           # F6
        self.dropout2 = nn.Dropout(p=0.5)
        self.output = nn.Linear(84, 2)          # Binary output

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        x = self.output(x)
        return x

# Load model
model = torch.load('Deep_Learning/Midterm/best_model_Glasses.pt', map_location='cpu')
model.eval()
model.eval()
model.to('cpu')

# Class labels
class_labels = ['Glasses', 'Plain']

# Transformations
data_transform = transforms.Compose(
    [transforms.Resize((256,256)), 
     transforms.ColorJitter(brightness=0.2, contrast=0.2, hue=0.1),  # Light jitter
     transforms.ToTensor(),
])

# Open webcam
capture = cv2.VideoCapture(0)

while True:
    isTrue, frame = capture.read()
    if not isTrue:
        break

    image = frame.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image) 
    image = data_transform(image)  # transform already returns a tensor
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
