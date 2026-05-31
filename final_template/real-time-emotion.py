import cv2
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO

# --- PyTorch Model Definition ---
# Define your CNN architecture here
import torch.nn.functional as F

class myCNNModel(torch.nn.Module):
    
    def __init__(self):
        
        super(myCNNModel, self).__init__()
        
        # Convolution followed by ReLU activation and a Maxpool
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=32,
                              kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # Convolution followed by ReLU activation and a Maxpool
        self.cnn2 = nn.Conv2d(in_channels=32, out_channels=64, 
                              kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # added
        self.cnn3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.bn3 = nn.BatchNorm2d(128)

        # Squishing down to fully connected chunk.
        self.fc1 = nn.Linear(128*6*6, 256)
        self.fc2 = nn.Linear(256, 7)

        # Batch normalizations used:
        self.bn = nn.BatchNorm1d(256)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)

        # dropouts
        self.dropout = nn.Dropout(0.5)
        self.dropout_conv = nn.Dropout(0.25)
    
    def forward(self, x):
        
        # Convolution, batch normalization, ReLU, Maxpool
        out = self.cnn1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.dropout_conv(out) # *

        # Convolution, batch normalization, ReLU, Maxpool
        out = self.cnn2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        out = self.dropout_conv(out) # * 

        # ^ same as before
        out = self.cnn3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.maxpool3(out)
        out = self.dropout_conv(out) # *Googled how to reduce overfitting and added this dropout

        # Flattening out our output
        out = out.view(out.size(0), -1)

        # Using a fully connected layer for the rest
        out = self.fc1(out)
        out = self.bn(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

# --- Initialization ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
face_model = YOLO("yolov11n-face.pt")

emotion_model = myCNNModel().to(device)
# Load weights (Ensure you have converted your .h5 to .pth or have a .pth file)
# emotion_model.load_state_dict(torch.load("model.pth", map_location=device))
emotion_model.eval()

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# --- Data Structures ---
baseline_stats = {'negative_faces': [], 'baseline_negative_avg': -1, 'start_time': time.time()}
BASELINE_DURATION = 30 
rolling_stats = {'rolling_negative_faces': []}
WINDOW_SIZE = 5

def analyze_emotions():
    cap = cv2.VideoCapture(0) 
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        results = face_model(frame)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # Pre-processing (OpenCV -> PyTorch Tensor)
                face = frame[y1:y2, x1:x2]
                if face.size == 0: continue
                face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face_resized = cv2.resize(face_gray, (48, 48))
                
                # Convert to Tensor: (Batch, Channel, Height, Width)
                face_tensor = torch.from_numpy(face_resized).float().to(device)
                face_tensor = face_tensor.unsqueeze(0).unsqueeze(0) / 255.0

                # Prediction
                with torch.no_grad():
                    output = emotion_model(face_tensor)
                    prob = F.softmax(output, dim=1)
                    confidence, pred_idx = torch.max(prob, 1)
                    
                    max_val = pred_idx.item()
                    emotion = emotion_dict[max_val]
                    conf_val = confidence.item()



                cv2.putText(frame, f"{emotion} ({conf_val:.2f})", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("PyTorch Emotion Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    analyze_emotions()