from __future__ import print_function, division
import warnings
warnings.filterwarnings("ignore")

import os
import torch
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.transforms as transforms
import dlib
import argparse
import cv2

# Load Dlib models
cnn_face_detector = dlib.cnn_face_detection_model_v1('dlib_models/mmod_human_face_detector.dat')
sp = dlib.shape_predictor('dlib_models/shape_predictor_5_face_landmarks.dat')

# Torch device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load FairFace 7-class model
model = torchvision.models.resnet34(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 18)
model.load_state_dict(torch.load('fair_face_models/res34_fair_align_multi_7_20190809.pt', map_location=device))
model = model.to(device).eval()

# Define transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Label mappings
race_labels = ['White', 'Black', 'Latino_Hispanic', 'East Asian', 'Southeast Asian', 'Indian', 'Middle Eastern']
gender_labels = ['Male', 'Female']
age_labels = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']

def predict_and_draw(image_path, default_max_size=800, chip_size=300, padding=0.25, save_output=False):
    img = dlib.load_rgb_image(image_path)
    original = cv2.imread(image_path)

    # Resize for face detection
    h, w, _ = img.shape
    resize_ratio = default_max_size / max(w, h)
    resized = dlib.resize_image(img, cols=int(w * resize_ratio), rows=int(h * resize_ratio))

    # Detect faces
    dets = cnn_face_detector(resized, 1)
    if len(dets) == 0:
        print("No faces found.")
        return

    faces = dlib.full_object_detections()
    for det in dets:
        faces.append(sp(resized, det.rect))

    # Get aligned face chips
    face_chips = dlib.get_face_chips(resized, faces, size=chip_size, padding=padding)

    for idx, (chip, det) in enumerate(zip(face_chips, dets)):
        input_tensor = transform(chip).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor).cpu().numpy().squeeze()

        # Slice predictions
        race_logits = output[:7]
        gender_logits = output[7:9]
        age_logits = output[9:18]

        # Apply softmax
        race = race_labels[np.argmax(np.exp(race_logits) / np.sum(np.exp(race_logits)))]
        gender = gender_labels[np.argmax(np.exp(gender_logits) / np.sum(np.exp(gender_logits)))]
        age = age_labels[np.argmax(np.exp(age_logits) / np.sum(np.exp(age_logits)))]

        # Scale bbox back to original size
        rect = det.rect
        l = int(rect.left() / resize_ratio)
        t = int(rect.top() / resize_ratio)
        r = int(rect.right() / resize_ratio)
        b = int(rect.bottom() / resize_ratio)

        # Draw and label
        cv2.rectangle(original, (l, t), (r, b), (0, 255, 0), 2)
        label = f"{race}, {gender}, {age}"
        cv2.putText(original, label, (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        print(f"Face {idx+1}: {label}")

    # Show or save result
    cv2.imshow("Prediction", original)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if save_output:
        out_path = os.path.splitext(image_path)[0] + "_predicted.jpg"
        cv2.imwrite(out_path, original)
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to an image')
    parser.add_argument('--save', action='store_true', help='Save result image with predictions')
    args = parser.parse_args()

    print("Using CUDA?:", dlib.DLIB_USE_CUDA)
    predict_and_draw(args.image, save_output=args.save)
