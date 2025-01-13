import os
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import faiss
from torchvision.models import resnet50, ResNet50_Weights


def vectorize_frames(frame_folder, output_file):
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    #model = models.resnet50(pretrained=True)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    vectors = []
    for frame in os.listdir(frame_folder):
        image = Image.open(os.path.join(frame_folder, frame)).convert("RGB")
        tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            vectors.append(model(tensor).squeeze().numpy())

    vectors = np.array(vectors).astype('float32')
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, output_file)

# Example usage
# vectorize_frames('data/frames/', 'embeddings/video_vectors/sample.index')
