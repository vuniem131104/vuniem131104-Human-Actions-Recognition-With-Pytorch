import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import random_split, DataLoader, TensorDataset

height, width = 64, 64
data_dir = 'UCF50/UCF50'
seq_len = 20
classes_list = ['WalkingWithDog', 'TaiChi', 'Swing', 'HorseRace']

all_video_paths = os.listdir(data_dir)

def visualize_data():
    plt.figure(figsize=(20, 20))
    for idx, video_path in enumerate(all_video_paths):
        video = os.listdir(os.path.join(data_dir, video_path))[0]
        frames = cv2.VideoCapture(os.path.join(data_dir, video_path, video))
        ret, frame = frames.read()
        plt.subplot(10, 5, idx + 1)
        plt.imshow(frame)
        plt.title(video_path)
        plt.axis(False)
    plt.show()

# visualize_data()

tf = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])
def create_frames(video_path):
     video_reader = cv2.VideoCapture(video_path)
     video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
     skip_frame_window = max(int(video_frames_count/seq_len), 1)
     frames_list = []
     for i in range(seq_len):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, i * skip_frame_window)
        ret, frame = video_reader.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pilImage = Image.fromarray(frame_rgb)
        tensor_image = tf(pilImage)
        frames_list.append(tensor_image)
     result = torch.cat([t.unsqueeze(0) for t in frames_list], dim=0)
    #  result = result.transpose(0, 1).contiguous()
     video_reader.release()
     return result


def create_dataset():
    features = []
    labels = []
    video_file_paths = []

    for i, class_path in enumerate(classes_list):
        videos = os.listdir(os.path.join(data_dir, class_path))
        for video in videos:
            video_path = os.path.join(data_dir, class_path, video)
            frames_list = create_frames(video_path)
            if frames_list.size(0) == seq_len:
                features.append(frames_list)
                labels.append(i)
                video_file_paths.append(video_path)

    result_tensor = torch.cat([t.unsqueeze(0) for t in features], dim=0)
    labels = torch.tensor(labels)
    return result_tensor, labels, video_file_paths

features, labels, video_file_paths = create_dataset()

dataset = TensorDataset(features, labels)
train_size = int(0.75 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, drop_last=True)