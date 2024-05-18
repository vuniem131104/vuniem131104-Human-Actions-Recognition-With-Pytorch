from pytube import YouTube
import cv2 
from PIL import Image
from collections import deque
import torch
from torchvision import transforms
from model import Model, Model2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Model2().to(device)
model.load_state_dict(torch.load('model.pt'))

seq_len = 20

classes_list = ['WalkingWithDog', 'TaiChi', 'Swing', 'HorseRace']

tf = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

# YouTube video URL
url = 'https://www.youtube.com/watch?v=8u0qjmHIOcE'

# Create a YouTube object
yt = YouTube(url)

# Get the highest resolution stream
stream = yt.streams.get_highest_resolution()

# Download the video
stream.download()

def predict_on_video(video_file_path, output_file_path, SEQUENCE_LENGTH):
    '''
    This function will perform action recognition on a video using the LRCN model.
    Args:
    video_file_path:  The path of the video stored in the disk on which the action recognition is to be performed.
    output_file_path: The path where the ouput video with the predicted action being performed overlayed will be stored.
    SEQUENCE_LENGTH:  The fixed number of frames of a video that can be passed to the model as one sequence.
    '''

    video_reader = cv2.VideoCapture(video_file_path)

    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'),
                                   video_reader.get(cv2.CAP_PROP_FPS), (original_video_width, original_video_height))

    frames_queue = deque(maxlen = seq_len)

    predicted_class_name = ''

    while video_reader.isOpened():

        ok, frame = video_reader.read()

        if not ok:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pilImage = Image.fromarray(frame_rgb)
        tensor_image = tf(pilImage)
        frames_queue.append(tensor_image)
        result_tensor = torch.cat([t.unsqueeze(0) for t in frames_queue], dim=0)
        prob = 0
        if len(frames_queue) == seq_len:

            predicted_labels_probabilities = model(result_tensor.unsqueeze(dim=0).to(device))

            predicted_label = predicted_labels_probabilities.argmax(dim=1)
            sm = predicted_labels_probabilities.softmax(dim=1)
            sm = sm.detach().cpu().squeeze()
            prob = torch.max(sm)
            predicted_class_name = classes_list[predicted_label.item()]

        cv2.putText(frame, f'{predicted_class_name} {prob:.4f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

        video_writer.write(frame)

    video_reader.release()
    video_writer.release()

input_video_file_path = 'Test Video.mp4'
output_video_file_path = 'Result Video.mp4'

# Perform Action Recognition on the Test Video.
predict_on_video(input_video_file_path, output_video_file_path, seq_len)