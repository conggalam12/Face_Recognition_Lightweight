import onnx
import onnxruntime
import cv2 as cv
import numpy as np
from torchvision import transforms
import torch

def pre_processing(img_path):
    image = cv.imread(img_path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((112,96)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
    image = trans(image)
    image = image.unsqueeze(0)
    return image.cpu().numpy()


path_model = "/home/congnt/face/custom.onnx"

img_path = "/home/congnt/face/face_recognition/test_img/img/person_2/img_2.jpg"
input = pre_processing(img_path)

session = onnxruntime.InferenceSession(path_model)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
output = session.run([output_name], {input_name:input})
output = output[0]
print(1)
