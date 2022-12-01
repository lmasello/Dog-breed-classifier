import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True  # https://stackoverflow.com/a/23575424


def VGG16_predict(img, use_cuda=False):
    '''
    Use pre-trained VGG-16 model to obtain index corresponding to 
    predicted ImageNet class for image at specified path
    
    Args:
        img_path: path to an image
        
    Returns:
        Index corresponding to VGG-16 model's prediction
    '''
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    img_tensor = data_transforms(img)
    img_tensor = img_tensor.unsqueeze(0)
    # define VGG16 model
    VGG16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    # move model to GPU if CUDA is available
    if use_cuda:
        VGG16 = VGG16.cuda()    

    VGG16.eval()
    if use_cuda:
        img_tensor = img_tensor.cuda()
    output = VGG16(img_tensor)
    _, pred = torch.max(output, 1)
    return pred[0]


def dog_detector(img, predictor, use_cuda=False):
    """Determine whether a dog is detected in the image stored at img_path or not"""
    first_dog_index = 151
    last_dog_index = 268
    predicted = predictor(img)
    if use_cuda:
        predicted = predicted.cpu()
    return first_dog_index <= predicted.numpy() <= last_dog_index


def face_detector(img):
    """Return whether the image stored at img_path has a human face or not"""
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')    
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


def predict_breed_transfer(img, model, class_names):
    """Determine if the image stored at img_path contains a human or a dog and return the dog breed"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    img_tensor = transform(img).unsqueeze(0)
    model.eval()
    model_output = model(img_tensor)
    _, pred = torch.max(model_output, dim=1)
    return class_names[pred[0]] if pred[0] < len(class_names) else "Error: Prediction out of range"
