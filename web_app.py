import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from dog_breed_classifier import predict_breed_transfer, dog_detector, face_detector, VGG16_predict
from PIL import Image


def show_prediction(pic, model, class_names):
    """Show the model's prediction of a given picture"""
    image = Image.open(pic).convert('RGB')
    pred = predict_breed_transfer(image, model, class_names)
    if dog_detector(image, VGG16_predict) and not pred.startswith("Error"):
        st.write(f"Hey! This dog looks like a {pred}.")
    elif face_detector(image) and not pred.startswith("Error"):
        st.write(f"Hey! That human face looks like a {pred}!")
    else:
        st.write("The model failed to detect human or dog faces in that image." +
              "Please make sure you are using an image of a human or a dog.")
    st.image(image)


st.sidebar.title("Dog breed classifier")
pic = st.sidebar.file_uploader("Insert a dog picture")

class_names = pd.read_csv('class_names.csv').dog_breed.values
n_classes = len(class_names)
model_transfer = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
use_cuda = False
if use_cuda:
    model_transfer = model_transfer.cuda()
model_transfer.classifier[6] = nn.Linear(model_transfer.classifier[6].in_features, n_classes)
model_transfer.load_state_dict(torch.load('model_transfer.pt', map_location=torch.device('cpu')))

if pic:
    show_prediction(pic, model_transfer, class_names)