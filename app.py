# Python In-built packages
from pathlib import Path
from ultralytics import YOLO
# External packages
import streamlit as st
import os
# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="Train Detection",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded"
)

logo_url = "./logo.png"
st.sidebar.image(logo_url)

videoType = st.sidebar.radio("Type de Lien", ("Stream Youtube", "Stream Lien Local"))
isYoutube = True if videoType == 'Stream Youtube' else False
sourceVideo = st.sidebar.text_input("Lien vers le flux video", value="https://www.youtube.com/watch?v=gFRtAAmiFbE")
source_radio = st.sidebar.radio(
    "Type de visualisation", settings.SOURCES_LIST,index=0)

# Listing all model paths in the weights folder
weight_files = os.listdir("./weigths")

model_path = st.sidebar.selectbox(
    "Choix du modèle YOLO",
    weight_files,
)

model_to = st.sidebar.selectbox(
    "Choix du Processeur de Calcul",
    ("cpu","mps","cuda")
)

# Load Pre-trained ML Model
#model_path = './weigths/yolov8n.pt'
try:
    model = YOLO(str("./weigths/"+model_path))
    model.to(model_to)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

confidence = float(st.sidebar.slider(
    "Confiance du Modèle", 25, 100, 35)) / 100

iou = float(st.sidebar.slider(
    "IOU", 0, 100, 30))/100

contrast = float(st.sidebar.slider(
    "Contrast", 0, 200, 100))/100

brightness = float(st.sidebar.slider(
    "Brightness", 0, 1000, 200))/10

# Créer un dictionnaire associant chaque classe à un nombre
class_mapping = {idx: class_name for idx, class_name in enumerate(settings.classNames)}
use_selected_classes = st.sidebar.checkbox("Sélection du choix des classes à détecter",value=True)
# Multiselect pour les classes
if use_selected_classes:
    selected_classes_ids = st.sidebar.multiselect("Selection des Classes à détecter", list(class_mapping.keys()), default=list(class_mapping.keys())[0], format_func=lambda x: class_mapping[x])
else:
    selected_classes_ids = list(class_mapping.keys())

resolutionVideo = st.sidebar.radio("Choix de la résolution de traitement et d'affichage (ne peut excéder la résolution de la source)", (240,360,480,720,1080),index=3)

displayOriginalVideo = st.sidebar.radio("Afficher le stream original ?", ("Non", "Oui"))
isDisplayOriginal = True if displayOriginalVideo == 'Oui' else False

if source_radio == settings.STREAM1:
    helper.visualize(confidence, model, selected_classes_ids, iou,sourceVideo,isYoutube,isDisplayOriginal,resolutionVideo,contrast,brightness)
elif source_radio == settings.STREAM2:
    helper.counter(confidence, model, selected_classes_ids, iou,sourceVideo,isYoutube,isDisplayOriginal,resolutionVideo,contrast,brightness)