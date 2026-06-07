import pickle
import streamlit as st
import pandas as pd
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity
import time



@st.cache_resource
def load_models():
  device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
  mtcnn = MTCNN(image_size=160)
  return model, mtcnn, device
model, mtcnn, device=load_models()
st.title("Which bollywood celebrity do you look like?")

userImg = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png"])
st.write("Note: The image should contain one face, better results if it is cropped to the face.")

@st.cache_data
def load_embeddings():
  with open('actor_embeddings.pkl', 'rb') as f:
    actor_embeddings = pickle.load(f)
    return actor_embeddings
actor_embeddings = load_embeddings() 

def feature_extractor(model, userImg):
  img = Image.open(userImg).convert('RGB')

  faces = mtcnn (img)
  if faces is None:
    st.error("Could not detect a face")
    st.stop()

  faces = faces.unsqueeze(0).to(device)
  with torch.no_grad():
    embedding = model(faces)
  return embedding.squeeze(0).cpu().numpy()
prediction=st.text("")


if userImg is not None:
   st.image(userImg, width=400)
   if st.button("Predict"):
      with st.spinner("Analyzing your face"):
         embeddings = feature_extractor(model, userImg)
         time.sleep(1)
         
         scores=[]

         for actor, actor_emb in actor_embeddings.items():
            score = cosine_similarity([embeddings], [actor_emb])[0][0]
            scores.append((actor, score))
         scores.sort(key=lambda x: x[1],reverse=True)
         top_scores = scores[:3]
         st.write("Your face looks like- ")
         for i in range(3):
            
            st.write(f"{top_scores[i][0]}-  {(top_scores[i][1]*100):.2f}% ")
      


        

   


