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

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN(image_size=160)

st.title("Which bollywood celebrity do you look like?")

userImg = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png"])

def feature_extractor(model):
  img = Image.open(userImg).convert('RGB')
  transform=transforms.ToTensor()
  device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

  faces = mtcnn (img)
  if faces is None:
    st.error("Could not detect a face")
    st.stop()

  faces = faces.unsqueeze(0).to(device)
  with torch.no_grad():
    embedding = model(faces)
  return embedding.squeeze(0).cpu().numpy()
prediction=st.text("")
Scores=[]

if userImg is not None:
   st.image(userImg, width=400)
   if st.button("Predict"):
      with st.spinner("Analyzing your face"):
         embeddings = feature_extractor(model)
         time.sleep(1)
         with open('actor_embeddings.pkl', 'rb') as f:
            actor_embeddings = pickle.load(f)
         scores=[]

         for actor, actor_emb in actor_embeddings.items():
            score = cosine_similarity([embeddings], [actor_emb])[0][0]
            scores.append((actor, score))
         scores.sort(key=lambda x: x[1],reverse=True)
         top_scores = scores[:3]
         st.write("Your face looks like- ")
         for i in range(3):
            
          st.write(f"{top_scores[i][0]}-  {round(top_scores[i][1]*100, 2)}% ")
      


        

   


