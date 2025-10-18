
# download model from aws s3
import streamlit as st
import boto3
import os
import torch
from transformers import pipeline
s3 = boto3.client("s3", region_name="ap-south-1")

local_path = "tinybert-sentiment-analysis_1"
s3_prefix = "model deployment/tinybert-sentiment-analysis/"
bucket_name = "ml-shubham"

def download_dir(local_path, s3_prefix):
    os.makedirs(local_path, exist_ok=True)
    paginator = s3.get_paginator("list_objects_v2")
    for result in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
        if "Contents" in result:
            for obj in result["Contents"]:
                s3_key = obj["Key"]
                local_file_path = os.path.join(local_path, os.path.relpath(s3_key, s3_prefix))
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                s3.download_file(bucket_name, s3_key, local_file_path)

st.title("machine learning model deployment at server")
button = st.button("download model")
if button:
    with st.spinner("downloading...please wait!"):
        download_dir(local_path, s3_prefix)
    #st.success(" Download completed successfully!")
    #st.toast(" Model downloaded!", icon="")   # popup notification
    #st.balloons() 

text = st.text_area("type your review here...","type...")
predict = st.button("predict")
device = torch.device("cuda")if torch.cuda.is_available() else torch.device("cpu")
classifier = pipeline("text-classification",model = "tinybert-sentiment-analysis_1",device=device)

if predict:
    with st.spinner("predicting...please wait!"):
        output =  classifier(text)
        st.write(output)



