# Build from the official TensorFlow 2 image
#FROM nvcr.io/nvidia/tensorflow:20.08-tf2-py3
FROM tensorflow/tensorflow:latest-gpu-jupyter

RUN pip install --upgrade pip &&\
    pip install --no-cache-dir pandas &&\
    pip install --no-cache-dir matplotlib &&\
    pip install --no-cache-dir lime &&\
    pip install --no-cache-dir -U scikit-learn &&\
    pip install --no-cache-dir gensim &&\
    pip install --no-cache-dir nltk