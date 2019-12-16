FROM python:3.7
RUN pip install scikit-learn numpy scipy keras matplotlib tensorflow-gpu pandas nltk gensim
WORKDIR data_mining
COPY main.py data_mining
ENTRYPOINT ["python", "main.py"]
CMD ["train", "-i", "data", "-t", "lstm"]
