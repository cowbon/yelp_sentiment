# Papa_ting: Yelp Sentiment Analysis
Author: Ian-chin Wang
We used yelp challenge dataset: https://www.yelp.com/dataset
Due to the size of the data, we can not upload the data.
We used review.json in the dataset as our raw data.
## Flies
CS235ProjectReport.pdf: project final report
source_code/: source code of the project
models/: include source code of CNN and RNN models
data/: include data for classification
main.py: data preprocessing
clustering/clustering.py: cluster the data from classification with K-means and tf-idf.(we have toy data for execute, and there are commands for executing in the comments of the file)
clustering/dataset/test/: contain the toy data for clustering(contains positive and negative labeled reviews)
## How to run
We present a Dockerfile for preprocessing and classification, for classification launch
```bash
docker run -v "$(pwd):/data_mining -t <name of container> <args>
```
Usage of our front-end
`./main.py <preprocess/train/validate> -i <Input file or Directory> -t <cnn or lstm>`
* Select `preprocessing` for converting the dataset into sentence, it take original `review.json` in Yelp dataset as input
* Select `train` for training, it read all file containing labeled data under a certain directory(From input), and save the model to the disk.
