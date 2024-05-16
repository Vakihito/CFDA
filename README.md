# CST - Cluster Fusion Domain Adaptation: Improving Over Cross Domain Tasks

> Cluster Fusion Domain Adaptation implementation for the paper "Cluster Fusion Domain Adaptation: Improving Over Cross Domain Tasks" 

## Motivation

<p align="center">
  <img src="https://redefineschool.files.wordpress.com/2017/05/jordan-peterson.png" alt="Alt text">
</p>
<p align="center">Smile a little</p>

## Requeirements

- transformers==4.37.2
- ktrain==0.41.3
- torch >= 2.0.0
- pandas >= 2.0.3
- tqdm >= 4.66.4
- Pillow >= 9.4.0
- scikit-metrics==0.1.0
- datasets==2.13.0
- python-dotenv
- scikit-learn >= 1.3
- evaluate==0.2.2

To install requirements, run <code> pip install -r requirements.txt </code>

## Usage

This project is devided into three different folders related with each task, image classification, text classification and question answering.

Each folder has its own set of sripts located in the folder <code>scripts</code> and a <code>.env</code> to set the envirment variables of each experiment.

### Training

To run the experiments and train using the method just run:
 - <code>make run_text</code> - for performing the experiments for text classification
 - <code>make run_image</code> - for performing the experiments for image classification
 - <code>make run_qa</code> - for performing the experiments for question and anwering

A small data sample is located in the <code>data</code> folder.

### Examples
Some examples of how to use the pipeline :
 - Example on how to use the text classification pipeline : [text notebook](https://colab.research.google.com/drive/1UL1CHFUrbTIhpD3asKJtS9zNw8Cjvg7h?usp=drive_open)
 - Example on how to use the question answering pipeline : [QA notebook](https://colab.research.google.com/drive/1ffXS3gJRv_YvbQN2YM39aKwNIouaqNFe#scrollTo=AD6R33_RLI6_)

