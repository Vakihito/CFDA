# CST - Cluster Fusion Domain Adaptation: Improving Over Cross Domain Tasks

> Cluster Fusion Domain Adaptation implementation for the paper "Cluster Fusion Domain Adaptation: Improving Over Cross Domain Tasks" 

## Motivation

<p align="center">
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT2rm8544O9PT3YGTLc5ic-upTpAoTN-IVeWOl02o5kFQ&s" alt="Alt text">
</p>
<p align="center">Smile a litte</p>

## Requeirements

- pytorch >= 2.2.0
- pandas >= 2.0.3
- sklearn
- tqdm >= 4.66.4
- Pillow >= 9.4.0
- ktrain==0.37.2
- scikit-metrics==0.1.0
- datasets==2.13.0

To install requirements, run <code> pip install -r requirements.txt </code>

## Usage

This project is devided into three different folders related with each task, image classification, text classification and question answering.

Each folder has its own set of sripts located in the folder <code>scripts</code> and a <code>.env</code> to set the envirment variables of each experiment.

### Training

To run the experiments and train using the method just run <code>make train</code> on the folder that want. This will export the enviroment variables from the .env, and set for training.

A small data sample is located in the <code>data</code> folder.

