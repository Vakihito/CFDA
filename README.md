# CST - Cluster Fusion Domain Adaptation: Improving Over Cross Domain Tasks

> Cluster Fusion Domain Adaptation implementation for the paper 'Cluster Fusion Domain Adaptation: Improving Over Cross-Domain Tasks.'

## Abstract

 > With the establishment of increasingly larger models defining the state of the art, there is a demand for massive amounts of data to train these architectures. However, in certain domains where data is scarce for training these models effectively, many studies resort to using cross-domain techniques. Traditional cross-domain methods aim to address this issue by extracting generic characteristics from different domains, thus reducing the relevance of the intrinsic characteristics of each domain. This approach reduces the semantic value that the domains have over each word. In this article, we propose a new method that aims to retain the intrinsic characteristics present in different domains through clustering techniques, by generating different representations for the same text. From this we employ fusion models to emphasize the characteristics shared between the domains. This novel approach has yielded significant improvements for cross-domain tasks, including question and answer (QA) tasks, text sentiment analysis, and image classification. Our results demonstrate the effectiveness of our method in leveraging domain-specific characteristics via cluste-wise domain adaptation and inter-domain adaptation via fusion models with adversarial training.

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

This project is divided into three different folders related to each task: image classification, text classification, and question answering.

Each folder has its own set of scripts located in the scripts <code>scripts</code> and a <code>.env</code> file to set the environment variables for each experiment.

### Training

To run the experiments and train using the method, just run:
 - <code>make run_text</code> - for performing the experiments for text classification
 - <code>make run_image</code> - for performing the experiments for image classification
 - <code>make run_qa</code> - for performing the experiments for question and anwering

This will run the scripts related to each step of the proposed method, creating all the models and obtaining their metrics.

A small data sample is located in the <code>data</code> folder within each task folder.

### Examples

Some examples of how to use the pipeline with the sampled data:
 - Example of how to use the text classification pipeline: [text notebook](https://colab.research.google.com/drive/1UL1CHFUrbTIhpD3asKJtS9zNw8Cjvg7h?usp=drive_open)
 - Example on how to use the question answering pipeline: [QA notebook](https://colab.research.google.com/drive/1ffXS3gJRv_YvbQN2YM39aKwNIouaqNFe#scrollTo=AD6R33_RLI6_)
 - Example on how to use the image classification pipeline: [Image notebook](https://colab.research.google.com/drive/1OvVzwbU8aMxm5M6ZceauDjZuB3g96bkm#scrollTo=kv8wqZWMTnm_)