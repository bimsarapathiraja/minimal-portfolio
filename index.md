# Portfolio
---
## Work Experience

### Machine Learning Engineer (Part-time remote) - <a href='https://www.promiseq.com/'>promiseQ</a>

promiseQ uses advanced real-time video analysis, object detection and tracking to reduce the cost and time wasted associated to false alarms.

**My contributions** 
* Planning next training iterations
* Integrating the system with <a href='https://neptune.ai/' > Neptune.ai </a>
* Improving object tracking module
* Filtering false positives using rule based techniques
* Image augmentation to reduce class imbalance 

---

## [Self-Driving-Car-Stage-II] Multi-Sensor based Dynamic Object Detection, Tracking, and Trajectory Prediction

The final year project of the degree program and our project is based on dynamic object detection, tracking, trajectory prediction, signal light identification and data collection using LiDAR and camera. 
- Detection and tracking - <a href='https://arxiv.org/abs/2006.11275'> Centerpoint </a> 
  - detection MAP **80.2%**
  - detection FPS **16**
  - tracking  AMOTA **0.65**

- Signal light identification - Simple CNN-LSTM model
- Trajectory prediction - based on <a href='https://drive.google.com/file/d/1Ksq7X5dzouMV2jG1QYcgWzpUl2dKWUDW/view'> ReCoAt (CVPR2021 Workshop on Autonomous Driving) </a>
<center><img src="images/fyp3.png"/></center>

### 3D object Detection and Tracking

<center><img src="images/centerpoint.png"/></center>
### Trajectory Prediction

Paper submitted to <a href='https://www.ieee-itsc2022.org/'> IEEE-ITSC </a> - **Class-Aware Attention for Multimodal Trajectory Prediction**

<!-- [![Run in Google Drive](https://img.shields.io/badge/Drive-View%20in%20Google%20Drive-blue?logo=googledrive&logoColor=#4285F4)](https://drive.google.com/file/d/1fg3wfGAm5fC2huAs-Va7XjCXAunanhGb/view?usp=sharing) -->

[![Open Research Poster](https://img.shields.io/badge/PDF-Open%20Research%20Paper-blue?logo=adobe-acrobat-reader&logoColor=white)](pdf/IEEE_ITSC.pdf)


**Abstract**

<div style="text-align: justify"> Abstract—Predicting the possible future trajectories of the surrounding dynamic agents is an essential requirement in autonomous driving. These trajectories mainly depend on the surrounding static environment, as well as the past movements of those dynamic agents. Furthermore, the multimodal nature of agent intentions makes the trajectory prediction problem more challenging. All of the existing models consider the target agent as well as the surrounding agents similarly, without considering the variation of physical properties. In this paper, we present a novel deep-learning based framework for multimodal trajectory prediction in autonomous driving, which considers the physical properties of the target and surrounding vehicles such as the object class and their physical dimensions through a weighted attention module, that improves the accuracy of the predictions. Our model has achieved the highest results in the nuScenes trajectory prediction benchmark, out of the models which use rasterized maps to input environment information. Furthermore, our model is able to run in real-time, achieving a high inference rate of over 300 FPS. </div>
<br>

**Sample Results**
<center><img src="images/paper-viz.png"/></center>

**Quantitative results for nuScenes dataset**

$MinADE_5$ - 1.67m
$MinFDE_1$ - 8.43m

---

## Computer Vision

### CS231n: Convolutional Neural Networks for Visual Recognition

My complete implementation of assignments and projects in [***CS231n: Convolutional Neural Networks for Visual Recognition***](http://cs231n.stanford.edu/2021/) by Stanford (Spring, 2021).

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/)

**Implementing CNN image classification module using Numpy:** 
An image classification model implementing with fully connected networks, non linear activations, batch normalization, dropout and convolutional networks including back propagation ([GitHub](https://github.com/)).

**Image Captioning:** An image captioning model with vanilla RNNs, LSTM and Transformer network. RNN and LSTM were implemented from scratch using  numpy including backpropagation. Attention, Multi-head attention and Transformer were implemented using Pytorch ([GitHub](https://github.com/chriskhanhtran/CS224n-NLP-Assignments/tree/master/assignments/a3)).

<center><img src="images/cs231n.png"/></center>

**GAN:** Implementing Vanilla GAN, Least Square GAN and Deep Convolutional GAN (DCGAN). 

**Network Visualization:** Visualizing a pretrained model using saliency maps, fooling images and class visualization.

<center><img src="images/saliency2.png"/></center>

---
## Natural Language Processing

### CS224n: Natural Language Processing with Deep Learning

My complete implementation of assignments and projects in [***CS224n: Natural Language Processing with Deep Learning***](http://web.stanford.edu/class/cs224n/) by Stanford (Winter, 2019).

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/)

**Neural Machine Translation:** An NMT system which translates texts from Spanish to English using a Bidirectional LSTM encoder for the source sentence and a Unidirectional LSTM Decoder with multiplicative attention for the target sentence ([GitHub](https://github.com/)).

**Dependency Parsing:** A Neural Transition-Based Dependency Parsing system with one-layer MLP ([GitHub](https://github.com)).

<center><img src="images/nlp.png"/></center>

---

## Internship Projects

Company: [Creative Software](https://www.creativesoftware.com/)
### Corrosion Detection using Semantic Segmentation

Corrosion Detection for industrial environment using semantic segmentation. I used U-Net model for semantic segmentation. I completed writing the model, testing and all the training. Using a combination of focal loss and dice loss increased the accuracy significantly and using lot of augmentations reduced false positives.

<center><img src="images/unet.png"/></center>

Synthetic data generation is also done using Unity 3D since the real image dataset was not enough.

<center><img src="images/synthe.png"/></center>


### Object Detection in Industrial Environment

Object detection model was trained using Detectron2 for idenitifying industrial objects like gauges, motors, valves, pumps etc. 

<center><img src="images/maskrcnn.png"/></center>

---

### Garment ReConstruction - NeurIPS Challenge

3D Texture garment reconstruction using CLOTH3D dataset and SMPL body parameters. PyMesh, Open3d, Meshlab, MeshlabXML, Pytorch Geometric libraires were used.

Only the data preprocessing part is done. The model is yet to be implemented.

**Subsampling points**
<center><img src="images/subsampling.png"/></center>

**Non-rigid Iterative Closest Point (ICP)**
<center><img src="images/nicp.png"/></center>

**Custom maxpooling**
<center><img src="images/maxpool.png"/></center>

### Deep Surveilance System (DSS) - SLIOT Challenges

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/)

<div style="text-align: justify"> Deep Surveillance System, an IoT device which is triggered by threatening sounds to activate the camera. The product included hardware, sensors, ML model, web based UI as well. Urban 8K sound dataset and TensorFlow were used for model training. Implemented using Raspberry Pi, OpenCV and Azure. I involved in model wrting, training and hardware implementation.

DSS won 2nd place in the open category of Sri Lanka IoT competition (SLIOT). </div>

---
### Detect Spam Messages: TF-IDF and Naive Bayes Classifier

[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](projects/detect-spam-nlp.html)
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/chriskhanhtran/detect-spam-messages-nlp/blob/master/detect-spam-nlp.ipynb)

<div style="text-align: justify">In order to predict whether a message is spam, first I vectorized text messages into a format that machine learning algorithms can understand using Bag-of-Word and TF-IDF. Then I trained a machine learning model to learn to discriminate between normal and spam messages. Finally, with the trained model, I classified unlabel messages into normal or spam.</div>
<br>
<center><img src="images/detect-spam-nlp.png"/></center>
<br>

---
## Data Science

### Credit Risk Prediction Web App

[![Open Web App](https://img.shields.io/badge/Heroku-Open_Web_App-blue?logo=Heroku)](http://credit-risk.herokuapp.com/)
[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](https://github.com/chriskhanhtran/credit-risk-prediction/blob/master/documents/Notebook.ipynb)
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/chriskhanhtran/credit-risk-prediction)

<div style="text-align: justify">After my team preprocessed a dataset of 10K credit applications and built machine learning models to predict credit default risk, I built an interactive user interface with Streamlit and hosted the web app on Heroku server.</div>
<br>
<center><img src="images/credit-risk-webapp.png"/></center>
<br>

---
### Kaggle Competition: Predict Ames House Price using Lasso, Ridge, XGBoost and LightGBM

[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](projects/ames-house-price.html)
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/chriskhanhtran/kaggle-house-price/blob/master/ames-house-price.ipynb)

<div style="text-align: justify">I performed comprehensive EDA to understand important variables, handled missing values, outliers, performed feature engineering, and ensembled machine learning models to predict house prices. My best model had Mean Absolute Error (MAE) of 12293.919, ranking <b>95/15502</b>, approximately <b>top 0.6%</b> in the Kaggle leaderboard.</div>
<br>
<center><img src="images/ames-house-price.jpg"/></center>
<br>

---
### Predict Breast Cancer with RF, PCA and SVM using Python

[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](projects/breast-cancer.html)
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/chriskhanhtran/predict-breast-cancer-with-rf-pca-svm/blob/master/breast-cancer.ipynb)

<div style="text-align: justify">In this project I am going to perform comprehensive EDA on the breast cancer dataset, then transform the data using Principal Components Analysis (PCA) and use Support Vector Machine (SVM) model to predict whether a patient has breast cancer.</div>
<br>
<center><img src="images/breast-cancer.png"/></center>
<br>

---
### Business Analytics Conference 2018: How is NYC's Government Using Money?

[![Open Research Poster](https://img.shields.io/badge/PDF-Open_Research_Poster-blue?logo=adobe-acrobat-reader&logoColor=white)](pdf/bac2018.pdf)

<div style="text-align: justify">In three-month research and a two-day hackathon, I led a team of four students to discover insights from 6 million records of NYC and Boston government spending data sets and won runner-up prize for the best research poster out of 18 participating colleges.</div>
<br>
<center><img src="images/bac2018.JPG"/></center>
<br>

---
## Filmed by me

[![View My Films](https://img.shields.io/badge/YouTube-View_My_Films-grey?logo=youtube&labelColor=FF0000)](https://www.youtube.com/watch?v=vfZwdEWgUPE)

<div style="text-align: justify">Besides Data Science, I also have a great passion for photography and videography. Below is a list of films I documented to retain beautiful memories of places I traveled to and amazing people I met on the way.</div>
<br>

- [Ada Von Weiss - You Regret (Winter at Niagara)](https://www.youtube.com/watch?v=-5esqvmPnHI)
- [The Weight We Carry is Love - TORONTO](https://www.youtube.com/watch?v=vfZwdEWgUPE)
- [In America - Boston 2017](https://www.youtube.com/watch?v=YdXufiebgyc)
- [In America - We Call This Place Our Home (Massachusetts)](https://www.youtube.com/watch?v=jzfcM_iO0FU)

---
<center>© 2020 Khanh Tran. Powered by Jekyll and the Minimal Theme.</center>