
# Melanoma diagnosis using CNNs

The objective of this project is to identify melanoma in images of skin lesions.

In particular, it contains the code for the [ISIC 2018 Skin Lesion Classification Task](https://challenge2018.isic-archive.com/task3/), which includes 7 possible diagnoses:

    AKIEC: Actinic keratosis / Bowen's disease (intraepithelial carcinoma)
    BCC: Basal cell carcinoma
    BKL: Benign keratosis (solar lentigo / seborrheic keratosis / lichen planus-like keratosis)
    DF: Dermatofibroma
    MEL: Melanoma
    NV: Melanocytic nevus
    VASC: Vascular lesion



Motivated by this challenge, the goal was to build an accurate model with good generalization
that can predict the diagnosis of any skin lesion. That is, given a dermoscopic or camera image,
we want to correctly classify it, especially when involving a possible melanoma diagnosis.

To accomplish that, we used the provided [HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) dataset with additional external data from
[BCN20000](https://arxiv.org/abs/1908.02288) and [PAD-UFES-20](https://data.mendeley.com/datasets/zr7vgbcyr2/1),
and implemented transfer learning using various CNN architectures (e.g., Inception-v3, ResNet-50, EfficientNet-B3).



The goal of the project was also to deploy the final models as a web application.  
The live version can be found at: 

## Demo

![Demo](https://s6.gifyu.com/images/Melanoma-Detection-App.gif)


  ## Table of contents

* [Introduction](#Introduction)
* [Training Process](#Training_Process)
* [Figures](#Figures)
* [Results](#Results)
* [Installation](#Installation)
* [Support](#Support)

  
## Introduction

In the United States, 5 million new cases of skin cancer are diagnosed every year.
Among those, melanoma is the most dangerous one,
being difficult to detect and causing serious problems once spread deeper.
On the other hand, localized melanoma has a very good prognosis. 
Therefore, detecting melanoma early is of utmost importance.
However, melanomas have many different shapes, sizes, and colors,
thus making it harder to provide comprehensive warning signs.
Convolutional neural networks provide significant progress in this field,
making it possible to detect melanomas and help doctors in their effort for a timely diagnosis.

  ## Training Process

- Downloaded and merged external datasets.
   - Balanced class distribution (see [Figures](#Figures)).
   - Improved generalization.
- Image augmentation with Keras Image Generator and custom functions, such as hair drawing (see [Figures](#Figures)).
- Experimented with several pre-trained models and optimizers.
- Loss function:  Weighted Categorical Cross-Entropy to account for class imbalance and disease severity.
- Evaluation metric: Weighted Accuracy, AUC.
- Trained each model for 50 epochs with early stopping and reduce learning rate on plateau callbacks.
- Implemented but with no improvement:
    - Augmentation & class balancing using GANs.
    - Bayesian Optimization to find the optimal CNN architecture.
## Figures

#### Class distribution before and after merging additional data.
![Class distribution](https://user-images.githubusercontent.com/60272607/123530073-376fa400-d6ff-11eb-85b6-cbe33d4abafe.png)

#### Image augmentation examples.
![Image augmentation](https://user-images.githubusercontent.com/60272607/123530117-a9e08400-d6ff-11eb-843e-ba5b00a09770.png)

  
## Results

With weighted accuracy as the evaluation metric, the models performed as follows:

| Base Models    | Training   | Validation |
| -------------- | ------ | -------- |
| ConvNet (baseline) | 67.0 % | 67.5 %  |
| NasNet-Large | 81.5 % | 68.5 % |
| DenseNet-201 | 78.5 % | 70.3 %  |
| ResNet-50 | 83.7 % |  75.7 % |
| InceptionResNet-v2 | 86.8 % | 78.8 %  |
| Inception-V3 | **95.3 %** | **83.6 %** |
| EfficientNet-B3 | **95.0 %** | **84.7 %** |


## Installation 
### Requirements

```
Numpy
Pandas
GPyOpt
OpenCV
Seaborn
Matplotlib
Tensorflow
Scikit-learn
Streamlit
tqdm
sla-cli
```


### 
Download the models' weights from the provided link and move them into the appropriate directory: 
[Models' Weights](https://drive.google.com/drive/folders/1n6wkDa-Qm6dtRgk37f01gPJmU7z7Iyew?usp=sharing)


### Download repository
```
$ git clone https://github.com/VKonstantakos/Melanoma-Diagnosis.git
```

After adding the models, run the following commands:

- Install all the required libraries.

```bash
pip install -r requirements.txt
```


- Run model app using Streamlit script.

```
streamlit run app.py
```
## Support

For support, email vkonstantakos@iit.demokritos.gr

  
