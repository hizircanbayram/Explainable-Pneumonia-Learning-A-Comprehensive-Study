# Explainable-Pneumonia-Learning-A-Comprehensive-Study


## **Abstract** 
> *Pneumonia causes certain deaths every year all over the world. Even though early diagnosis
might be likely by various imaging techniques such as CT, CXR, it is a vast amount of time for radiologists
to analyze medical images by conventional techniques since it requires certain amount of experts and deep
knowledge. However, number of experts in this area makes early diagnosis inaccessible for everyone. To
eliminate this, CAD systems can help average radiologists to catch expert-level performance in terms of
having more meaningful and interpretable results from these imaging techniques. In this work, we explored
pneumonia detection in chest X-ray images and how CAD systems can help us to have an interpretable
outcome. So, our work’s main focus is not confined with diagnosis. We rather explored if we could interpret
X-ray images in a meaningful manner, how the interpretability of these models can be boosted up, when
considering robustness of the models especially. To do so, we performed various experiments such as
training different deep net architectures, measuring how they differ from unlearnt deep net architectures,
how lung segmentation affect interpretability of these models etc. We showed that creating interpretable
models is possible and we can outperform these models’ performances using different techniques.*

## **Dataset**
RSNA dataset is formed by Radiological Society of North America(RSNA) by benefiting ChestXray-14 dataset [5] in order to diagnose thoracic diseases. Where this dataset differs from ChestXray-14 is that the images are shared in DICOM format and the dataset is arranged so that it can be used to diagnose pneumonia in particular. DICOM format consists of pixel values of X-ray images as well as the metadata that belongs to the images. The dataset contains 26.684 images where each one is labeled either “Not Normal/No Lung Opacity”, “Normal”, “Lung Opacity”. “Normal” label represents neither pneumonia nor any other conditions detected in the image. “Not Normal/No Lung Opacity” represents opacity in the X-ray image is caused by a condition other than
pneumonia, hence it is related to any other conditions. Finally “Lung Opacity” represents opacity in the image caused by pneumonia. Opaque areas are segmented by radiologists using bounding box technique. More than one bounding box can be captured in the images. Images labeled as “Lung Opacity” are assumed as pneumonia. In other words, the dataset consists of 9555 pneumonia positive images as well as 20672 pneumonia negative images. Distribution of the labels are shown in the table below.

| Label               	| Number of Images 	|
|-----------------------------	|-------	|
| Normal 	|  8851 	|
| Lung Opacity    	| 9555  	|
| No Lung Opacity / Not Normal 	| 11821 	|

## **Results**

This section presents an overview of our experimental findings and a preliminary analysis of each contribution individually. Classification was performed using different models and different labeling techniques as stated in the Materials and Method section of the report in the repository. We benchmarked these models for comparison. Lung Segmented model was trained with our dataset that consists of various pneumonia and COVID-19 datasets. Interpretability of our models was measured using Dice coefficient and we benchmarked different saliency map, deep architecture and lung segmentation supported techniques in order to see the big picture. Tthe table below presents results for our classification scenario for different networks, different labeling techniques and lung segmented support. Models using segmented CXR images presented better results than the models that used segmented images. There was no superior relation between models with or without lung segmented support. Both settings were on par in the normal class. In all cases, the models using segmented images performed similar, considering the selected metric. That result alone might discourage the usage of segmentation in practice in terms of classification.

| Models |            Sensitivity |  Specificity |  Accuracy | 
|-----------------------------	| -------	| -------	| -------	|
| Random DenseNet   |               0 | 1|  0,507| 
| Random ResNet      |              0 | 1 | 0,507| 
| Normal DenseNet    |          0,903 | 0,939 | 0,921 | 
| Normal ResNet         |       0,895 | 0,954|  0,924 | 
| Lung Segmented DenseNet   |     0,9 | 0,94 | 0,916 | 
| Lung Segmented ResNet    |      0,91 | 0,93 | 0,92 | 

When using the GradCam and LayerCam methods, the
pixel values are set to 0 or 1 according to the threshold
value. Images were turned gray without setting this threshold value. The threshold value have a critical role in measuring
the Gcam and Layercam methods. The success rate varies
according to the threshold value determined. Performing
performance analysis using equal and frequent intervals is
very costly. Testing a threshold value with each model takes
approximately 2 hours. Since there are 6 models, this period
takes 13-14 hours. In order to minimize this cost, 5000
images were analyzed. We aimed to determine the threshold
value ranges, start and end points of the images according to
the pixel distributions. We observed that there is a distribution
between 100-200 pixel values according to the histogram
distribution. The rate of change was determined as 25 pixels.
Therefore, analysis was performed using values of 100, 125,
150, 175 and 200. The histogram distribution is shown on the
4. Since the background color is black in lung images, the
number of pixels in the 0-50 range is quite high. Because of
this, the graph was corrupted. To prevent this, the 0-50 range
has been removed from the chart.

![histogram_dist_1](https://user-images.githubusercontent.com/23126077/156029909-c296135a-ec97-4a74-9787-1a25c43417f1.jpg)

After the threshold value is determined, the area below the
threshold value is marked as 0. The area above the threshold
value is marked as 1. In the picture, 0s are represented by
purple pixels. 1 is represented by yellow pixels. After the
threshold value is applied, the middle color values disappear.
It is clearly visible in the figure below. The degree of importance
in the color distribution is shown in the Figure 5. The most
important regions are shown in red, the least important regions in blue.

![th_eleminate_1](https://user-images.githubusercontent.com/23126077/156030116-743e63e1-2734-4530-8542-f21228e69958.jpg)

The figure below shows the difference between the randomly labeled model and the correctly labeled model. Since learning did not occur from the randomly labeled model,random models the class heat maps almost did not focus on the lung area at all.

![normal_vs_random_1](https://user-images.githubusercontent.com/23126077/156030405-76a2b364-5b6a-4712-8281-e6205ac0cba5.jpg)

Figures 7 and 8 present the Grad-CAM
heatmaps for our classification scenario.  We can notice that the models created CXR images focused primarily in the lung area in figure 7 and 8. It is seen
that Gradcam focuses on a narrower area than Layercam.

| Sample 1  | Sample 2 |
| ------------- | ------------- |
| ![labeled_gcam_layercam_2](https://user-images.githubusercontent.com/23126077/156030409-ca4109c9-cb9a-4002-bd54-d2041a518c1d.jpg)              |    ![labeled_gcam_layercam_1](https://user-images.githubusercontent.com/23126077/156030413-f5d9b55b-3fb1-421e-984a-a8574605fc35.jpg)           |


## **Discussion**
We didn’t see a major difference between using DenseNet
and ResNet for a wide variety of metrics such as sensitivity,
specificity, accuracy and dice coefficient, we recommended
using DenseNet since it is stated in the literature and our
scores validated this fact once more though. However, we
determined a significant improvement when lung segmenta-
tion model was used. This is because models trained without
lung segmentation model somestimes perturbed gradients’
attention outside of the lung area. When we colored non-
lung pixels black, model could not learn any misinformation
from these regions since there was not any action that can
be diffentiated, turning its attention into lung areas. Thus, it
led the model to force itself learn features inside lung area.
Another important observation is choosing threshold value
affects the result of our saliency maps. We chose a threshold
value to turn our RGB saliency map into a binary image so
that we can benchmark our maps and ground truth bounding
boxes in terms of dice coefficient. Choosing this threshold
affected how our maps and bounding boxes overlap and
union. Since pixel values are between 0 and 255, we needed
to choose some good representative points. To do so, we
examined where pixel intensivities are higher in our saliency
maps. Focusing those areas, we detected some threshold
values and generated our binary saliency maps. We detected
a threshold value of nearly 150 is promising in terms of dice
coefficient.


## **Usage**

In order to train, execute
```console
python main.py
```

## **Report**
The report can be found in the repository.


## Presentation Link 
[![Presentation](https://img.youtube.com/vi/lpNlS8H6Pww/0.jpg)](https://youtu.be/lpNlS8H6Pww)

## **Authors**
- [Hızır Can Bayram](https://github.com/hizircanbayram)
- [Emre Çetin](https://github.com/emrectn)


## **Acknowledgement** 
This project is carried out for Trustable AI Course at Istanbul Technical University during 2021 fall semester.
