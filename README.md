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


In order to train, execute
```console
python main.py
```


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

## **Report**
The report can be found in the repository.


## Presentation Link 
[![Presentation](https://img.youtube.com/vi/lpNlS8H6Pww/0.jpg)](https://youtu.be/lpNlS8H6Pww)

## **Authors**
- [Hızır Can Bayram](https://github.com/hizircanbayram)
- [Emre Çetin](https://github.com/emrectn)


## **Acknowledgement** 
This project is carried out for Trustable AI Course at Istanbul Technical University during 2021 fall semester.
