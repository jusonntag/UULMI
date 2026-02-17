# UULMI - Unilateral upper limb motor imagery classification

Implementation of the research paper:

**"Neurophysiological predictors of deep learning based unilateral upper limb motor imagery classification"**

This repository contains deep learning and machine learning implementations for classifying unilateral upper limb motor imagery using EEG data.

## Paper:
Sonntag J, Yu L, Wang X and Schack T (2025)  
- *Neurophysiological predictors of deep learning based unilateral upper limb motor imagery classification.* 

- Frontiers in Human Neuroscience, 19:1617748.

- https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2025.1617748/full


---
The raw data is publicly available here:
- https://pub.uni-bielefeld.de/record/3004681#

Please ensure you comply with the dataset's usage terms.

---

## Abstract

### Introduction:

Motor imagery-based brain-computer interfaces (BCIs) are a technique for decoding and classifying the intention of motor execution, solely based on imagined (rather than executed) movements. Although deep learning techniques have increased the potential of BCIs, the complexity of decoding unilateral upper limb motor imagery remains challenging. To understand whether neurophysiological features, which are directly related to neural mechanisms of motor imagery, might influence classification accuracy, most studies have largely leveraged traditional machine learning frameworks, leaving deep learning-based techniques underexplored.

### Methods:

In this work, three different deep learning models from the literature (EEGNet, FBCNet, NFEEG) and two common spatial pattern-based machine learning classifiers (SVM, LDA) were used to classify imagined right elbow flexion and extension from participants using electroencephalography data. From two recorded resting states (eyes-open, eyes-closed), absolute and relative alpha and beta power of the frontal, fronto-central and central electrodes were used to predict the accuracy of the different classifiers.

## Results:

The prediction of classifier accuracies by neurophysiological features revealed negative correlations between the relative alpha band and classifier accuracies and positive correlations between the absolute and relative beta band and classifiers accuracies. Most ipsilateral EEG channels yielded significant correlations with classifier accuracies, especially for the machine learning classifier.

## Discussion:

This pattern contrasts with previous findings from bilateral MI paradigms, where contralateral alpha and beta activity were more influential. These inverted correlations suggest task-specific neurophysiological mechanisms in unilateral MI, emphasizing the role of ipsilateral inhibition and attentional processes.


---
```bibtex
@article{sonntag2025neurophysiological,
  title={Neurophysiological predictors of deep learning based unilateral upper limb motor imagery classification},
  author={Sonntag, J and Yu, L and Wang, X and Schack, T},
  journal={Frontiers in Human Neuroscience},
  volume={19},
  pages={1617748},
  year={2025},
  doi={10.3389/fnhum.2025.1617748}
}
```
---
APA (6th ed.):
```bibtex
@article{Sonntag J, Yu L, Wang X and Schack T (2025) Neurophysiological predictors of deep learning based unilateral upper limb motor imagery classification. Front. Hum. Neurosci. 19:1617748. doi: 10.3389/fnhum.2025.1617748}

