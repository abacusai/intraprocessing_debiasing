# Intra-Processing Methods for Debiasing Neural Networks

[Intra-Processing Methods for Debiasing Neural Networks](https://arxiv.org/abs/2006.08564)\
Yash Savani, Colin White, Naveen Sundar Govindarajulu.\
_Advances in Neural Information Processing Systems 33 (2020)_.

## Three New Post-Hoc Techniques

In this work, we initiate the study of a new paradigm in debiasing research, _intra-processing_, which sits between in-processing and post-processing methods. Intra-processing methods are designed specifically to debias large models which have been trained on a generic dataset, and fine-tuned on a more specific task. We show how to repurpose existing in-processing methods for this use-case, and we also propose three baseline algorithms: random perturbation, layerwise optimization, and adversarial debiasing. All of our techniques can be used for all popular group fairness measures such as equalized odds or statistical parity difference. We evaluate these methods across three popular datasets from the [aif360](https://aif360.readthedocs.io/en/latest/modules/datasets.html) toolkit, as well as on the [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) faces dataset.

<p align="center">
<img src="images/FairAI_fig.png" alt="debias_fig" width="99%">
</p>
