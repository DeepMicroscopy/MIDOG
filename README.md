![MIDOG logo](img/cropped-midog_at_miccai-2.png)

Welcome to the MIDOG github repository. Here you can find code of our own evaluations and a dockered reference algorithm for mitotic figures to use as a template.

If you haven't registered yet and want to take part in the challenge, please register [here](https://imi.thi.de/midog/register/).

The folder [DomainShiftQuantization](https://github.com/DeepPathology/MIDOG/tree/main/DomainShiftQuantification) contains code of our MIDL paper [Quantifying the Scanner-Induced Domain Gap in Mitosis Detection](https://arxiv.org/pdf/2103.16515.pdf).

The folder [baseline](https://github.com/DeepPathology/MIDOG/tree/main/baseline) contains code for training a RetinaNet architecture that was extended by a domain adversarial path. This was submitted as baseline to the MIDOG challenge.

The repository [MIDOG_reference_docker](http://github.com/DeepPathology/MIDOG_reference_docker) contains a reference docker container with the baseline algorithm for the MIDOG challenge.

We also make available the [evaluation container for the challenge](https://github.com/DeepPathology/MIDOG_evaluation_docker) as a separate repository  (without the ground truth data itself). 

And finally, the folder [databases](https://github.com/DeepPathology/MIDOG/tree/main/databases) contains all databases (annotations) of the MIDOG challenge training set.
 
