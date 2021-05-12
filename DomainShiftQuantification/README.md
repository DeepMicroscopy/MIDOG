### Quantifying the Scanner-Induced Domain Gap in Mitosis Detection

This is the code used in the experiments reported about in our paper:
> M. Aubreville et al.: Quantifying the Scanner-Induced Domain Gap in Mitosis Detection (MIDL 2021)

Dependencies:
- MIDOG dataset (available through: [https://zenodo.org/record/4643381#.YGGOYi221mB])
- MIDOG.sqlite database file (from this repository)
- SlideRunner_dataAccess == 1.0.1 (and dependencies)
- fast.ai == 1.0.62
- A model pre-trained on the alternative labels set of TUPAC16: https://github.com/DeepPathology/TUPAC16_AlternativeLabels
  (place the output produced by learn.export() under RetinaNet-TUPAC_AL-OrigSplit-512s.pkl)

## Training process

### 1. Training of the RetinaNet

First run RetinaNet-MIDOG-flex.py to train the model. Parameters are as follows:

python3 RetinaNet-TUPAC_AL-OrigSplit-512s.pkl <source_scanner> <run>
  source_scanner: String representing the scanner that shall be used for training.
                  XR: Hamamatsu XR
                  S360: Hamamatsu S360
                  CS2: Aperio ScanScope CS2
  run: Number of the training run (1,2,3,...)


### 2. Training of the Domain Classifier from the last layer of the RetinaNet classifier

python3 RetinaNet-DomainShift-MIDOG-PAD.py <scanner_source_domain> <scanner_target_domain> <run>

Here, source_scanner_domain and scanner_target_domain are numbers:
                1: Hamamatsu XR
                2: Hamamatsu S360
                3: Aperio ScanScope CS2
                4: Leica GT450

where run is a number indicating the training run of the original RetinaNet

### 3. Inference

Inference can be found in the notebook Inference-MIDOG.ipynb

### 4. Evaluation

The evaluation is found in the notebook Evaluation.ipynb




