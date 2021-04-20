#### EHAN (EHR History-based prediction using Attention Network)
Interpretable Prediction of Vascular Diseases from Electronic Health Records via Deep Attention Networks published 2018 IEEE 18th International Conference on Bioinformatics and Bioengineering.

This code was implemnted with tensorflow 2.x which is enable eager execution

```python3
import tensorflow as tf
from model import EHAN


## Build model
config = {'k':10, # timestamp
          'c_m':100, # vocasize
          'm':4, # output vector size
          'n_d': 13
          'n_v': 13
          'n_h': 100 # unit of dense laye
          }

mode = EHAN(config)

```

#### interpretation
```python3
from interpretation import GradCAM

gcam = GradCAM(model)
gcam.generated_grad_cam(xs[0:1])

```


#### Original article:
Interpretable Prediction of Vascular Diseases from Electronic Health Records via
Deep Attention Networks

Seunghyun Park1, You Jin Kim1, Jeong Whun Kim2, Jin Joo Park2, Borim Ryu2, Jung-Woo Ha*,1
1Clova AI Research, NAVER Corp., Seongnam 13561, Korea
2Seoul National University Bundang Hospital, Seongnam
