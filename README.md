#### EHAN (EHR History-based prediction using Attention Network)
Interpretable Prediction of Vascular Diseases from Electronic Health Records via Deep Attention Networks published 2018 IEEE 18th International Conference on Bioinformatics and Bioengineering.

This code was implemnted with tensorflow 2.x which is enable eager execution

```python3
import tensorflow as tf
from model import EHAN


## Build model
config = {'k':60, 'c_d':3000, 'c_m':500, 'm':300, 'n_d':128, 'n_h':128}

ehan = EHAN(config)
model = ehan.build_model()

```

#### interpretation
```python3
from interpretation import GradCAM

gcam = GradCAM(model)
gcam.generated_grad_cam(xs[0:1])

```


#### What is the difference between RETAIN and EHAN?
![image](https://user-images.githubusercontent.com/45510932/115355111-50d30c00-a1f5-11eb-9084-0aacb477dbe7.png)


#### Original article:
Interpretable Prediction of Vascular Diseases from Electronic Health Records via
Deep Attention Networks

Seunghyun Park1, You Jin Kim1, Jeong Whun Kim2, Jin Joo Park2, Borim Ryu2, Jung-Woo Ha*,1
1Clova AI Research, NAVER Corp., Seongnam 13561, Korea
2Seoul National University Bundang Hospital, Seongnam
