# EHAN
Interpretable Prediction of Vascular Diseases from Electronic Health Records via Deep Attention Networks published 2018 IEEE 18th International Conference on Bioinformatics and Bioengineering.

```python3
import tensorflow as tf
from model import EHAN


## Build model
config = {'n_features':vars_,
          'n_auxs':9,
          'steps':16,
          'hidden_units': 20
          }

mode = EHAN(config)

```


Interpretable Prediction of Vascular Diseases from Electronic Health Records via
Deep Attention Networks
Seunghyun Park1, You Jin Kim1, Jeong Whun Kim2, Jin Joo Park2, Borim Ryu2, Jung-Woo Ha*,1
1Clova AI Research, NAVER Corp., Seongnam 13561, Korea
2Seoul National University Bundang Hospital, Seongnam
