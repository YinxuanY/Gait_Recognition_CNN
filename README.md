# Gait Recognition
Gait recongnition using Gaitset Network
Make some improvement on the origianal code
- Modified loss function
- Extract ffine-grained feature
- Fused muti-layers features

The main network and loss function code is in ./model/network

## Prerequisites
- Python 3.6
- PyTorch 0.4+
- GPU

### Dataset & Preparation
Download [CASIA-B Dataset](http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp)

#### Pretreatment
```pretreatment.py```

### Train
Train a model by
```bash
python train.py
```
### Evaluation
Evaluate the trained model by
```bash
python test.py
```
Some of our trained model in /work
