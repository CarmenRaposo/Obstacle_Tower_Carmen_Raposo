# Obstacle Tower Carmen Raposo

We face the **Obstacle Tower Challenge**. Our project will consist of the following steps:

1. PPO algorithm implementation. Study of how certains hyperparameters can affect training.

2. Image classifier, from *Obstacle Tower Challenge* winner, Alex Nichol (http://github.com/unixpickle/obs-tower2)

3. RND architecture.

![image](https://github.com/CarmenRaposo/Obstacle_Tower_Carmen_Raposo/blob/master/General_Diagram.png?raw=true)


## Getting Started

You can download this project and reproduce it locally by folllowing next steps.


### Prerequisites

The next libraries versions will be needed in order to reproduce the project:

```
baselines==0.1.6 
gym==0.17.2 
mlagents==0.10.1 
mlagents-envs==0.10.1 
numpy==1.18.5 
obstacle-tower-env==3.1 
Pillow==5.4.1 
plotly==4.8.0 
stable-baselines==2.10.0 
tensorboard==1.7.0
tensorflow==1.7.0 
torch==1.5.0
```

## Running the different steps

How to train indepently each of the three step mentioned before.

### 1. PPO algorithm study

To train the PPO algorithm in study version in order to compare the results from trainings with different hyperparameters values:

```
python main.py --study
```

If some fixed hyperparameter value needs to be changed, it can be changed as a parse argument of the execution line.

```
python main.py --value-loss-coef 0.4 --num-steps 1024
```

If any value of the hyperparameters are set in the execution line, default values from [utils.py](utils.py) script will be used.

### 2. Image classifier

First of all, image classifier must be trained. To do that, classifier folder has to be used, following the instructions explained on Alex Nichol [repository](https://github.com/unixpickle/obs-tower2).

To train with the image classifier running, execution line is as follows:
```
python main.py --features
```

### 3. RND Architecture

We will implement RND architecture since it performs a good score on *Montezuma's Revenge* game, on which *Obstacle Tower* is based.

![image](https://github.com/CarmenRaposo/Obstacle_Tower_Carmen_Raposo/blob/master/RND_Diagram.png?raw=true)

```
python main.py --features --rnd
```


## Test

In order to test how are previous trained agents work, we can use the pretrained models as follows:

```
python main.py --test --pretrained model [pretrained model file's route] --features
```

## Built With

* [PyCharm](https://www.jetbrains.com/es-es/pycharm/) Python IDE


## Acknowledgments


* [Obstacle Tower Challenge paper](https://arxiv.org/pdf/1902.01378.pdf) by Arthur Juliani 
* [Alex Nichol Image Classifier implementation](https://github.com/unixpickle/obs-tower2) 
