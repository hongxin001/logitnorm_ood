
# Mitigating Neural Network Overconfidence with Logit Normalization


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the model(s) in the paper, run this command:

```train

# CE loss
python train.py cifar10 --alg standard -m wrn --exp_name normal --gpu 7


# LogitNorm loss
python train.py cifar10 --alg standard -m wrn --exp_name logitnorm --gpu 7 --loss logit_norm --temp 0.01

```


## Evaluation

To evaluate the model on CIFAR-10, run:

```eval
python test.py cifar10 --method_name cifar10_wrn_${exp_name}_standard --num_to_avg 10 --gpu 0 --seed 1 --prefetch 0

# Example with pretrained model
python test.py cifar10 --method_name cifar10_wrn_logitnorm_standard --num_to_avg 10 --gpu 0 --seed 1 --prefetch 0

```


