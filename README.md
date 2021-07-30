# Semantic Image Synthesis with Score-Based Generative Models
 
This repo contains the implementation for my Bachelor thesis [Semantic Image Synthesis with Score-Based Generative Models](https://github.com/TimK1998/Bachelorarbeit)

by [Tim Küchler](tim.kuechler@online.de)

Please find my thesis (english) following this link: https://github.com/TimK1998/Bachelor-Thesis/blob/main/Bachelorarbeit.pdf
 
--------------------

Note: This README is work in progress!


# How to run the code

## Dependencies

First install PyTorch 1.8, then run ```setup.py``` with the command 
```sh
python setup.py install
```

## Usage
Train or sample from models trough ```main.py```:
```sh
main.py:
  workdir: Working directory
  config: Name of the config
  mode: <train|sample>: Running mode: train or sample
  --sample_mode: Sampling mode
```

* ```workdir``` is the path where all checkpoints and samples should be saved. The path you specify gets appended to ```./output``` so you might want to specify the working directory with only one word.
* ```config``` is the name of the config to use. Refer to ```configs/ve``` for examples.
* ```mode``` is either "train" or "sample". When set to train is starts the training pipeline for a new model, or continues training if workdir already contains a valid checkpoint. When set to sample it loads the latest checkpoint in ```./output/workdir/checkpoints``` and starts sampling with the sampling mode specified.
* ```sample_mode```: The mode for the sampling procedure. Already implemented are ```uncond``` for unconditional samples and ```cond``` for conditional samples. Feel free to add modes by implementing them in the ```sample(...)``` function in ```run.py```.

## Train

For example to train a Score-Based Generative Model on the Cityscapes dataset run
```sh
python main.py cityscapes_workdir cityscapes256_ve train
```

## Sample

For example to conditinally sample from a trained Score-Based Generative Model on the Cityscapes dataset run
```sh
python main.py cityscapes_workdir cityscapes256_ve sample cond
```

# References

If you find the code useful for your research, please consider citing
```bib
@unpublished{kuechler_sem_synth_score_based,
    author="Tim Küchler",
    title={Semantic Image Synthesis with Score-Based Generative Models},
    year={2021}
    howpublished={\url{https://github.com/TimK1998/SemanticSynthesisForScoreBasedModels}}
}
```

This work is built upon previous papers which might also interest you:

* Yang Song and Jascha Sohl-Dickstein and Diederik P Kingma and Abhishek Kumar and Stefano Ermon and Ben Poole. "Score-Based Generative Modeling through Stochastic Differential Equations." *International Conference on Learning Representations*. 2021.
* Yang Song and Stefano Ermon. "Generative Modeling by Estimating Gradients of the Data Distribution." *Proceedings of the 33rd Annual Conference on Neural Information Processing Systems*. 2019.
* Yang Song and Stefano Ermon. "Improved techniques for training score-based generative models." *Proceedings of the 34th Annual Conference on Neural Information Processing Systems*. 2020.

The code heavily borrows from https://github.com/yang-song/score_sde_pytorch

# License

This implementation is licensed under the Apache License 2.0.
