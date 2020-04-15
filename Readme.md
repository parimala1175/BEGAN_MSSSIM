###### Requirements
- python 2.7
- tensorflow >= 1.2 (verified on 1.2 and 1.3)
- tqdm
- (optional) pynvml - for automatic gpu selection
####### Dataset 
Download CelebA dataset:

1. python download.py celebA 

Convert images to tfrecords format:
Options for converting are hard-coded, so ensure to modify it before run `convert.py`.

1. CelebA

http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
- All experiments were performed on 64x64 CelebA dataset
- The dataset has 202599 images

2. Stanford cars dataset

https://ai.stanford.edu/~jkrause/cars/car_dataset.html
- The dataset has 202599 images


#### converting data to Tf records

1. python convert.py 

Convert images to tfrecords format:

Options for converting are hard-coded, so ensure to modify it before run `convert.py`.
please specify the path for image database either car or celebA in convert.py 


 
######## Training
###1. If you want to change the settings of each model, you must also modify code directly.

python train.py --help
usage: train.py [-h] [--num_epochs NUM_EPOCHS] [--batch_size BATCH_SIZE]
                [--num_threads NUM_THREADS] --model MODEL [--name NAME]
                --dataset DATASET [--ckpt_step CKPT_STEP] [--renew]

optional arguments:
  -h, --help            show this help message and exit
  --num_epochs NUM_EPOCHS
                        default: 20
  --batch_size BATCH_SIZE
                        default: 128
  --num_threads NUM_THREADS
                        # of data read threads (default: 4)
  --model MODEL         BEGAN
  --name NAME           default: name=model
  --dataset DATASET, -D DATASET  CelebA / car
  --ckpt_step CKPT_STEP
                        # of steps for saving checkpoint (default: 5000)
  --renew               train model from scratch - clean saved checkpoints and
                        summaries
### the loss function 

1. The parameters(lambda1 and lambda2) can be varied in the began model file in models.

Note: As the weight for MSSSIM increase training will become unstable.
      learning rate need to be decreased.
      increase the decay rate and increase the decay steps.

2. In case of tensorflow version 1.11.0 

   # MSSSIM was there in tensor flow library

   # We can use that MSSSIM function to formulate the loss function.

   ## tf.image.ssim_multiscale(
    img1,
    img2,
    max_val,
    power_factors=_MSSSIM_WEIGHTS
)

######## Generating 

python eval.py --help

usage: eval.py [-h] --model MODEL [--name NAME] --dataset DATASET
               [--sample_size SAMPLE_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         BEGAN
  --name NAME           default: name=model
  --dataset DATASET, -D DATASET
                        CelebA / car
  --sample_size SAMPLE_SIZE, -N SAMPLE_SIZE


####### Evaluating by metrics FID and NIQE
change directory to . /Quantitative Evaluation

1. FID:
  i)  Give the paths for real images and generated images.
  ii) Make sure that two image sets will have more than 3000 images.
  iii) Run fid.py 

2. NIQE_feature analysis_codes :

  i) Mention path for the images 
  ii) Run niqe_scores.m 

### Ackownledgements

Code references:
1. BEGAN: https://github.com/khanrc/tf.gans-comparison
2. NIQE:  http://live.ece.utexas.edu/research/Quality/blind.htm
3. FID:   https://github.com/bioinf-jku/TTUR

@inproceedings{kancharla2018improving,

  title={Improving the Visual Quality of Generative Adversarial Network (GAN)-Generated Images Using the Multi-Scale Structural Similarity Index},
  
  author={Kancharla, Parimala and Channappayya, Sumohana S},
  
  booktitle={2018 25th IEEE International Conference on Image Processing (ICIP)},
  
  pages={3908--3912},
  
  year={2018},
  
  organization={IEEE}
  
}

