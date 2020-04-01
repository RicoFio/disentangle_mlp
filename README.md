# MLP Semester 2 - Group Coursework Group 079
We introduce a modern retake on the paper "Autoencoding beyond pixels using a learned similarity matrix" by Larsen et al. ([paper](https://arxiv.org/abs/1512.09300)). By introducing a beta-VAE architecture we improve on the latent space disentangling and present a new beta-VAE-GAN hybrid.

## Abstract 
Recent research in probabilistic generative models based on deep neural networks has led to image generation systems of a quality previously unseen. We re-explore an algorithm first introduced by that combines Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs). Its aim is to combine the strength of the two approaches by informing the VAE's loss with the GAN's discriminator, thus creating a *feature-wise* loss. This helps find *disentangled*latent representations, which often better capture features of the data capable of generalizing beyond training samples. These result in improved sample generation. We explore the reproducibility challenges of the algorithm, compare it with VAE and GAN and augment it with $\beta$-VAE, an extension that has been shown to improve the disentanglement of latent representation in VAE. This choice involves the tuning of one key hyperparameter: we avoid the expensive heuristics proposed in the literature and show improved results on our baselines with a simple Bayesian optimization procedure with a 10 % decrease in Frechet Inception Distance score.

## Setup
- Install dependencies from requirements.txt
- Download data from [here](http://tamaraberg.com/faceDataset/)

## References
- "Autoencoding beyond pixels using a learned similarity matrix" by Larsen et al. ([paper](https://arxiv.org/abs/1512.09300))
- " beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework." ([paper](https://pdfs.semanticscholar.org/a902/26c41b79f8b06007609f39f82757073641e2.pdf))

## Change Log
- Imported boilerplate from [here](https://github.com/unmeg/pytorch-boilerplate)
