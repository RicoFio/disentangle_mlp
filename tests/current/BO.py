# BAYESIAN OPT 

# REQUIRED INPUT :  set of final FID values for K distinct fully trained models ,  eg. up to epoch 10 
# as well as the beta used in that run 



import math
import torch
from torchvision import transforms
from botorch.models import SingleTaskGP
from gpytorch.constraints import GreaterThan
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.optim import SGD
from matplotlib import pyplot as plt
import sys
import numpy as np
from sklearn import preprocessing
from botorch.utils.transforms import standardize, normalize, unnormalize


# use a GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float




###################################################################
# use regular spaced points on the interval [0, 1]
# train_X = torch.linspace(0, 1, 15, dtype=dtype, device=device)
# # training data needs to be explicitly multi-dimensional
# train_X = train_X.unsqueeze(1)


# # THIS WILL ACTUALLY  BE OUR FID VALUES 
# # sample observed values and add some synthetic noise
# train_Y = torch.sin(train_X * (2 * math.pi)) + 0.15 * torch.randn_like(train_X)


#################################################################################



train_X = np.array([0., 10., 50. , 100., 150.]).reshape(-1,1)
train_Y = np.array([150.34, 143.56, 133.2, 160.7, 162.33]).reshape(-1,1)

train_X = torch.tensor(train_X)
train_Y = torch.tensor(train_Y)

train_X = torch.tensor(train_X,dtype=dtype)  # BETAS
# train_X = train_X.unsqueeze(1)
train_Y = torch.tensor(train_Y,dtype=dtype) # FID VALUE OF TRAINED MODEL 
# train_Y = train_Y.unsqueeze(1)

bounds_X = torch.tensor((torch.min(train_X), torch.max(train_X))).unsqueeze(1)
bounds_Y = torch.tensor((torch.min(train_Y), torch.max(train_Y))).unsqueeze(1)

train_X = normalize(train_X, bounds_X)
train_Y = normalize(train_Y, bounds_Y)




# ################################################################



model = SingleTaskGP(train_X=train_X, train_Y=train_Y)
model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-5))

# We will jointly optimize the kernel hyperparameters and the likelihood's noise parameter, 
# by minimizing the negative gpytorch.mlls.ExactMarginalLogLikelihood
mll = ExactMarginalLogLikelihood(likelihood=model.likelihood, model=model)
# set mll and all submodules to the specified dtype and device
mll = mll.to(train_X)

# Define optimizer and specify parameters to optimize¶
optimizer = SGD([{'params': model.parameters()}], lr=0.6)




###############################################################################
# Fit model hyperparameters and noise level

NUM_EPOCHS = 150

model.train()

for epoch in range(NUM_EPOCHS):
    # clear gradients
    optimizer.zero_grad()
    # forward pass through the model to obtain the output MultivariateNormal
    output = model(train_X)
    # Compute negative marginal log likelihood
    loss = - mll(output, model.train_targets)
    # back prop gradients
    loss.backward()
    # print every 10 iterations
    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch+1:>3}/{NUM_EPOCHS} - Loss: {loss.item():>4.3f} "
            f"lengthscale: {model.covar_module.base_kernel.lengthscale.item():>4.3f} " 
            f"noise: {model.likelihood.noise.item():>4.3f}" 
         )
    optimizer.step()


# Compute posterior over test points and plot fit

# set model (and likelihood)
model.eval();

# Initialize plot
f, ax = plt.subplots(1, 1, figsize=(6, 4))
# test model on 101 regular spaced points on the interval [0, 1]
# test_X = torch.linspace(0, 1, 101, dtype=dtype, device=device)

test_X = torch.linspace(0., 300., 200, dtype=dtype, device=device)
bounds_X_test = torch.tensor((torch.min(test_X), torch.max(test_X))).unsqueeze(1)
test_X = normalize(test_X, bounds_X_test)

bounds_Y_test = bounds_Y
# no need for gradients
with torch.no_grad():
    # compute posterior
    posterior = model.posterior(test_X)
    # Get upper and lower confidence bounds (2 standard deviations from the mean)
    lower, upper = posterior.mvn.confidence_region()
    # Plot training points as black stars
    ax.plot(unnormalize(train_X, bounds_X).cpu().numpy(), unnormalize(train_Y, bounds_Y).cpu().numpy(), 'k*')
    # Plot posterior means as blue line
    ax.plot(unnormalize(test_X, bounds_X_test).cpu().numpy(), unnormalize(posterior.mean, bounds_Y_test).cpu().numpy(), 'b')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(unnormalize(test_X, bounds_X_test).cpu().numpy(), unnormalize(lower, bounds_Y_test).cpu().numpy(), unnormalize(upper, bounds_Y_test).cpu().numpy(), alpha=0.5)
ax.legend(['Observed Data', 'Mean', 'Confidence'])
plt.tight_layout()
plt.show()






