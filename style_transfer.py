# In this code I created a program which takes 2 images, a content image and a style image, and we would like
# to blend them together to extract the style of one image and imply it on another image.
# The way we do it is by using CNN, we extract the style of the style image by taking always the first convolutional
#layer from each batch of convolutional layers, so basically we have a couple of (conv - RELU - conv - RELU - MaxPool) layers,
# and we take the first conv. For the content image we take only one conv layer(in here the second conv layer in the
# fourth batch). And using these features we try tp blend them for creating the target image which is the wanted image.
# The target is created by taking the basic content image and apply the features of the style image on this.


# Writer- Elian Iluk
# Email- elian10119@gmail.com


#----------------------------------------------------------------------------------------------------------------------
#some imports

from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
import requests
from torchvision import transforms, models
#----------------------------------------------------------------------------------------------------------------------
#functions for help

def load_image(img_path, max_size=400, shape=None):
    ''' Load in and transform an image, making sure the image
       is <= 400 pixels in the x-y dims.'''
    if "http" in img_path:
        response = requests.get(img_path)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(img_path).convert('RGB')

    # large images will slow down processing
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    if shape is not None:
        size = shape

    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = in_transform(image)[:3, :, :].unsqueeze(0)

    return image


# helper function for un-normalizing an image
# and converting it from a Tensor image to a NumPy image for display
def im_convert(tensor):
    """ Display a tensor as an image. """

    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image


def get_features(image, model, layers=None):
    """ Run an image forward through a model and get the features for
        a set of layers. Default layers are for VGGNet matching Gatys et al (2016)
    """

    ## Need the layers for the content and style representations of an image
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1',
                  '10': 'conv3_1',
                  '19': 'conv4_1',
                  '21': 'conv4_2',
                  '28': 'conv5_1'}

    ## -- do not need to change the code below this line -- ##
    features = {}
    x = image
    # model._modules is a dictionary holding each module in the model
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x

    return features


def gram_matrix(tensor):
    """ Calculate the Gram Matrix of a given tensor
        Gram Matrix: https://en.wikipedia.org/wiki/Gramian_matrix
    """

    ## get the batch_size, depth, height, and width of the Tensor
    ## reshape it, so we're multiplying the features for each channel
    ## calculate the gram matrix
    batch_size, d, h, w = tensor.shape
    tensor = tensor.view(-1, h * w)
    gram = torch.mm(tensor, tensor.t())

    return gram



#----------------------------------------------------------------------------------------------------------------------
#load the VGG19 model and the images, and extract the features

# get the "features" portion of VGG19
vgg = models.vgg19(pretrained=True).features

# freeze all VGG parameters since we're only optimizing the target image
for param in vgg.parameters():
    param.requires_grad_(False)

# move the model to GPU, if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg.to(device)

# load in content and style image
content = load_image('images/ab007650-cbfe-11ee-b83b-0f87a864f372.jpg').to(device)
# Resize style to match content, makes code easier
style =load_image('images/snow2.jpg', shape=content.shape[-2:]).to(device)

# Display the images
print("The images look like this:")

# Unnormalize content and style images and prepare for display
content_img = content.clone().detach().cpu().squeeze(0).permute(1, 2, 0).numpy() * 0.5 + 0.5
style_img = style.clone().detach().cpu().squeeze(0).permute(1, 2, 0).numpy() * 0.5 + 0.5

# Plot the content and style images side-by-side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

# Content image
ax1.imshow(content_img.clip(0, 1))  # Clip to valid range [0, 1]
ax1.set_title("Content Image")
ax1.axis("off")

# Style image
ax2.imshow(style_img.clip(0, 1))  # Clip to valid range [0, 1]
ax2.set_title("Style Image")
ax2.axis("off")

plt.show()

# print out VGG19 structure so you can see the names of various layers
print(vgg)

# get content and style features only once before forming the target image
content_features = get_features(content, vgg)
style_features = get_features(style, vgg)

# calculate the gram matrices for each layer of our style representation
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# create a third "target" image and prep it for change
# it is a good idea to start off with the target as a copy of our *content* image
# then iteratively change its style
target = content.clone().requires_grad_(True).to(device)

# weights for each style layer
# weighting earlier layers more will result in *larger* style artifacts
# notice we are excluding `conv4_2` our content representation
style_weights = {'conv1_1': 1.,
                 'conv2_1': 0.8,
                 'conv3_1': 0.5,
                 'conv4_1': 0.3,
                 'conv5_1': 0.1}

# the scale between the affect of the content image and the style image
content_weight = 1  # alpha
style_weight = 1e6  # beta

#----------------------------------------------------------------------------------------------------------------------
#train the model
print("now we train the model:")

# for displaying the target image, intermittently
show_every = 50

# iteration hyperparameters
optimizer = optim.Adam([target], lr=0.003)
steps = 100  # decide how many iterations to update your image (5000)

for ii in range(1, steps + 1):

    ## Then calculate the content loss
    target_features = get_features(target, vgg)
    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)

    # the style loss
    # initialize the style loss to 0
    style_loss = 0
    # iterate through each style layer and add to the style loss
    for layer in style_weights:
        # get the "target" style representation for the layer
        target_feature = target_features[layer]
        _, d, h, w = target_feature.shape

        ## Calculate the target gram matrix
        target_gram = gram_matrix(target_feature)

        ## get the "style" style representation
        style_gram = gram_matrix(style_features[layer])
        ## Calculate the style loss for one layer, weighted appropriately
        layer_style_loss = torch.mean((target_gram - style_gram) ** 2) * style_weights[layer]

        # add to the style loss
        style_loss += layer_style_loss / (d * h * w)

    #calculate the *total* loss
    total_loss = content_weight * content_loss + style_weight * style_loss

    ## -- do not need to change code, below -- ##
    # update your target image
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # display intermediate images and print the loss
    if ii % show_every == 0:
        print('Total loss: ', total_loss.item())
        plt.imshow(im_convert(target))
        plt.show()

#----------------------------------------------------------------------------------------------------------------------
# Display content and final target image
print("Display the target image:")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))  # Create subplots for side-by-side images

# Prepare content image for display
content_img = content.clone().detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
content_img = content_img * 0.5 + 0.5  # Unnormalize (if normalized to [-1, 1])

# Prepare target image for display
target_img = target.clone().detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
target_img = target_img * 0.5 + 0.5  # Unnormalize

# Display content image
ax1.imshow(content_img.clip(0, 1))  # Clip values to [0, 1]
ax1.set_title("Content Image")
ax1.axis("off")  # Remove axes for better visualization

# Display target image
ax2.imshow(target_img.clip(0, 1))  # Clip values to [0, 1]
ax2.set_title("Target Image")
ax2.axis("off")  # Remove axes for better visualization

# Show the plot
plt.show()

