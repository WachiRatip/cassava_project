# This is adapted from https://github.com/denistanjingyu/Image-Classification-Web-App-using-PyTorch-and-Streamlit
import os
from PIL import Image

import torch
from torchvision import transforms
import streamlit as st

MODEL_PATH = os.path.join(".", "model")

# transform the input image through resizing, normalization
data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# load the model
model = torch.load(MODEL_PATH)
model.eval()

def predict(image):
    """Return top 5 predictions ranked by highest probability.
    Parameters
    ----------
    :param image: uploaded image
    :type image: jpg
    :rtype: list
    :return: top 5 predictions ranked by highest probability
    """

    # load the image, pre-process it, and make predictions
    img = Image.open(image)
    batch_t = torch.unsqueeze(data_transforms(img), 0)
    model.eval()
    out = model(batch_t)

    with open('imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]

    # return the top 5 predictions ranked by highest probabilities
    prob = torch.nn.functional.softmax(out, dim = 1)[0] * 100
    _, indices = torch.sort(out, descending = True)
    return [(classes[idx], prob[idx].item()) for idx in indices[0][:5]]


def main():
    # set title of app
    st.title("Cassava species Image Classification Application")
    st.write("")

    # enable users to upload images for the model to make predictions
    file_up = st.file_uploader("Upload an image", type = "jpg")

    if file_up is not None:
        # display image that user uploaded
        image = Image.open(file_up)
        st.image(image, caption = 'Uploaded Image.', use_column_width = True)
        st.write("")
        st.write("Just a second ...")
        labels = predict(file_up)

        # print out the top 5 prediction labels with scores
        for i in labels:
            st.write("Prediction (index, name)", i[0], ",   Score: ", round(i[1], 2), "%.")


if __name__=="__main__":
    main()