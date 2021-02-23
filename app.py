from PIL import Image
from torchvision import models, transforms
import torch
import torch.nn.functional as F
import streamlit as st


file_up = st.file_uploader("Upload an image.", type="jpg")


def predict(image_path):
    net = models.densenet121(pretrained=True)
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    img = Image.open(image_path)
    batch_t = torch.unsqueeze(transform(img), dim=0)
    net.eval()
    out = net(batch_t)

    with open("imagenet_classes.txt") as f:
        classes = [line.strip("") for line in f.readlines()]

    probs = F.softmax(out, dim=1)[0] * 100
    _, idxs = torch.sort(probs, descending=True)
    return [(classes[idx].split(" ")[-1], probs[idx].item()) for idx in idxs[:5]]


if __name__ == "__main__":
    if file_up is not None:
        image = Image.open(file_up)
        st.image(image, caption="Uploaded Image.", use_column_width=True)
        st.write("")
        st.write("Just a second...")
        labels = predict(file_up)

        # print out the top 5 prediction labels with scores
        for i, label in enumerate(labels):
            name, prob = label[0], label[1]
            if prob < 0.05:
                break
            out = f"{i + 1}. {name} (Score: {prob:.1f}%)"
            st.write(out)
