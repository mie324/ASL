from PIL import Image
import numpy as np
import torch

def load_model(path):
    model = torch.load(path, map_location='cpu')
    model = model.eval()
    return model

def load_image(path):
    image = Image.open(path)
    image = image.resize((200,200), Image.ANTIALIAS)
    image = np.asarray(image)
    image = np.transpose(image, (2,0,1))
    image = image/255.0
    image = torch.from_numpy(image)
    image = image.unsqueeze(0)
    return image

def predict_file(image_path, model_path):
    model = load_model(model_path)
    image = load_image(image_path)

    predictions = model(image)
    prediction = predictions.argmax(dim=1).item()
    if prediction < 26:
        letter = chr(prediction + 97)
    elif prediction == 26:
        letter = 'del'
    elif prediction == 27:
        letter = 'nothing'
    elif prediction == 28:
        letter = 'space'

    return letter




