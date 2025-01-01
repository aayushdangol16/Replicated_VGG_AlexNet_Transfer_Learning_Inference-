import gradio as gr
from resnet_model import model
from torchvision import transforms
from PIL import Image


def predict(image):
    image_size = 227
    stats = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    transform=transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(*stats)
    ])
    transform1=transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
    ])

    img_tensor = transform(image).unsqueeze(0)

    model.eval()
    output=model(img_tensor)
    if(output.item()>0.5):
        return transforms.ToPILImage()(transform1(image)),"Dog"
    else:
        return transforms.ToPILImage()(transform1(image)),"Cat"



iface = gr.Interface(fn=predict, 
                     inputs=gr.Image(type="pil"), 
                     outputs=["image","text"])


iface.launch()
