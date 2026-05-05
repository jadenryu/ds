import torch
import numpy as np
import gradio as gr
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

COUNTRIES = [
    "Afghanistan", "Albania", "Algeria", "Andorra", "Angola", "Argentina",
    "Armenia", "Australia", "Austria", "Azerbaijan", "Bangladesh", "Belarus",
    "Belgium", "Bolivia", "Bosnia and Herzegovina", "Botswana", "Brazil",
    "Bulgaria", "Cambodia", "Canada", "Chile", "China", "Colombia", "Croatia",
    "Czech Republic", "Denmark", "Ecuador", "Egypt", "Estonia", "Ethiopia",
    "Finland", "France", "Georgia", "Germany", "Ghana", "Greece", "Guatemala",
    "Hungary", "Iceland", "India", "Indonesia", "Ireland", "Israel", "Italy",
    "Japan", "Jordan", "Kazakhstan", "Kenya", "Kyrgyzstan", "Laos", "Latvia",
    "Lebanon", "Lithuania", "Malaysia", "Mexico", "Mongolia", "Morocco",
    "Netherlands", "New Zealand", "Nigeria", "Norway", "Pakistan", "Panama",
    "Peru", "Philippines", "Poland", "Portugal", "Romania", "Russia", "Rwanda",
    "Saudi Arabia", "Senegal", "Serbia", "Singapore", "Slovakia", "Slovenia",
    "South Africa", "South Korea", "Spain", "Sri Lanka", "Sweden", "Switzerland",
    "Taiwan", "Thailand", "Tunisia", "Turkey", "Uganda", "Ukraine",
    "United Arab Emirates", "United Kingdom", "United States", "Uruguay",
    "Vietnam", "Zimbabwe",
]

print("Loading CLIP model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = model.to(device)
model.eval()
print(f"Model ready on {device}")

PROMPTS = [f"a street view photo from {c}" for c in COUNTRIES]


def predict(image):
    if image is None:
        return {}

    pil_image = Image.fromarray(image).convert("RGB")

    inputs = processor(
        text=PROMPTS,
        images=pil_image,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]

    top_idx = np.argsort(probs)[::-1][:10]
    return {COUNTRIES[i]: float(probs[i]) for i in top_idx}


with gr.Blocks(title="GeoGuessr Classifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# GeoGuessr Image Classifier")
    gr.Markdown(
        "Upload a street-view or landscape photo and the model will guess which country it's from."
    )

    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(label="Upload Image", type="numpy")
            classify_btn = gr.Button("Classify", variant="primary")
        with gr.Column(scale=1):
            label_output = gr.Label(num_top_classes=10, label="Top Country Predictions")

    classify_btn.click(fn=predict, inputs=img_input, outputs=label_output)
    img_input.change(fn=predict, inputs=img_input, outputs=label_output)

demo.launch()
