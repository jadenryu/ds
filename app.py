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

PROMPT_TEMPLATES = [
    "A Street View photo in {}",
    "a photo I took in {}",
    "a photo I took while traveling in {}",
    "a street view photo from {}",
    "a photo of a street in {}",
    "a landscape photo from {}",
    "a photo showing the architecture of {}",
    "a road photo taken in {}",
]

print("Loading StreetCLIP model (ViT-L)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("geolocal/StreetCLIP")
processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")
model = model.to(device)
model.eval()

print("Precomputing text embeddings...")
with torch.no_grad():
    template_embeds = []
    for template in PROMPT_TEMPLATES:
        prompts = [template.format(c) for c in COUNTRIES]
        inputs = processor(
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(device)
        embeds = model.get_text_features(**inputs)
        embeds = embeds / embeds.norm(dim=-1, keepdim=True)
        template_embeds.append(embeds)
    TEXT_EMBEDS = torch.stack(template_embeds).mean(dim=0)
    TEXT_EMBEDS = TEXT_EMBEDS / TEXT_EMBEDS.norm(dim=-1, keepdim=True)
print("Ready.")


def _make_crops(pil_image: Image.Image) -> list[Image.Image]:
    """Multi-crop ensemble: original + 4 corner crops at 85% scale."""
    w, h = pil_image.size
    s = int(min(w, h) * 0.85)
    crops = [pil_image]
    for ox, oy in [(0, 0), (w - s, 0), (0, h - s), (w - s, h - s)]:
        ox = max(0, min(ox, w - s))
        oy = max(0, min(oy, h - s))
        crops.append(pil_image.crop((ox, oy, ox + s, oy + s)))
    return crops


def predict(image):
    if image is None:
        return {}

    pil_image = Image.fromarray(image).convert("RGB")
    crops = _make_crops(pil_image)

    inputs = processor(images=crops, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        image_embeds = model.get_image_features(**inputs)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        image_embed = image_embeds.mean(dim=0, keepdim=True)
        image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True)

        logit_scale = model.logit_scale.exp()
        logits = logit_scale * image_embed @ TEXT_EMBEDS.T
        probs = logits.softmax(dim=-1).cpu().numpy()[0]

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
