"""
Accuracy evaluation using Google Street View Static API.

Get a free API key at: https://developers.google.com/maps/documentation/streetview/get-api-key
Enable "Street View Static API" in your Google Cloud project.
"""

import io
import requests
import numpy as np
import torch
from PIL import Image
from app import model, processor, COUNTRIES, PROMPTS, device

GOOGLE_API_KEY = "API_KEY"

TEST_LOCATIONS = [
    (48.8566, 2.3522, "France"),
    (51.5074, -0.1278, "United Kingdom"),
    (40.7128, -74.0060, "United States"),
    (35.6895, 139.6917, "Japan"),
    (52.5200, 13.4050, "Germany"),
    (-33.8688, 151.2093, "Australia"),
    (55.7558, 37.6173, "Russia"),
    (28.6139, 77.2090, "India"),
    (-23.5505, -46.6333, "Brazil"),
    (39.9042, 116.4074, "China"),
    (41.9028, 12.4964, "Italy"),
    (40.4168, -3.7038, "Spain"),
    (37.9838, 23.7275, "Greece"),
    (59.9139, 10.7522, "Norway"),
    (59.3293, 18.0686, "Sweden"),
    (60.1699, 24.9384, "Finland"),
    (55.6761, 12.5683, "Denmark"),
    (52.3676, 4.9041, "Netherlands"),
    (50.8503, 4.3517, "Belgium"),
    (47.3769, 8.5417, "Switzerland"),
    (48.2082, 16.3738, "Austria"),
    (50.0755, 14.4378, "Czech Republic"),
    (52.2297, 21.0122, "Poland"),
    (47.4979, 19.0402, "Hungary"),
    (44.4268, 26.1025, "Romania"),
    (42.6977, 23.3219, "Bulgaria"),
    (45.8150, 15.9819, "Croatia"),
    (44.8176, 20.4633, "Serbia"),
    (46.0569, 14.5058, "Slovenia"),
    (42.4304, 19.2594, "Montenegro"),
    (41.3275, 19.8189, "Albania"),
    (37.9715, 23.7257, "Greece"),
    (41.0082, 28.9784, "Turkey"),
    (31.7683, 35.2137, "Israel"),
    (33.8869, 35.5131, "Lebanon"),
    (31.9522, 35.2332, "Jordan"),
    (24.7136, 46.6753, "Saudi Arabia"),
    (25.2048, 55.2708, "United Arab Emirates"),
    (30.0444, 31.2357, "Egypt"),
    (33.9716, -6.8498, "Morocco"),
    (36.8190, 10.1658, "Tunisia"),
    (-25.7479, 28.2293, "South Africa"),
    (1.2921, 36.8219, "Kenya"),
    (0.3476, 32.5825, "Uganda"),
    (-1.9441, 30.0619, "Rwanda"),
    (5.6037, -0.1870, "Ghana"),
    (6.5244, 3.3792, "Nigeria"),
    (14.7167, -17.4677, "Senegal"),
    (-18.9249, 47.5185, "Madagascar"),
    (15.5007, 32.5599, "Sudan"),
    (12.3714, -1.5197, "Burkina Faso"),
    (4.3612, 18.5550, "Central African Republic"),
    (-11.2027, 17.8739, "Angola"),
    (-25.9692, 32.5732, "Mozambique"),
    (-13.9626, 33.7741, "Malawi"),
    (-15.4167, 28.2833, "Zambia"),
    (-17.8252, 31.0335, "Zimbabwe"),
    (-22.9068, 47.5362, "Madagascar"),
    (43.2965, 5.3698, "France"),
    (19.4326, -99.1332, "Mexico"),
    (4.7110, -74.0721, "Colombia"),
    (-12.0464, -77.0428, "Peru"),
    (-34.6037, -58.3816, "Argentina"),
    (-33.4489, -70.6693, "Chile"),
    (-0.1807, -78.4678, "Ecuador"),
    (10.4806, -66.9036, "Venezuela"),
    (-16.5000, -68.1500, "Bolivia"),
    (-25.2867, -57.6470, "Paraguay"),
    (-34.9011, -56.1645, "Uruguay"),
    (8.9936, -79.5197, "Panama"),
    (14.6349, -90.5069, "Guatemala"),
    (13.6929, -89.2182, "El Salvador"),
    (14.0818, -87.2068, "Honduras"),
    (12.1364, -86.2514, "Nicaragua"),
    (9.9281, -84.0907, "Costa Rica"),
    (18.4861, -69.9312, "Dominican Republic"),
    (23.1136, -82.3666, "Cuba"),
    (18.5944, -72.3074, "Haiti"),
    (17.9975, -76.7930, "Jamaica"),
    (43.8003, 20.4651, "Serbia"),
    (42.0000, 21.4333, "North Macedonia"),
    (42.6629, 21.1655, "Kosovo"),
    (53.9006, 27.5590, "Belarus"),
    (50.4501, 30.5234, "Ukraine"),
    (41.6938, 44.8015, "Georgia"),
    (40.1872, 47.1926, "Azerbaijan"),
    (40.1596, 44.5093, "Armenia"),
    (51.1801, 71.4460, "Kazakhstan"),
    (41.2995, 69.2401, "Uzbekistan"),
    (37.9601, 58.3261, "Turkmenistan"),
    (38.5598, 68.7738, "Tajikistan"),
    (42.8746, 74.5698, "Kyrgyzstan"),
    (47.8864, 106.9057, "Mongolia"),
    (27.4712, 89.6339, "Bhutan"),
    (27.7172, 85.3240, "Nepal"),
    (6.9271, 79.8612, "Sri Lanka"),
    (7.8731, 80.7718, "Sri Lanka"),
    (23.8103, 90.4125, "Bangladesh"),
    (15.5527, 32.5324, "Sudan"),
    (15.3694, 38.9183, "Eritrea"),
    (11.8251, 42.5903, "Djibouti"),
    (2.0469, 45.3182, "Somalia"),
    (9.0579, 7.4951, "Nigeria"),
    (3.8480, 11.5021, "Cameroon"),
    (-4.3217, 15.3222, "Democratic Republic of Congo"),
    (-4.2634, -15.2832, "Angola"),
    (0.4162, 9.4673, "Gabon"),
    (-0.8037, 11.6094, "Gabon"),
    (3.3841, 29.3543, "South Sudan"),
    (9.1450, 40.4897, "Ethiopia"),
    (-6.1630, 35.7516, "Tanzania"),
    (13.5127, 2.1128, "Niger"),
    (12.6392, -8.0029, "Mali"),
    (11.8037, 15.0111, "Chad"),
    (17.3292, -62.7342, "Saint Kitts and Nevis"),
    (13.1600, -61.2248, "Saint Vincent and the Grenadines"),
    (13.9094, -60.9789, "Saint Lucia"),
    (12.1165, -61.6790, "Grenada"),
    (15.3010, -61.3704, "Dominica"),
    (17.1274, -61.8468, "Antigua and Barbuda"),
    (13.4432, 144.7937, "Guam"),
    (-13.7590, -172.1046, "Samoa"),
    (-21.1789, -175.1982, "Tonga"),
    (-17.7333, 168.3220, "Vanuatu"),
    (-8.9742, 160.1562, "Solomon Islands"),
    (-18.1248, 178.4501, "Fiji"),
    (-0.5477, 166.9209, "Nauru"),
    (7.1095, 171.3803, "Marshall Islands"),
    (6.8874, 158.2150, "Micronesia"),
    (-8.5568, 179.2170, "Tuvalu"),
    (-9.4438, -159.9475, "Cook Islands"),
    (-21.2120, -159.7750, "Cook Islands"),
]

seen = set()
EVAL_SET = []
for lat, lng, country in TEST_LOCATIONS:
    if country not in seen and country in COUNTRIES:
        seen.add(country)
        EVAL_SET.append((lat, lng, country))


def fetch_street_view(lat: float, lng: float, size: str = "640x640") -> Image.Image | None:
    url = (
        f"https://maps.googleapis.com/maps/api/streetview"
        f"?size={size}&location={lat},{lng}"
        f"&fov=90&heading=0&pitch=0"
        f"&key={GOOGLE_API_KEY}"
    )
    resp = requests.get(url, timeout=15)
    if resp.status_code != 200 or resp.headers.get("content-type", "").startswith("application/json"):
        return None
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def classify(pil_image: Image.Image) -> list[tuple[str, float]]:
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
    ranked = np.argsort(probs)[::-1]
    return [(COUNTRIES[i], float(probs[i])) for i in ranked]


def evaluate():
    if GOOGLE_API_KEY == "YOUR_API_KEY_HERE":
        print("ERROR: set GOOGLE_API_KEY in evaluate.py first.")
        return

    top1_correct = 0
    top5_correct = 0
    total = 0
    failures = 0

    print(f"Evaluating {len(EVAL_SET)} locations...\n")
    for lat, lng, true_country in EVAL_SET:
        img = fetch_street_view(lat, lng)
        if img is None:
            print(f"  [SKIP] No Street View for {true_country} ({lat}, {lng})")
            failures += 1
            continue

        results = classify(img)
        top1 = results[0][0]
        top5 = [r[0] for r in results[:5]]
        hit1 = top1 == true_country
        hit5 = true_country in top5

        top1_correct += hit1
        top5_correct += hit5
        total += 1

        status = "✓" if hit1 else ("~" if hit5 else "✗")
        print(f"  {status} {true_country:30s}  predicted: {top1}")

    if total == 0:
        print("\nNo images could be fetched.")
        return

    print(f"\n{'─'*50}")
    print(f"Evaluated : {total}  (skipped {failures})")
    print(f"Top-1 acc : {top1_correct}/{total} = {100*top1_correct/total:.1f}%")
    print(f"Top-5 acc : {top5_correct}/{total} = {100*top5_correct/total:.1f}%")


if __name__ == "__main__":
    evaluate()
