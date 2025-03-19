import io
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from PIL import Image
import torch
from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

image_processor = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf")
model = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf").to(device)

class URLRequest(BaseModel):
    url: str

@app.post("/depthmap")
async def generate_depth_map(request: URLRequest):
    # Open the image
    image = Image.open(requests.get(request.url, stream=True).raw)
    
    inputs = image_processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    post_processed_output = image_processor.post_process_depth_estimation(
        outputs, target_sizes=[(image.height, image.width)],
    )

    # field_of_view = post_processed_output[0]["field_of_view"]
    # focal_length = post_processed_output[0]["focal_length"]
    depth = post_processed_output[0]["predicted_depth"]
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    depth = depth * 255.
    depth = depth.detach().cpu().numpy()
    depth = Image.fromarray(depth.astype("uint8"))
    
    # Save to a bytes buffer
    img_byte_arr = io.BytesIO()
    depth.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()
    
    return Response(content=img_byte_arr, media_type="image/png")
