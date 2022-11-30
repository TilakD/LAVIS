import os

import cv2
import glob
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
import torch

from lavis.models import load_model_and_preprocess

imdir = './test_images/'
save_dir = './test_output/'
ext = ['png', 'jpg', 'jpeg']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)

font = ImageFont.load_default()

files = []
[files.extend(glob.glob(imdir + '*.' + e)) for e in ext]

for file in files:
	print(file)
	raw_image = Image.open(file).convert("RGB")
	image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

	caption = model.generate({"image": image})
	print(caption)

	captions = model.generate({"image": image}, use_nucleus_sampling=True, num_captions=3)
	print(captions)
	print("*"*80)

	draw = ImageDraw.Draw(raw_image)
	draw.text((100, 100),caption[0],(255,255,255),font=font)
	draw.text((100, 120),captions[0],(255,255,255),font=font)
	draw.text((100, 140),captions[1],(255,255,255),font=font)
	draw.text((100, 160),captions[2],(255,255,255),font=font)


	file_name = os.path.basename(file).split('/')[-1]
	raw_image.save(os.path.join(save_dir, file_name))
