from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2, os
from pathlib import Path 
from tqdm import tqdm

def load_image_path(path, img_type='jpg'):
    path = Path(path)
    files = path.glob("*.{}".format(img_type))
    return sorted(list(files))


model = load_model("groundingdino/config/GroundingDINO_SwinB_cfg.py", "weights/groundingdino_swinb_cogcoor.pth")
IMAGE_PATH = "/home/luciana/workspace/others/course/vision_and_image/homework1/image_matcher/results/clip/"
TEXT_PROMPTS = [
    ['panda card', 'maple card'],
    ['Orange mini person model', 'mini old person model'],
    ['book', 'Colorful can'],
    ['orange box', 'Purple Gamer'],
    ['Giraffe model', 'mini girl model in dress'],
    ['milk box', 'black-and-white photo']
    ]
OUT_PATH = 'results/'
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

for image_file_idx in range(0, 6):
    image_file_path = IMAGE_PATH + "source_image_{}".format(image_file_idx)
    images_path = load_image_path(image_file_path)
    text_prompt = TEXT_PROMPTS[image_file_idx]
    out_file_path = OUT_PATH + "/source_image_bbox_{}/".format(image_file_idx)
    if not os.path.exists(out_file_path):
        os.makedirs(out_file_path)
    for image_path in tqdm(images_path):

        image_source, image = load_image(IMAGE_PATH)
        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=text_prompt,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )
        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        image_name = image_path.split('/')[-1]
        out_img_path = out_file_path + image_name
        cv2.imwrite(out_img_path, annotated_frame)