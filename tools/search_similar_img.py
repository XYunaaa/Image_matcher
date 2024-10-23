import faiss
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel, AutoProcessor, CLIPModel
from PIL import Image
from pathlib import Path 
from tqdm import tqdm
from collections import defaultdict
import os, shutil


def load_image_path(path, img_type='jpg'):
    path = Path(path)
    files = path.glob("*.{}".format(img_type))
    return sorted(list(files))

def extract_features_clip(image, processor_clip, model_clip):
    with torch.no_grad():
        inputs = processor_clip(images=image, return_tensors="pt").to(device)
        image_features = model_clip.get_image_features(**inputs)
        return image_features

def extract_features_dino(image, processor_dino, model_dino):
    with torch.no_grad():
        inputs = processor_dino(images=image, return_tensors="pt").to(device)
        outputs = model_dino(**inputs)
        image_features = outputs.last_hidden_state
        return image_features.mean(dim=1)
    
#Define a function that normalizes embeddings and add them to the index
def add_vector_to_index(embedding, index):
    #convert embedding to numpy
    vector = embedding.detach().cpu().numpy()
    #Convert to float32 numpy
    vector = np.float32(vector)
    #Normalize vector: important to avoid wrong results when searching
    faiss.normalize_L2(vector)
    #Add to index
    index.add(vector)

def normalizeL2(embeddings):
    vector = embeddings.detach().cpu().numpy()
    vector = np.float32(vector)
    faiss.normalize_L2(vector)
    return vector

def copy_similar_images(most_similar_indices, similarities, source_file_path, all_matched_imgs_path, target_dir):
    """
    将最相似的图片复制到目标文件夹中。
    :param matching_image_paths: 待匹配图片的路径列表
    :param most_similar_indices: 最相似图片的索引矩阵 (50, top_n)
    :param target_dir: 目标文件夹路径
    """
    source_file_name = Path(source_file_path).name.split('.')[0]
    # 遍历每个源图片的最相似图片索引
    target_file = os.path.join(target_dir, f"source_image_{source_file_name}")
    if not os.path.exists(target_file):
        os.makedirs(target_file)
    shutil.copy(source_file_path, os.path.join(target_file, f"source.jpg"))
    for rank, similar_idx in enumerate(most_similar_indices):
        similarity = round(similarities[rank], 2)
        matched_image_path = all_matched_imgs_path[similar_idx]
        matched_file_name = Path(matched_image_path).name.split('.')[0]
        shutil.copy(matched_image_path, os.path.join(target_file, f"rank_{rank}_{similarity}_{matched_file_name}.jpg"))
    
if __name__ == "__main__":
    #Retrieve all filenames
    source_path = 'data/images/'
    all_matched_path = '/home/luciana/workspace/others/course/vision_and_image/homework1/data/gallery/gallery/'
    result_path = 'results/'
    
    source_imgs_path = load_image_path(source_path)
    all_matched_imgs_path = load_image_path(all_matched_path)
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    # Save features ##############################################################
    #Create 2 indexes.
    index_clip = faiss.IndexFlatL2(512)
    index_dino = faiss.IndexFlatL2(768)
    #Load CLIP model and processor
    processor_clip = AutoProcessor.from_pretrained("/home/luciana/.cache/huggingface/clip-vit-base-patch32")
    model_clip = CLIPModel.from_pretrained("/home/luciana/.cache/huggingface/clip-vit-base-patch32").to(device)
    
    #Load DINOv2 model and processor
    processor_dino = AutoImageProcessor.from_pretrained('/home/luciana/.cache/huggingface/dinov2-base')
    model_dino = AutoModel.from_pretrained('/home/luciana/.cache/huggingface/dinov2-base').to(device)

    # Iterate over the dataset to extract features X2 and store features in indexes
    # for image_path in tqdm(all_matched_imgs_path):
    #     img = Image.open(image_path).convert('RGB')
    #     clip_features = extract_features_clip(img, processor_clip, model_clip)
    #     add_vector_to_index(clip_features,index_clip)
    #     dino_features = extract_features_dino(img, processor_dino, model_dino)
    #     add_vector_to_index(dino_features,index_dino)
    # #store the indexes locally
    # faiss.write_index(index_clip,result_path + "matched_clip.index")
    # faiss.write_index(index_dino,result_path + "matched_dino.index")

    # Input image
    clip_result_txt_file = open(result_path + 'clip_rankLists.txt', 'w')
    dino_result_txt_file = open(result_path + 'dino_rankLists.txt', 'w')
    clip_result_dict, dino_result_dict = defaultdict(list), defaultdict(list)
    for source_idx, source_img_path in tqdm(enumerate(source_imgs_path)):
        source_image = Image.open(source_img_path)
        source_clip_features = extract_features_clip(source_image, processor_clip, model_clip)
        source_dino_features = extract_features_dino(source_image, processor_dino, model_dino)
        source_clip_features = normalizeL2(source_clip_features)
        source_dino_features = normalizeL2(source_dino_features)
        
        # Search the top 5 images
        index_clip = faiss.read_index(result_path + "matched_clip.index")
        index_dino = faiss.read_index(result_path + "matched_dino.index")

        # Get distance and indexes of images associated
        d_dino, i_dino = index_dino.search(source_dino_features, 10)
        d_clip, i_clip = index_clip.search(source_clip_features, 10)
        i_clip_str, i_dino_str = [], []
        for c in i_clip[0]:
            matched_image_path = all_matched_imgs_path[c]
            matched_file_name = Path(matched_image_path).name.split('.')[0]
            i_clip_str.append(matched_file_name)
        for d in i_dino[0]:
            matched_image_path = all_matched_imgs_path[d]
            matched_file_name = Path(matched_image_path).name.split('.')[0]
            i_dino_str.append(matched_file_name)
        source_name = source_img_path.name.split('.')[0]
        clip_result_dict[source_name] = i_clip_str
        dino_result_dict[source_name] = i_dino_str
        # import pdb; pdb.set_trace()
        # visualization
        # copy_similar_images(i_dino[0], d_dino[0], source_img_path, all_matched_imgs_path, result_path + '/dino/')
        # copy_similar_images(i_clip[0], d_clip[0], source_img_path, all_matched_imgs_path, result_path + '/clip/')
for i in range(0, 50):
    i = str(i)
    i_clip_str = clip_result_dict[i]
    i_dino_str = dino_result_dict[i]
    clip_index_txt = ' '.join(i_clip_str)
    dino_index_txt = ' '.join(i_dino_str)
    clip_result_txt_file.write(clip_index_txt + '\n')
    dino_result_txt_file.write(dino_index_txt + '\n')

clip_result_txt_file.close()
dino_result_txt_file.close()

