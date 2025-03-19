from scipy import spatial
import torch
import os
from PIL import Image
import json
from torchvision import transforms
from .metric import Metric

transform_img = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

def check_image_path(text):
    if any(ext in text for ext in ['.jpg', '.png', '.jpeg']):
        if not os.path.exists(text):
            print("Warning: input is an image path, will load image")

def preprocess_list(input, device):
    processed_input = []
    for item in input:
        # if os.path.isfile(item):
        image = Image.open(item)
        processed_input.append(transform_img(image).unsqueeze(0).to(device))
    return processed_input
def _get_dino_score(input1, input2, metric_model):
    '''
    Args:
        - input1: list
        - input2: list
        - dino_model: dino model name or path
    '''
    with torch.no_grad():
        img_feature1 = metric_model(input1).detach().cpu().float()
        img_feature2 = metric_model(input2).detach().cpu().float()
        similarity = 1 - spatial.distance.cosine(img_feature1.view(img_feature1.shape[1]), img_feature2.view(img_feature2.shape[1]))
    return similarity

class DinoScore(Metric):
    def __init__(self, device = 'cuda', dino_model="'dino_vits16'", batch_size=100):
        self.device = device
        self.batch_size = batch_size
        self.dino_model = torch.hub.load('facebookresearch/dino:main', dino_model)
        self.dino_model.eval()
        self.dino_model.to(self.device)

    def __call__(self, input: str, keys=['target', 'pred']):
        '''
        Args:
            - input: path(.json or .jsonl or dir) and param::keys = ['dir1', 'dir2']:
                - if dir, there must be 2 dirs in the input dir, and 
                    the number and filename of images in the two dirs must be the same. The dir looks like this:
                    
                    input_dir
                    ├── dir1
                    │   ├── img1.jpg
                    │   ├── img2.jpg
                    │   └── ...
                    └── dir2
                        ├── img1.jpg
                        ├── img2.jpg
                        └── ...

                - if .json or .jsonl the file looks like this and param::keys = ['args1', 'args2']:
                    [
                        {
                            "args1": "image path",
                            "args2": "image path,
                        },
                        ...
                        {
                            "args1": "image path",
                            "args2": "image path,
                        }
                    ]
        Returns:
            - psnr score
        '''
        target_list = []
        pred_list = []
        
        if os.path.isdir(input):
            dir_path_1 = os.path.join(input, keys[0])
            dir_path_2 = os.path.join(input, keys[1])
            assert len(os.listdir(dir_path_1)) == len(os.listdir(dir_path_2)), f"the number of images in {dir_path_1} and {dir_path_2} must be the same"
            imgs = os.listdir(dir_path_1)
            for file in imgs:
                assert os.path.exists(os.path.join(dir_path_2, file)), f"file not exists {os.path.join(dir_path_2, file)}"
                target_list.append(os.path.join(dir_path_1, file))
                pred_list.append(os.path.join(dir_path_2, file))

        elif input.endswith(".json") or input.endswith(".jsonl"):
            with open(input, "r") as f:
                data = json.load(f)
            assert len(keys) == 2, f"keys must be 2, but got {keys}"
            for item in data:
                target_path = item[keys[0]]
                pred_path = item[keys[1]]
                target_list.append(target_path)
                pred_list.append(pred_path)
        
        else:
            raise ValueError(f"{input} must be dir or json file")
        
        assert len(target_list) == len(pred_list), f"the number of images in {input} must be the same"
        # processed_target_list, processed_pred_list = preprocess_list(target_list, pred_list, self.device)
        processed_target_list = preprocess_list(target_list, self.device)
        processed_pred_list = preprocess_list(pred_list, self.device)
        score = 0.0
        
        # print(processed_target_list, processed_pred_list)

        with torch.no_grad():
            for pred, target in zip(processed_pred_list, processed_target_list):
                score += _get_dino_score(pred, target, self.dino_model)
        score = score / len(processed_target_list)
        return "DINO", score, len(target_list)

# print(get_ssim_batch("data/", keys=['gt', 'pred']))