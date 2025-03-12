from torchmetrics.image import PeakSignalNoiseRatio
import torch
import os
from PIL import Image
import json
from torchvision import transforms
from .metric import Metric
from tqdm import tqdm

transform_img = transforms.Compose([
        transforms.ToTensor(),
        lambda x: (x * 255).to(torch.uint8)
])

def preprocess_list(target_input, pred_input, device):
    processed_pred_input = []
    processed_target_input = []
    for target_item, pred_item in tqdm(zip(target_input, pred_input), total=len(target_input)):
        target_image = Image.open(target_item).convert('RGB')
        pred_image = Image.open(pred_item).convert('RGB')
        if target_image.size != pred_image.size:
            Warning("Target image size is not equal to pred image size in test PSNR!!!")
            pred_image = pred_image.resize(target_image.size)

        processed_pred_input.append(transform_img(pred_image).unsqueeze(0).to(device))
        processed_target_input.append(transform_img(target_image).unsqueeze(0).to(device))
    return processed_pred_input, processed_target_input

class PSNR(Metric):
    def __init__(self, device = 'cuda'):
        self.device = device
        self.psnr = PeakSignalNoiseRatio().to(self.device)
        

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
                            "args1": "image path ",
                            "args2": "image path ,
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
        processed_target_list, processed_pred_list = preprocess_list(target_list, pred_list, self.device)
        
        score = 0.0
        for pred, target in zip(processed_pred_list, processed_target_list):
            score += self.psnr(pred, target)
        score = score / len(processed_target_list)
        return "PSNR", score.detach().item(), len(target_list)

# print(get_psnr_batch("data/", keys=['gt', 'pred']))