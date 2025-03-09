from torchmetrics.multimodal.clip_score import CLIPScore
import torch
import os
from PIL import Image
import json
from torchvision import transforms
from .metric import Metric

transform_img = transforms.Compose([
    transforms.ToTensor(),
    lambda x: (x * 255)
])

def check_image_path(text):
    if any(ext in text for ext in ['.jpg', '.png', '.jpeg']):
        if not os.path.exists(text):
            print("Warning: input is an image path, will load image")

def preprocess_list(input, device, batch_size):
    processed_input = []
    for item in input:
        if os.path.isfile(item):
            image = Image.open(item)
            processed_input.append(transform_img(image).to(device))
            # processed_batch_input = [torch.stack(processed_input[i:i + batch_size]) for i in range(0, len(processed_input), batch_size)]
        else:
            processed_input.append(item)
    return processed_input

def _get_clip_score(input1, input2, metric_model):
    '''
    Args:
        - input1: list
        - input2: list
        - clip_model: clip model name or path
    '''
    with torch.no_grad():
        score = metric_model(input1, input2)
    return score.detach().item()

class ClipScore(Metric):
    def __init__(self, device = 'cuda', clip_model="openai/clip-vit-base-patch16", batch_size=100):
        self.device = device
        self.metric_model = CLIPScore(model_name_or_path=clip_model).to(self.device)
        self.clip_model_name = clip_model
        self.batch_size = batch_size
    def __call__(self, input: str, keys: list=['gt', 'pred']):
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
                            "args1": "image path or text",
                            "args2": "image path or text,
                        },
                        ...
                        {
                            "args1": "image path or text",
                            "args2": "image path or text,
                        }
                    ]

            - clip_model: clip model name or clip model path. Default: "openai/clip-vit-base-patch16"
                Available models are:
                - `"openai/clip-vit-base-patch16"`
                - `"openai/clip-vit-base-patch32"`
                - `"openai/clip-vit-large-patch14-336"`
                - `"openai/clip-vit-large-patch14"`
        Returns:
            - clip score
        '''

        if os.path.isdir(input):

            dir_path_1 = os.path.join(input, keys[0])
            dir_path_2 = os.path.join(input, keys[1])
            
            imgs = os.listdir(dir_path_1)
            input1_list = []
            input2_list = []
            for file in imgs:
                assert os.path.exists(os.path.join(dir_path_2, file)), f"file not exists {os.path.join(dir_path_2, file)}"
                input1_list.append(os.path.join(dir_path_1, file))
                input2_list.append(os.path.join(dir_path_2, file))

        elif input.endswith(".json") or input.endswith(".jsonl"):
            input1_list = []
            input2_list = []
            with open(input, "r") as f:
                data = json.load(f)
            assert len(keys) == 2, f"keys must be 2, but got {keys}"
            for item in data:
                text1 = item[keys[0]]
                text2 = item[keys[1]]

                # if there is a path, check if the path exists
                check_image_path(text1)
                check_image_path(text2)

                input1_list.append(text1)
                input2_list.append(text2)
        else:
            raise ValueError(f"{input} must be dir or json file")
        
        assert len(input1_list) == len(input2_list), f"the number of images in {input} must be the same"
        
        processed_input1_list = preprocess_list(input1_list, self.device, self.batch_size)
        processed_input2_list = preprocess_list(input2_list, self.device, self.batch_size)
        # metric_model = CLIPScore(model_name_or_path=clip_model).to(self.device)
        # score = 0.0
        # for input1, input2 in zip(processed_input1_list, processed_input2_list):
        score = _get_clip_score(processed_input1_list, processed_input2_list, self.metric_model)
            
        # score = score / len(input1_list)
        return "CLIP_score_{self.clip_model_name}", score, len(input1_list)


    # print(get_clip_score_batch('/data1/jiangzj/code/benchmark/data/', clip_model="openai/clip-vit-base-patch16",keys=['gt', 'pred'], device='cuda'))