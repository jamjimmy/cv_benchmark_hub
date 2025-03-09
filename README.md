# Computer Vision Benchmark Hub

Welcome to the Computer Vision Benchmark Hub — a flexible and user-friendly tool designed to evaluate and benchmark image-based models.  With support for key metrics such as SSIM, PSNR, CLIP-Score, and more, our platform enables seamless one-click evaluations, helping you compare and optimize model performance effortlessly.

## 🚀 Installation
To get started, simply install the required dependencies:
```bash
conda create -n cv_benchmark python==3.10
conda activate cv_benchmark
pip install -r requirments.txt
```

## 📁 Data Preparation
The toolkit supports two types of inputs: <big>**folders**</big> and <big>**JSON files**</big>.

### Folder Structure
Ensure your input folder is organized as follows:
```bash
input_path
├── input_dir1 
├   ├── dir1 
├   │   ├── img1.jpg # all the filename must match in each dir
├   │   ├── img2.jpg 
├   │   └── ...
├   └── dir2 
├       ├── img1.jpg
├       ├── img2.jpg
├       └── ...
├
├── input_dir2 
├   ├── dir1 
├       └── ...
├   └── dir2 
├       └── ...
...
```

### JSON Format
Alternatively, you can use a JSON file structured like this:
```json
[
    {
        "args1": "image path or text of target",
        "args2": "image path or text of pred",
    },
    
    {
            ...
    },
    {
        "args1": "image path or text of target",
        "args2": "image path or text of pred",
    }
]
```
## ⚙️ Configuration
The toolkit uses a YAML configuration file for customization.

### Input Paths
`path_key:` Filters files containing specific keywords in their names within the `--input_path ` directory.
`special_input_paths:` A list of paths that require special processing.

### Keys
Define the input keys corresponding to your JSON or folder structure:
```yaml
keys:
  pred_key: args1
  target_key: args2
```


## 📊 Usage

Run the benchmark using the following command:
```bash
python benchmark.py --config config/example.yaml --input_path example_data --output_path ./result/test1
```
The result will be saved in `./result/test1` folder. There will be a 'benchmark.xlsx' file.



## ✅ TODO

Completed
- [X] Support Diffusion Metric:SSIM, PSNR, CLIP-Score, LIPIS
- [ ] L1, L2...

- [ ] Upcoming Features

--- 


Feel free to tweak the config and data formats to suit your needs. Happy benchmarking! 🎯
