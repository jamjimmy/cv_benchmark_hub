import yaml
import importlib
import pandas as pd
import argparse
import os
import shutil
from datetime import datetime
# import wandb


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_class(class_path):
    module_path, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls

def postprocess(results, args, metric_name):
    df = pd.DataFrame(results)
    os.makedirs(args.output_path, exist_ok=True)
    excel_path = os.path.join(args.output_path, 'benchmark.xlsx')
    mode = 'a' if os.path.exists(excel_path) else 'w'
    if mode == 'a':
        with pd.ExcelWriter(excel_path, mode=mode, engine='openpyxl', if_sheet_exists='replace') as writer:
            if metric_name in writer.book.sheetnames:
                existing_df = pd.read_excel(excel_path, sheet_name=metric_name)
                df = pd.concat([existing_df, df], ignore_index=True)
            df.to_excel(writer, sheet_name=metric_name, index=False)
    else:
        with pd.ExcelWriter(excel_path, mode=mode, engine='openpyxl') as writer:
            if metric_name in writer.book.sheetnames:
                existing_df = pd.read_excel(excel_path, sheet_name=metric_name)
                df = pd.concat([existing_df, df], ignore_index=True)
            df.to_excel(writer, sheet_name=metric_name, index=False)
    destination_file = os.path.join(args.output_path, os.path.basename(args.config))
    if not os.path.exists(destination_file):
        shutil.copy(args.config, destination_file)

def main(config, args):

    
    # columns = ['experiment_name', 'score', 'path', 'pred_key', 'target_key', 'current_datetime']
    # table = wandb.Table(columns=columns)

    metrics = config.get('metrics', {})
    input_paths = []
    path_key = config.get('path_key', None)
    for file in os.listdir(args.input_path):
        if path_key == None or path_key in file:
            input_paths.append(os.path.join(args.input_path, file))

    targetKey = config['keys']['target']
    pred_key = config['keys']['pred']
    
    for metric_name, metric_config in metrics.items():
        
        # Instantiate metric class
        class_name = metric_config['metric']['class']
        MetricClass = get_class(class_name)
        metric_instance = MetricClass(**{k: v for k, v in metric_config['metric'].items() if k != 'class'})

        # Special keys and paths
        targetKey_input = metric_config.get('special_target_key', targetKey)
        pred_key_input = metric_config.get('special_pred_key', pred_key)
        
        all_input_paths = metric_config.get('special_input_paths', [])
        if metric_config.get('just_special_input_paths', False) == False:
            all_input_paths = input_paths + all_input_paths
        assert(len(all_input_paths) > 0), "No input paths found"
        results = []
        # wandb.log({"score": 0.9})
        # Run metric on all input paths
        for path in all_input_paths:
            print(f"Running {metric_name} on {path}")
            experiment_name, score, count = metric_instance(path, keys=[targetKey_input, pred_key_input]) # xx_score: xx
            result = {'experiment_name': experiment_name, 'score': score, 'count': count, 'path': path, 'pred_key': pred_key_input, 'target_key': targetKey_input, 'current_datetime': datetime.now()}
            results.append(result)
            # table.add_data(experiment_name, score, path, pred_key_input, targetKey_input, datetime.now())
        postprocess(results, args, metric_name)

       
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Benchmarking')
    parser.add_argument('--output_path', type=str, default="./result", help='output_path')
    parser.add_argument('--input_path', type=str, default="./example_data", help='output_path')
    parser.add_argument('--config', type=str, default="config/example.yaml", help='Path to config file')
    # parser.add_argument("--use_wandb", type=bool, default=True, help="Enable or disable wandb logging.")
    args = parser.parse_args()

    # if args.use_wandb:
    #     wandb.init(project='benchmark', name=os.path.basename(args.output_path), mode="online")
    # else:
    #     wandb.init(mode="disabled")
    args.output_path = os.path.join(args.output_path, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    config = load_config(args.config)
    main(config, args)