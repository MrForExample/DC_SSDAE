import os
from datasets import load_dataset
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import argparse

# Configuration
MAX_TRAIN_IMAGES = 1281167  # Set to a number to limit, None for all images
MAX_VAL_IMAGES = 50000    # Set to a number to limit, None for all images
NUM_WORKERS = cpu_count()  # Number of parallel workers

def save_image(args):
    """Function to save a single image"""
    image, save_path = args
    try:
        image.save(save_path)
        return True
    except Exception as e:
        print(f"Error saving {save_path}: {e}")
        return False

def process_split(output_dir, raw_data_dir, split, max_images=None):
    """Process a single split with multiprocessing"""
    split_dir = os.path.join(output_dir, split)
    os.makedirs(split_dir, exist_ok=True)
    
    # Load dataset in streaming mode
    ds = load_dataset("ILSVRC/imagenet-1k", split=split, cache_dir=raw_data_dir, streaming=True)
    
    # Get class names
    class_names = ds.features['label'].names
    
    # Prepare class directories and counters
    class_dirs = {}
    for label in range(len(class_names)):
        class_name = class_names[label].split(',')[0].strip().replace(' ', '_')
        class_path = os.path.join(split_dir, class_name)
        os.makedirs(class_path, exist_ok=True)
        class_dirs[label] = {'name': class_name, 'path': class_path, 'counter': 0}
    
    # Collect batch of images to save
    batch_size = NUM_WORKERS * 10
    batch = []
    total_saved = 0
    
    with Pool(NUM_WORKERS) as pool:
        for idx, sample in enumerate(tqdm(ds, desc=f"Processing {split}", total=max_images)):
            if max_images and total_saved >= max_images:
                break
                
            image = sample['image']
            label = sample['label']
            
            if label == -1:
                continue
            
            class_info = class_dirs[label]
            img_format = image.format if image.format else 'JPEG'
            filename = f"{class_info['name']}_{class_info['counter']:08d}.{img_format.lower()}"
            save_path = os.path.join(class_info['path'], filename)
            
            batch.append((image, save_path))
            class_info['counter'] += 1
            
            # Process batch when it reaches batch_size
            if len(batch) >= batch_size:
                pool.map(save_image, batch)
                total_saved += len(batch)
                batch = []
        
        # Process remaining images in batch
        if batch:
            pool.map(save_image, batch)
            total_saved += len(batch)
    
    print(f"Saved {total_saved} images for {split} split")

if __name__ == "__main__":
    # It's better to download the dataset using the Hugging Face CLI first since remote streaming can be slow & unreliable
    # hf download ILSVRC/imagenet-1k --repo-type dataset --local-dir /workspace/raw_data

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Download and process ImageNet-1K dataset')
    parser.add_argument('--root_dir', type=str, default='/workspace', 
                        help='Root directory for storing dataset (default: /workspace)')
    args = parser.parse_args()
    root_dir = args.root_dir

    raw_data_dir = os.path.join(root_dir, "raw_data")
    output_dir = os.path.join(root_dir, "imagenet_data")

    os.makedirs(output_dir, exist_ok=True)
    
    # Process train and validation splits
    #process_split(output_dir, raw_data_dir, 'train', max_images=MAX_TRAIN_IMAGES)
    process_split(output_dir, raw_data_dir, 'validation', max_images=MAX_VAL_IMAGES)
    
    print("Dataset download and saving complete.")