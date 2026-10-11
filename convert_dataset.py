import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm

def convert_coco_to_yolo_seg(coco_base_dir, output_dir):
    coco_base_dir = Path(coco_base_dir)
    output_dir = Path(output_dir)
    
    # Category mapping: COCO id 1 ('hold') -> 0, id 2 ('volume') -> 1
    cat_map = {1: 0, 2: 1}
    
    for split in ['train', 'valid', 'test']:
        split_dir = coco_base_dir / split
        json_path = split_dir / '_annotations.coco.json'
        
        if not json_path.exists():
            continue
            
        out_img_dir = output_dir / split / 'images'
        out_lbl_dir = output_dir / split / 'labels'
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nLoading {split} JSON...")
        with open(json_path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
            
        images = {img['id']: img for img in coco_data['images']}
        img_annotations = {img['id']: [] for img in coco_data['images']}
        
        for ann in coco_data['annotations']:
            cid = ann['category_id']
            if cid in cat_map:
                img_annotations[ann['image_id']].append(ann)
                
        print(f"Writing {split} YOLO labels and linking images...")
        for img_id, anns in tqdm(img_annotations.items(), desc=f"Converting {split}"):
            img_info = images[img_id]
            file_name = img_info['file_name']
            w = img_info['width']
            h = img_info['height']
            
            src_img = split_dir / file_name
            dst_img = out_img_dir / file_name
            
            # Hardlink if possible (instant, 0 extra disk space), fallback to copy
            if src_img.exists() and not dst_img.exists():
                try:
                    os.link(src_img, dst_img)
                except Exception:
                    shutil.copyfile(src_img, dst_img)
                    
            # Write normalized YOLO polygon segmentation txt
            txt_name = Path(file_name).stem + '.txt'
            txt_path = out_lbl_dir / txt_name
            
            lines = []
            for ann in anns:
                cls_idx = cat_map[ann['category_id']]
                segs = ann.get('segmentation', [])
                if isinstance(segs, list):
                    for poly in segs:
                        if len(poly) >= 6:
                            # Normalize coordinates to [0, 1] and clamp
                            norm_coords = []
                            for i in range(0, len(poly), 2):
                                nx = min(1.0, max(0.0, poly[i] / w))
                                ny = min(1.0, max(0.0, poly[i+1] / h))
                                norm_coords.extend([f"{nx:.6f}", f"{ny:.6f}"])
                            lines.append(f"{cls_idx} " + " ".join(norm_coords) + "\n")
                            
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
                
    # Create data.yaml
    yaml_content = f"""path: {output_dir.as_posix()}
train: train/images
val: valid/images
test: test/images

nc: 2
names:
  0: hold
  1: volume
"""
    with open(output_dir / 'data.yaml', 'w') as f:
        f.write(yaml_content)
        
    print(f"\nFinished! Dataset ready at: {output_dir}")

if __name__ == '__main__':
    convert_coco_to_yolo_seg(
        coco_base_dir=r"c:\Users\Wojtek\PycharmProjects\PythonProject\datasets\Climbing Hold Detection.coco-segmentation",
        output_dir=r"c:\Users\Wojtek\PycharmProjects\PythonProject\datasets\climbing_hold_seg_yolo"
    )
