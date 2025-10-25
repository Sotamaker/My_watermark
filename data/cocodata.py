import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pycocotools.coco import COCO
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, CenterCrop, InterpolationMode

from torchvision.transforms.functional import to_pil_image




class COCODataset(Dataset):
    def __init__(self, opt):
        """
        root: 图片文件夹路径，例如 'coco/train2017'
        ann_file: 标注文件路径，例如 'coco/annotations/instances_train2017.json'
        transforms: 图像和 mask 的变换函数
        """
        self.img_path = opt['img_path']
        self.coco = COCO(opt['ann_file'])
        self.data_len = opt['data_len']
        self.image_size =  opt['image_size']

        self.img_ids = list(self.coco.imgs.keys())
        
        if self.data_len > 0:
            self.img_ids = self.img_ids[:self.data_len]

        self.img_transforms = Compose([        
            Resize(self.image_size),
            CenterCrop(self.image_size),
            ToTensor(),
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.mask_transforms = Compose([        
            Resize(self.image_size, interpolation=InterpolationMode.NEAREST),
            CenterCrop(self.image_size),
        ])


    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.img_ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = coco.loadAnns(ann_ids)

        # 加载图像
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_path, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        #image.save(f"{img_id}.png")
        H, W = img_info['height'], img_info['width']
        # 生成 mask
        chosen_mask = None
        for ann in anns:
            mask_instance = coco.annToMask(ann).astype(np.float32)
            area = mask_instance.sum()

            # 过滤太小或太大的实例
            if area < 0.01 * H * W:
                continue
            if area > 0.8 * H * W:
                continue

            chosen_mask = mask_instance
            break  # 找到一个合适的就立即返回

        # 如果没有找到合适的实例，就返回全 0 mask
        if chosen_mask is None:
            chosen_mask = np.zeros((H, W), dtype=np.float32)

        mask = torch.tensor(chosen_mask, dtype=torch.float32)
        if self.img_transforms:
            image = self.img_transforms(image)
            # view resize img
            # img_tensor = (image + 1) / 2
            # img_pil = to_pil_image(img_tensor)  # 自动处理 [0,1] float tensor
            # img_pil.save(f"{img_id}_resize.png")
        if self.mask_transforms:
            mask = self.mask_transforms(mask.unsqueeze(0))
            binary_mask = torch.where(mask < 0.5, torch.zeros_like(mask), torch.ones_like(mask))
            # view mask img
            # mask_uint8 = (mask.squeeze(0) * 255).byte().numpy()
            # mask_pil = Image.fromarray(mask_uint8, mode='L')
            # mask_pil.save(f"binary_mask_{img_id}_resize.png")
        return {'img':image,'mask': binary_mask}







# if __name__ == '__main__':
#     ## Sec Net config
#     dataset = COCODataset('/mnt/h/dataset/coco2017/train2017/train2017','/mnt/h/dataset/coco2017/annotations/annotations/instances_train2017.json',image_size=512)
#     dataloader = DataLoader(dataset=dataset,batch_size=2)
#     for i,data in enumerate(dataloader):
#         print(i)
#     print('end')