from tqdm import tqdm
import os
import numpy as np
import zipfile

import torch
from torch.utils.data import DataLoader

from utils.dataset_utils import TestSpecificDataset
from utils.image_io import save_image_tensor
from utils.loss_utils import CombinedLoss, CombinedLossWoVgg
from net.model import PromptIR
from options import options as opt

import lightning.pytorch as pl

os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.device)
MODEL_NAME = 'PromptIR_ensemble'


class PromptIRModel1(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.loss_fn = CombinedLoss()

        self.test_img_dict = {}

    def forward(self, x):
        return self.net(x)


class PromptIRModel2(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.loss_fn = CombinedLossWoVgg()

        self.test_img_dict = {}

    def forward(self, x):
        return self.net(x)


class PromptIRModel3(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.loss_fn = CombinedLoss()
        self.test_img_dict = {}

    def forward(self, x):
        return self.net(x)


def main():
    print("Options")
    print(opt)

    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_name1 = "PromptIR_loss"
    if opt.ckpt_dir is None or opt.ckpt_name is None:
        ckpt_path = os.path.join(opt.ckpt_dir, f'{model_name1}-best.ckpt')
    else:
        ckpt_path = os.path.join(opt.ckpt_dir, opt.ckpt_name)

    print(f"Loading model1 from {ckpt_path}")
    model1 = PromptIRModel1.load_from_checkpoint(
        ckpt_path, map_location=device
    )
    model1.eval()

    model_name2 = "PromptIR_loss_woVGG"
    if opt.ckpt_dir is None or opt.ckpt_name is None:
        ckpt_path = os.path.join(opt.ckpt_dir, f'{model_name2}-best.ckpt')
    else:
        ckpt_path = os.path.join(opt.ckpt_dir, opt.ckpt_name)

    print(f"Loading model2 from {ckpt_path}")
    model2 = PromptIRModel2.load_from_checkpoint(
        ckpt_path, map_location=device
    )
    model2.eval()

    model_name3 = "PromptIR_loss_aug"
    if opt.ckpt_dir is None or opt.ckpt_name is None:
        ckpt_path = os.path.join(opt.ckpt_dir, f'{model_name3}-best.ckpt')
    else:
        ckpt_path = os.path.join(opt.ckpt_dir, opt.ckpt_name)

    print(f"Loading model3 from {ckpt_path}")
    model3 = PromptIRModel3.load_from_checkpoint(
        ckpt_path, map_location=device
    )
    model3.eval()

    # Prepare test dataset and dataloader
    test_set = TestSpecificDataset(opt)
    test_loader = DataLoader(
        test_set, batch_size=1, pin_memory=True, shuffle=False,
        num_workers=opt.num_workers
    )

    # TTA transform
    tta_transforms = [
        lambda x: x,
        lambda x: torch.flip(x, [3]),
        lambda x: torch.flip(x, [2]),
        lambda x: torch.flip(torch.flip(x, [2]), [3])
    ]

    print('Start testing...')
    test_img_dict = {}
    with torch.no_grad():
        for [pathname], degrad_patch in tqdm(test_loader, desc="Testing"):
            degrad_patch = degrad_patch.to(device)

            tta_results = []
            for transform in tta_transforms:
                transformed_input = transform(degrad_patch)

                transformed_output1 = model1(transformed_input)
                transformed_output2 = model2(transformed_input)
                transformed_output3 = model3(transformed_input)

                # Restore
                if transform == tta_transforms[1]:
                    transformed_output1 = torch.flip(transformed_output1, [3])
                    transformed_output2 = torch.flip(transformed_output2, [3])
                    transformed_output3 = torch.flip(transformed_output3, [3])
                elif transform == tta_transforms[2]:
                    transformed_output1 = torch.flip(transformed_output1, [2])
                    transformed_output2 = torch.flip(transformed_output2, [2])
                    transformed_output3 = torch.flip(transformed_output3, [2])
                elif transform == tta_transforms[3]:
                    transformed_output1 = torch.flip(
                        torch.flip(transformed_output1, [2]), [3]
                    )
                    transformed_output2 = torch.flip(
                        torch.flip(transformed_output2, [2]), [3]
                    )
                    transformed_output3 = torch.flip(
                        torch.flip(transformed_output3, [2]), [3]
                    )

                transformed_output = (
                    transformed_output1 +
                    transformed_output2 +
                    transformed_output3
                ) / 3.0
                tta_results.append(transformed_output)

            # Average all TTA results
            restored = torch.stack(tta_results).mean(dim=0)

            # Transfer to numpy array (N, H, W, C)
            restored_np = restored.cpu().numpy().transpose(0, 2, 3, 1) * 255.0
            for (pathname_, restored_img) in zip(pathname, restored_np):
                restored_img = np.transpose(restored_img, (2, 0, 1))  # 3xHxW
                restored_img = np.clip(restored_img, 0, 255).astype(np.uint8)

                filename = pathname_.split('/')[-1]
                test_img_dict[filename] = restored_img

                # Save image
                test_save_dir = os.path.join(opt.output_path, "img")
                save_path = os.path.join(test_save_dir, filename)
                os.makedirs(test_save_dir, exist_ok=True)
                save_image_tensor(restored, save_path)

    npz_path = os.path.join(opt.output_path, f'{MODEL_NAME}_pred.npz')
    zip_path = os.path.join(opt.output_path, f'{MODEL_NAME}.zip')
    np.savez(npz_path, **test_img_dict)
    print(f"Saved {len(test_img_dict)} images to {npz_path}")
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.write(npz_path, arcname='pred.npz')
    print(f'ZIP file saved to {zip_path}')


if __name__ == '__main__':
    main()
