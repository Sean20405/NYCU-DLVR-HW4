import os
import numpy as np
import zipfile

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from utils.dataset_utils import PromptTrainDataset, TestSpecificDataset
from utils.schedulers import LinearWarmupCosineAnnealingLR
from utils.val_utils import compute_psnr_ssim
from utils.image_io import save_image_tensor
from utils.loss_utils import CombinedLossWoVgg
from net.model import PromptIR
from options import options as opt

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.device)
MODEL_NAME = 'PromptIR_loss_woVGG'


class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.loss_fn = CombinedLossWoVgg()

        self.test_img_dict = {}

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        total_loss, loss_dict = self.loss_fn(restored, clean_patch)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", total_loss)
        for key, value in loss_dict.items():
            self.log(f"train_{key}", value)
        return total_loss

    def validation_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        total_loss, loss_dict = self.loss_fn(restored, clean_patch)
        self.log(
            "val_loss", total_loss,
            on_step=False, on_epoch=True, sync_dist=True
        )
        for key, value in loss_dict.items():
            self.log(
                f"val_{key}", value,
                on_step=False, on_epoch=True, sync_dist=True
            )

        # Logging additional metrics
        psnr, ssim, N = compute_psnr_ssim(restored, clean_patch)
        self.log(
            "val_psnr", psnr,
            on_step=False, on_epoch=True, sync_dist=True
        )
        self.log(
            "val_ssim", ssim,
            on_step=False, on_epoch=True, sync_dist=True
        )

        return total_loss

    def test_step(self, batch, batch_idx):
        ([pathname], degrad_patch) = batch
        restored = self.net(degrad_patch)

        # Transfer to numpy array (N, H, W, C)
        restored_np = restored.cpu().numpy().transpose(0, 2, 3, 1) * 255.0
        for (pathname_, restored_img) in zip(pathname, restored_np):
            restored_img = np.transpose(restored_img, (2, 0, 1))  # 3 x H x W
            restored_img = np.clip(restored_img, 0, 255).astype(np.uint8)

            filename = pathname_.split('/')[-1]
            self.test_img_dict[filename] = restored_img

            # Save image
            test_save_dir = os.path.join(opt.output_path, "img")
            save_path = os.path.join(test_save_dir, filename)
            os.makedirs(test_save_dir, exist_ok=True)
            save_image_tensor(restored, save_path)

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=opt.lr)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer, warmup_epochs=15, max_epochs=150
        )

        return [optimizer], [scheduler]


def main():
    print("Options")
    print(opt)

    if opt.train:
        if opt.wblogger is not None:
            logger = WandbLogger(project=opt.wblogger, name=MODEL_NAME)
        else:
            logger = TensorBoardLogger(save_dir="logs/", name=MODEL_NAME)

        full_train_set = PromptTrainDataset(opt)
        train_set, valid_set = random_split(full_train_set, [0.8, 0.2])
        train_loader = DataLoader(
            train_set,
            batch_size=opt.batch_size,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
            num_workers=opt.num_workers
        )
        val_loader = DataLoader(
            valid_set,
            batch_size=opt.batch_size,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
            num_workers=opt.num_workers
        )

        best_ckpt_callback = ModelCheckpoint(
            dirpath=opt.ckpt_dir,
            filename=f'{MODEL_NAME}-best',
            every_n_epochs=1,
            save_top_k=1,
            monitor='val_psnr',
            mode='max',
        )
        # periodic_ckpt_callback = ModelCheckpoint(
        #     dirpath=opt.ckpt_dir,
        #     filename=MODEL_NAME+'-{epoch}',
        #     save_top_k=-1,
        #     every_n_epochs=20,
        # )
        lr_monitor = LearningRateMonitor(logging_interval='step')

        model = PromptIRModel()

        trainer = pl.Trainer(
            max_epochs=opt.epochs,
            accelerator="gpu",
            devices=opt.num_gpus,
            strategy="ddp_find_unused_parameters_true",
            logger=logger,
            callbacks=[best_ckpt_callback, lr_monitor]
        )
        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader
        )

        print(
            f"Training completed. Best checkpoint saved to "
            f"{best_ckpt_callback.best_model_path} with PSNR "
            f"{best_ckpt_callback.best_model_score:.4f}"
        )
    elif opt.test:
        # Load the model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if opt.ckpt_dir is None or opt.ckpt_name is None:
            ckpt_path = os.path.join(opt.ckpt_dir, f'{MODEL_NAME}-best.ckpt')
        else:
            ckpt_path = os.path.join(opt.ckpt_dir, opt.ckpt_name)
        print(f"Loading model from {ckpt_path}")
        model = PromptIRModel.load_from_checkpoint(
            ckpt_path, map_location=device
        )
        model.eval()

        # Prepare test dataset and dataloader
        test_set = TestSpecificDataset(opt)
        test_loader = DataLoader(
            test_set, batch_size=1, pin_memory=True, shuffle=False,
            num_workers=opt.num_workers
        )

        print('Start testing...')
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=opt.num_gpus,
            strategy="ddp_find_unused_parameters_true",
            logger=None  # No logger needed for testing
        )

        trainer.test(model=model, dataloaders=test_loader)

        npz_path = os.path.join(opt.output_path, f'{MODEL_NAME}_pred.npz')
        zip_path = os.path.join(opt.output_path, f'{MODEL_NAME}.zip')
        np.savez(npz_path, **model.test_img_dict)
        print(f"Saved {len(model.test_img_dict)} images to {npz_path}")

        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.write(npz_path, arcname='pred.npz')
        print(f'ZIP file saved to {zip_path}')
    else:
        print("Please specify --train or --test mode.")


if __name__ == '__main__':
    main()
