import hydra
from hydra.utils import instantiate
from lightning import seed_everything
from omegaconf import OmegaConf, open_dict
import torch
import wandb


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(config):
    # set to prevent warning
    torch.set_float32_matmul_precision("high")

    # reproducibility
    seed_everything(config.seed, workers=True)

    # dataset + dataloader = lightning datamodule
    datamodule = instantiate(config.datamodule)

    # network + loss + optimizer = lightning module
    network = instantiate(config.network)
    loss_fns = instantiate(config.loss_fns)
    optimizer = instantiate(config.optimizer)
    litmodule = instantiate(config.litmodule, network, loss_fns, optimizer)

    # callbacks
    callbacks = instantiate(config.callbacks)

    # logger
    # NOTE: https://docs.wandb.ai/guides/app/features/panels/code/
    # stores code as an artifact but doesn't work that well yet
    logger = instantiate(config.logger)
    if logger is not None:
        logger.log_hyperparams(OmegaConf.to_container(config, resolve=True, throw_on_missing=True))
    else:
        logger = False

    # resume_from_checkpoint is not a Trainer ctor arg in Lightning 2+;
    # pop it before instantiation and forward to trainer.fit(ckpt_path=...)
    with open_dict(config.trainer):
        ckpt_path = config.trainer.pop("resume_from_checkpoint", None)

    # trainer and train!
    trainer = instantiate(
        config.trainer,
        logger=logger,
        callbacks=[cb for cb in callbacks.values()],
    )
    trainer.fit(litmodule, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
