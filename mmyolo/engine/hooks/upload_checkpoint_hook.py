from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class UploadCheckpointHook(Hook):
    def after_train_epoch(self, runner):
        wandb = runner.visualizer.get_backend("WandbVisBackend").experiment
        # https://docs.wandb.ai/ref/python/save#docusaurus_skipToContent_fallback
        wandb.save("checkpoint.pth", policy="now")
