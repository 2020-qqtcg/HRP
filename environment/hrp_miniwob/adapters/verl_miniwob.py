"""verl engine and AdamW checkpoint support for the paper's Exact-PG updates."""

import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
from hrp_miniwob.lm_policy import LanguagePolicy
from verl.workers.config import HFModelConfig, TrainingWorkerConfig
from verl.workers.config.engine import EngineConfig
from verl.workers.config.optimizer import FSDPOptimizerConfig, build_optimizer
from verl.workers.engine import BaseEngine, EngineRegistry
from verl.workers.engine_workers import TrainingWorker


@dataclass
class SemanticEngineConfig(EngineConfig):
    strategy: str = "hrp_canonical"
    device: str = "cuda:0"
    lora_rank: int = 8
    use_dynamic_bsz: bool = False
    use_remove_padding: bool = False
    micro_batch_size_per_gpu: int = 1
    infer_micro_batch_size_per_gpu: int = 1
    infer_use_dynamic_bsz: bool = False


@EngineRegistry.register(model_type="hrp_actor", backend="hrp_canonical")
class SemanticEngine(BaseEngine):
    def __init__(self, model_config, engine_config, optimizer_config, checkpoint_config):
        self.model_config, self.engine_config = model_config, engine_config
        self.optimizer_config = optimizer_config
        self.role = model_config.model_type
        self.updates = 0

    def initialize(self):
        self.policy = LanguagePolicy(self.model_config.path, seed=self.engine_config.seed,
                                     device=self.engine_config.device, rank=self.engine_config.lora_rank)
        self.policy.metadata["optimizer"] = "Exact-PG with verl TrainingWorker"
        self.policy.metadata["decoder"]["optimizer_updates"] = "mean task gradients per optimizer update"
        self.parameters = [p for p in self.policy.model.parameters() if p.requires_grad]
        if any(p.dtype != torch.float32 for p in self.parameters):
            raise AssertionError("Trainable LoRA parameters must retain FP32 master weights")
        self.optimizer = build_optimizer(self.parameters, self.optimizer_config)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1.0)

    @property
    def is_param_offload_enabled(self):
        return False

    @property
    def is_optimizer_offload_enabled(self):
        return False

    @contextmanager
    def train_mode(self, **kwargs):
        previous = self.policy.model.training
        self.policy.model.train()
        try:
            yield
        finally:
            self.policy.model.train(previous)

    @contextmanager
    def eval_mode(self, **kwargs):
        previous = self.policy.model.training
        self.policy.model.eval()
        try:
            yield
        finally:
            self.policy.model.train(previous)


    def optimizer_zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def optimizer_step(self):
        norm = torch.nn.utils.clip_grad_norm_(self.parameters, 1.0, error_if_nonfinite=True)
        self.optimizer.step()
        self.updates += 1
        return norm

    def lr_scheduler_step(self):
        self.scheduler.step()
        return self.optimizer.param_groups[0]["lr"]

    def get_data_parallel_size(self):
        return 1

    def get_data_parallel_rank(self):
        return 0

    def get_data_parallel_group(self):
        return None

    def is_mp_src_rank_with_outputs(self):
        return True

    def to(self, device, model=True, optimizer=True, grad=True):
        if str(device) not in {str(self.policy.device), "cuda", "device"}:
            raise ValueError("This single-device engine keeps parameters and optimizer resident")

    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None, **kwargs):
        path = Path(local_path)
        self.policy.save_adapter(path / "adapter")
        torch.save({"optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
                    "updates": self.updates, "value_head": None}, path / "training.pt")
        (path / "engine.json").write_text(json.dumps({"role": self.role, "global_step": global_step,
            "updates": self.updates, "trainable_parameters": sum(p.numel() for p in self.parameters),
            "backend": "verl-TrainingWorker/BaseEngine/hrp_canonical", "dtype": "bf16",
            "trainable_dtype": "float32"}, indent=2) + "\n")

    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False, **kwargs):
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        path = Path(local_path)
        set_peft_model_state_dict(self.policy.model, load_file(path / "adapter/adapter_model.safetensors"))
        state = torch.load(path / "training.pt", map_location=self.policy.device, weights_only=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.updates = state["updates"]

    def initialize_adapter(self, adapter_path):
        """Start a new optimization run from adapter weights with fresh moments."""
        from peft import get_peft_model_state_dict, set_peft_model_state_dict
        from safetensors.torch import load_file

        if self.updates or self.optimizer.state:
            raise ValueError("Adapter initialization requires a fresh optimizer")
        weights = load_file(Path(adapter_path) / "adapter_model.safetensors")
        current = get_peft_model_state_dict(self.policy.model)
        if weights.keys() != current.keys() or any(weights[k].shape != current[k].shape for k in weights):
            raise ValueError("Source adapter architecture differs from the new worker")
        set_peft_model_state_dict(self.policy.model, weights)
        actual = get_peft_model_state_dict(self.policy.model)
        if any(not torch.equal(actual[k].detach().cpu(), weights[k]) for k in weights):
            raise AssertionError("Source adapter weights did not load exactly")
        self.policy.metadata["initial_adapter"] = str(Path(adapter_path).resolve())


def make_worker(model_path, role, device, seed, lr, rank=8):
    config = TrainingWorkerConfig(
        model_type=f"hrp_{role}",
        model_config=HFModelConfig(path=model_path, load_tokenizer=False,
                                  use_remove_padding=False, enable_gradient_checkpointing=False),
        engine_config=SemanticEngineConfig(device=device, seed=seed, lora_rank=rank),
        optimizer_config=FSDPOptimizerConfig(lr=lr, betas=(0.9, 0.999), weight_decay=0,
            lr_warmup_steps=0, total_training_steps=1024, override_optimizer_config={"eps": 1e-8}),
    )
    worker = TrainingWorker(config)
    worker.reset()
    return worker
