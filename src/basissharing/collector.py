import torch
import torch.nn as nn
import os
import threading
import queue
import numpy as np
from collections import defaultdict
from typing import List, Dict
from contextlib import contextmanager
from tqdm import tqdm


class ShelfWriter:
    def __init__(self, shelf_dir: str):
        self.shelf_dir = shelf_dir
        os.makedirs(shelf_dir, exist_ok=True)
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.queue.join()
        self.queue.put(None)
        self.thread.join()

    def _worker(self):
        for item in iter(self.queue.get, None):
            try:
                for uid, new_xtx in item.items():
                    path = os.path.join(self.shelf_dir, f"{uid}.npy")
                    shape = new_xtx.shape

                    # Write directly to disk without loading into DRAM
                    if not os.path.exists(path):
                        mm = np.lib.format.open_memmap(
                            path, mode="w+", dtype=np.float32, shape=shape
                        )
                    else:
                        mm = np.lib.format.open_memmap(path, mode="r+")
                    mm += new_xtx
                    del mm  # flush and release

            except Exception as e:
                print(f"Error in ShelfWriter: {e}")
            finally:
                self.queue.task_done()

    def flush(self, data: Dict[str, np.ndarray]):
        if data:
            self.queue.put(data)


class InputCollector:
    def __init__(
        self,
        model: nn.Module,
        target_nn_modules: List[str],
        save_dir: str,
        dram_limit_gb: float = 4.0,
    ):
        self.model = model
        self.target_nn_modules = target_nn_modules
        self.save_dir = save_dir
        self.dram_limit_gb = dram_limit_gb
        self.current_buffer: Dict[str, np.ndarray] = defaultdict(lambda: 0)

    def _get_buffer_size_gb(self) -> float:
        return sum(a.nbytes for a in self.current_buffer.values()) / (1024**3)

    @contextmanager
    def _attach_hooks(self):
        hooks = []
        for name, module in self.model.named_modules():
            if name.split(".")[-1] not in self.target_nn_modules:
                continue

            def make_hook(n):
                def hook(_m, inp, _o):
                    X = (
                        inp[0].detach().reshape(-1, inp[0].shape[-1])
                    )  # compute on GPU in org precision
                    xtx = (X.T @ X).cpu().float().numpy()  # now store in CPU DRAM
                    self.current_buffer[n] += xtx

                return hook

            hooks.append(module.register_forward_hook(make_hook(name)))
        try:
            yield
        finally:
            for h in hooks:
                h.remove()

    def _flush_buffer(self, writer: ShelfWriter):
        if not self.current_buffer:
            return
        writer.flush(dict(self.current_buffer))
        writer.queue.join()
        self.current_buffer = defaultdict(lambda: 0)

    def collect(self, dataloader):
        with (
            ShelfWriter(self.save_dir) as writer,
            torch.no_grad(),
            self._attach_hooks(),
        ):
            self.model.eval()
            for batch in tqdm(
                dataloader, desc="Gathering inputs", total=len(dataloader), unit="batch"
            ):
                device = next(self.model.parameters()).device
                self.model(batch.to(device))

                if self._get_buffer_size_gb() > self.dram_limit_gb:
                    self._flush_buffer(writer)

            self._flush_buffer(writer)
