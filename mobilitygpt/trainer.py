"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import time
from collections import defaultdict

import torch
from torch.utils.data.dataloader import DataLoader
from scipy.spatial import distance
from opacus import PrivacyEngine 
from opacus.validators import ModuleValidator
from opacus.utils.batch_memory_manager import BatchMemoryManager

class Trainer:

    def __init__(self, config, model, train_dataset, train_sampler, val_sampler, DP=False, eps = None):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)
        self.val_loss = None
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.DP = DP
        self.eps = eps

        # determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("running on device", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0
        self.test_num_samples = 100

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def test_statistic(self, links, porto_geo, OD_test, length_test):
        OD_synth=[]
        length_synth=[]
        for link_ids in links:
            OD_synth.append(link_ids[0])
            OD_synth.append(link_ids[-1])
            length = porto_geo[porto_geo['geo_id'].isin(link_ids)].length.sum()
            length_synth.append(length)

            
        js_OD=distance.jensenshannon(OD_synth, OD_test)
        js_length=distance.jensenshannon(length_synth, length_test)
        return js_OD, js_length

    def run(self):
        model, config = self.model, self.config

        print("--------------------------------------------------------------------------------")
        # Check if model is compatible with opacus 
        if not ModuleValidator.is_valid(model):
            print("Mobility GPT model is not compatible with DP training")
            model = ModuleValidator.fix(model)
        else:
            print("Mobility GPT model can be trained with DP")

        print("--------------------------------------------------------------------------------")

        # setup the optimizer
        self.optimizer = model.configure_optimizers(config)

        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            sampler=self.train_sampler,
            pin_memory=False,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        self.iter_num = 0
        self.iter_time = time.time()

        if self.DP:

            privacy_engine = PrivacyEngine(accountant = "rdp")
            model, self.optimizer, train_loader = privacy_engine.make_private_with_epsilon(
                module = model,
                optimizer = self.optimizer,
                data_loader = train_loader,
                epochs = config.max_iters,
                target_epsilon = self.eps,
                target_delta = config.delta,
                max_grad_norm = config.grad_norm_clip,
                )
            print("Training model with DP...")

            with BatchMemoryManager(
                data_loader=train_loader, 
                max_physical_batch_size=config.batch_size, 
                optimizer=self.optimizer
            ) as memory_safe_data_loader:

                data_iter = iter(memory_safe_data_loader)

                while True:

                    self.optimizer.zero_grad()  # Clear gradients

                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        data_iter = iter(memory_safe_data_loader)
                        batch = next(data_iter)
                    batch = [t.to(self.device) for t in batch]
                    x, y = batch

                    # forward the model
                    logits, self.loss = model(x, y)

                    # backprop and update the parameters
                    model.zero_grad(set_to_none=True)
                    self.loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    self.optimizer.step()

                    self.trigger_callbacks('on_batch_end')
                    self.iter_num += 1
                    tnow = time.time()
                    self.iter_dt = tnow - self.iter_time
                    self.iter_time = tnow

                    # termination conditions
                    if config.max_iters is not None and self.iter_num >= config.max_iters:
                        break
        else:
            data_iter = iter(train_loader)
            while True:

                self.optimizer.zero_grad()  # Clear gradients
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader)
                    batch = next(data_iter)
                batch = [t.to(self.device) for t in batch]
                x, y = batch

                # forward the model
                logits, self.loss = model(x, y)

                # backprop and update the parameters
                model.zero_grad(set_to_none=True)
                self.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                self.optimizer.step()

                self.trigger_callbacks('on_batch_end')
                self.iter_num += 1
                tnow = time.time()
                self.iter_dt = tnow - self.iter_time
                self.iter_time = tnow

                # termination conditions
                if config.max_iters is not None and self.iter_num >= config.max_iters:
                    break