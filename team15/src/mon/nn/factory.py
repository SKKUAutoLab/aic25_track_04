#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements factory classes for optimizers and LR schedulers."""

__all__ = [
    "LRSchedulerFactory",
    "OptimizerFactory",
]

import copy

import humps
from torch import nn, optim

import mon
from mon.core import factory


# ----- Factory -----
class OptimizerFactory(factory.Factory):
    """Registers and builds optimizers for neural networks."""

    def build(
        self,
        network: nn.Module,
        name   : str  = None,
        config : dict = None,
        network_params_only: bool = False,
        to_dict: bool = False,
        **kwargs
    ) -> any:
        """Builds an optimizer instance by name or config.

        Args:
            network: Neural network providing parameters.
            name: Optimizer name. Default is ``None``.
            config: Optimizer config dict. Default is ``None``.
            network_params_only: Use only network params if ``True``. Default is ``False``.
            to_dict: Return dict of {name: instance} if ``True``. Default is ``False``.
            **kwargs: Additional optimizer arguments.
        
        Raises:
            AssertionError: If name missing when required.
        
        Returns:
            Optimizer instance, dict, or ``None``.
        """
        if name is None and config is None:
            return None
        if name is None and config:
            if "name" not in config:
                raise AssertionError("[name] must be in config, got None.")
            config_  = copy.deepcopy(config)
            name_    = config_.pop("name")
            name     = name or name_
            kwargs  |= config_
        if not name or name not in self:
            raise AssertionError(f"[name] must be valid and registered, got {name}.")

        if hasattr(network, "named_parameters"):
            params = []
            if network_params_only:
                for n, p in network.named_parameters():
                    if all(x not in n for x in ["loss", "train/", "val/", "test/"]):
                        params.append(p)
            else:
                params = network.parameters()
            instance = self[name](params=params, **kwargs)

            if getattr(instance, "name", None) is None:
                instance.name = humps.depascalize(humps.pascalize(name))
            return {name: instance} if to_dict else instance

        return None

    def build_instances(
        self,
        network: nn.Module,
        configs: list,
        network_params_only: bool = True,
        to_dict: bool = False,
        **kwargs
    ) -> any:
        """Builds multiple optimizer instances from configs.

        Args:
            network: Neural network providing parameters.
            configs: List of optimizer names or config dicts with name.
            network_params_only: Use only network params if ``True``. Default is ``True``.
            to_dict: Return dict of {name: instance} if ``True``. Default is ``False``.
            **kwargs: Additional optimizer arguments.
        
        Raises:
            AssertionError: If configs is not a list.
       
        Returns:
            List or dict of optimizers, or ``None`` if empty.
        """
        if configs is None:
            return None
        if not isinstance(configs, list):
            raise AssertionError(f"[configs] must be list, got {type(configs).__name__}.")

        configs_   = copy.deepcopy(configs)
        optimizers = {} if to_dict else []
        for config in configs_:
            if isinstance(config, str):
                name = config
            else:
                name    = config.pop("name")
                kwargs |= config
            opt = self.build(
                network             = network,
                name                = name,
                network_params_only = network_params_only,
                to_dict             = to_dict,
                **kwargs
            )
            if opt:
                if to_dict:
                    optimizers |= opt
                else:
                    optimizers.append(opt)

        return optimizers if optimizers else None
    

class LRSchedulerFactory(factory.Factory):
    """Registers and builds learning rate schedulers."""

    def build(
        self,
        optimizer: optim.Optimizer,
        name     : str  = None,
        config   : dict = None,
        **kwargs
    ) -> any:
        """Builds a scheduler instance by name or config.

        Args:
            optimizer: Optimizer for the scheduler.
            name: Scheduler name. Default is ``None``.
            config: Scheduler config dict. Default is ``None``.
            **kwargs: Additional scheduler arguments.
        
        Raises:
            AssertionError: If name missing or invalid.
        
        Returns:
            Scheduler instance or ``None``.
        """
        if name is None and config is None:
            return None
        if name is None and config:
            if "name" not in config:
                raise AssertionError("[name] must be in config, got None.")
            config_ = copy.deepcopy(config)
            name_   = config_.pop("name")
            name    = name or name_
            kwargs |= config_
        if not name or name not in self:
            raise AssertionError(f"[name] must be valid and registered, got {name}.")

        if name in ["GradualWarmupScheduler",
                    "gradual_warmup_scheduler",
                    "gradual-warmup-scheduler"]:
            after_scheduler = kwargs.pop("after_scheduler")
            if isinstance(after_scheduler, dict):
                name_ = after_scheduler.pop("name")
                after_scheduler = self[name_](optimizer=optimizer, **after_scheduler) if name_ in self else None
            return self[name](optimizer=optimizer, after_scheduler=after_scheduler, **kwargs)

        return self[name](optimizer=optimizer, **kwargs)

    def build_instances(
        self,
        optimizer: optim.Optimizer,
        configs  : list,
        **kwargs
    ) -> list:
        """Builds multiple scheduler instances from configs.

        Args:
            optimizer: Optimizer for the schedulers.
            configs: List of scheduler names or config dicts with name.
            **kwargs: Additional scheduler arguments.
        
        Raises:
            AssertionError: If configs is not a list.
        
        Returns:
            List of schedulers or ``None`` if empty.
        """
        if configs is None:
            return None
        if not isinstance(configs, list):
            raise AssertionError(f"[configs] must be list, got {type(configs).__name__}.")

        configs_   = copy.deepcopy(configs)
        schedulers = []
        for config in configs_:
            if isinstance(config, str):
                name = config
            else:
                name    = config.pop("name")
                kwargs |= config
            scheduler = self.build(optimizer=optimizer, name=name, **kwargs)
            if scheduler:
                schedulers.append(scheduler)

        return schedulers if schedulers else None


# ----- Registering -----
mon.constants.OPTIMIZERS    = OptimizerFactory("Optimizer")
mon.constants.LR_SCHEDULERS = LRSchedulerFactory("LRScheduler")
