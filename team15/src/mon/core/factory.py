#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements factory method for registering and building classes at runtime."""

__all__ = [
    "Factory",
    "ModelFactory",
]

import copy
import inspect
from typing import Any, Callable

from mon.core import humps


# ----- Base Factory -----
class Factory(dict):
    """Base factory class for registering and building objects.

    Notes:
        Inherits from Python's ``dict`` to store registered classes.

    Args:
        name: Factory name.
    """
    
    def __init__(self, name: str, mapping: dict = None, *args, **kwargs):
        """Initializes the factory.

        Args:
            name: Factory name.
            mapping: Initial dict of registered classes. Default is ``None``.

        Raises:
            ValueError: If ``name`` is empty.
        """
        if not name:
            raise ValueError("[name] must not be empty.")
        self.name = name
        super().__init__(mapping or {})
    
    def __repr__(self) -> str:
        """Returns a string representation of the factory.

        Returns:
            String as ``Factory(name=<name>, items=<items>)``.
        """
        return f"{self.__class__.__name__}(name={self.name}, items={self})"
    
    def register(
        self,
        name   : str  = None,
        module : Any  = None,
        replace: bool = False
    ) -> Callable:
        """Registers a module or class with an optional decorator.

        Args:
            name: Module/class name, inferred if ``None``. Default is ``None``.
            module: Module/class to register. Default is ``None``.
            replace: If ``True``, overwrites existing entry. Default is ``False``.

        Returns:
            Decorator if ``module`` is ``None``, else registers directly.

        Raises:
            TypeError: If ``name`` is not a ``str`` or ``None``.
        """
        if name and not isinstance(name, str):
            raise TypeError(f"[name] must be str or None, got {type(name).__name__}.")
        
        def _register(cls):
            self.register_module(module_cls=cls, module_name=name, replace=replace)
            return cls
        
        return _register(module) if module else _register
    
    def register_module(
        self,
        module_cls : Any,
        module_name: str  = None,
        replace    : bool = False
    ):
        """Registers a module or class to the factory.

        Args:
            module_cls: Module or class to register.
            module_name: Name, inferred if ``None``. Default is ``None``.
            replace: If ``True``, overwrites existing entry. Default is ``False``.

        Raises:
            ValueError: If ``module_cls`` is not a class.
        """
        if not inspect.isclass(module_cls):
            raise ValueError(f"[module_cls] must be a class, "
                             f"got {type(module_cls).__name__}.")
        
        key = module_name or humps.kebabize(module_cls.__name__)
        if replace or key not in self:
            self[key] = module_cls
    
    def build(
        self,
        name   : str  = None,
        config : dict = None,
        to_dict: bool = False,
        **kwargs
    ):
        """Builds an instance of a registered class.

        Args:
            name: Class name, inferred from ``config`` if ``None``. Default is ``None``.
            config: Dict of class args. Default is ``None``.
            to_dict: If ``True``, returns dict with ``{name: instance}``.
                Default is ``False``.

        Returns:
            Instance or dict if ``to_dict`` is ``True``, or ``None`` if no name.

        Raises:
            ValueError: If ``name`` is not in the registry.
        """
        if not name and (not config or "name" not in config):
            return None
        
        config = copy.deepcopy(config) if config else {}
        name   = name or config.pop("name", None)
        kwargs.update(config)
        
        for candidate in [name,
                          humps.kebabize(name),
                          humps.depascalize(name),
                          humps.pascalize(name)]:
            if candidate in self:
                instance = self[candidate](**kwargs)
                if not hasattr(instance, "name"):
                    instance.name = humps.depascalize(humps.pascalize(candidate))
                return {candidate: instance} if to_dict else instance
        raise ValueError(f"[name] must be in registry, got {name}.")
    
    def build_instances(self, configs: list[Any], to_dict: bool = False, **kwargs):
        """Builds multiple instances from a list of configurations.

        Args:
            configs: List of configs (str or dict with ``name`` key).
            to_dict: If ``True``, returns dict of ``{name: instance}``.
                Default is ``False``.

        Returns:
            List or dict of instances, or ``None`` if no valid instances.

        Raises:
            ValueError: If ``configs`` is not a list or items are invalid.
        """
        if not isinstance(configs, list):
            raise ValueError(f"[configs] must be a list, got {type(configs).__name__}.")
        
        result = {} if to_dict else []
        for config in configs:
            if isinstance(config, str):
                name, config = config, {}
            elif isinstance(config, dict):
                name   = config.pop("name", None)
                config = copy.deepcopy(config)
            else:
                raise ValueError(f"[configs] items must be str or dict, "
                                 f"got {type(config).__name__}.")
            
            instance = self.build(name=name, config=config, to_dict=to_dict, **kwargs)
            if instance:
                if to_dict:
                    result.update(instance)
                else:
                    result.append(list(instance.values())[0] if isinstance(instance, dict) else instance)
        
        return result if result else None


# ----- Model Factory -----
class ModelFactory(Factory):
    """Factory for registering and building deep learning models.

    Notes:
        Inherits from ``Factory`` and organizes models by architecture.

    Example:
        >>> MODEL = ModelFactory("Model")
        >>> @MODEL.register(arch="resnet", name="resnet")
        >>> class ResNet:
        >>>     pass
        >>> resnet = MODEL.build(name="resnet", config={})
    """
    
    @property
    def archs(self) -> list[str]:
        """Returns registered architecture names.

        Returns:
            List of architecture names as strings.
        """
        return list(self)
    
    @property
    def models(self) -> list[str]:
        """Returns registered model names across all architectures.

        Returns:
            List of model names as strings from nested registries.
        """
        return [
            model for models in self.values()
            if isinstance(models, dict)
            for model in models
        ]
    
    def register(
        self,
        name   : str  = None,
        arch   : str  = None,
        module : Any  = None,
        replace: bool = False,
    ) -> Callable[[type], type]:
        """Registers a model with an optional decorator.

        Args:
            name: Model name, inferred if ``None``. Default is ``None``.
            arch: Arch name, inferred if ``None``. Default is ``None``.
            module: Model class to register. Default is ``None``.
            replace: If ``True``, overwrites entry. Default is ``False``.

        Returns:
            Decorator if ``module`` is ``None``, else registers directly.

        Raises:
            TypeError: If ``name`` is not a ``str`` or ``None``.
        """
        if name and not isinstance(name, str):
            raise TypeError(f"[name] must be str or None, got {type(name).__name__}.")
        
        def _register(cls: type) -> type:
            self.register_module(cls, name, arch, replace)
            return cls
        
        return _register(module) if module else _register
    
    def register_module(
        self,
        module_cls : Any,
        module_name: str  = None,
        arch_name  : str  = None,
        replace    : bool = False
    ):
        """Registers a model class under an architecture.

        Args:
            module_cls: Model class to register.
            module_name: Model name, inferred if ``None``. Default is ``None``.
            arch_name: Arch name, inferred if ``None``. Default is ``None``.
            replace: If ``True``, overwrites entry. Default is ``False``.

        Raises:
            ValueError: If ``module_cls`` is not a class.
        """
        if not inspect.isclass(module_cls):
            raise ValueError(f"[module_cls] must be a class, got {type(module_cls).__name__}.")
        
        module_key = module_name or humps.kebabize(module_cls.__name__)
        arch_key   = arch_name   or humps.kebabize(getattr(module_cls, "arch", module_cls.__name__))
        
        if arch_key not in self:
            self[arch_key] = {}
        if replace or module_key not in self[arch_key]:
            self[arch_key][module_key] = module_cls
    
    def build(
        self,
        name   : str  = None,
        arch   : str  = None,
        config : dict = None,
        to_dict: bool = False,
        **kwargs
    ) -> Any | dict[str, Any] | None:
        """Builds a model instance from the registry.

        Args:
            name: Model name, inferred if ``None``. Default is ``None``.
            arch: Arch name, inferred if ``None``. Default is ``None``.
            config: Dict of model args. Default is ``None``.
            to_dict: If ``True``, returns ``{name: instance}``. Default is ``False``.

        Returns:
            Model instance, dict if ``to_dict`` is ``True``, or ``None`` if no name.

        Raises:
            ValueError: If ``arch`` and ``name`` not in the registry.
        """
        if not name and (not config or "name" not in config):
            return None
        
        config = copy.deepcopy(config) if config else {}
        name   = name or config.pop("name", None)
        arch   = arch or name
        kwargs.update(config)
        
        for candidate in [name,
                          humps.kebabize(name),
                          humps.pascalize(name),
                          humps.depascalize(name)]:
            for a, models in self.items():
                if candidate in models:
                    instance = models[candidate](**kwargs)
                    if not hasattr(instance, "name"):
                        instance.name = humps.depascalize(humps.pascalize(candidate))
                    return {candidate: instance} if to_dict else instance
        
        raise ValueError(f"[arch] [{arch}] and [name] [{name}] must be in registry.")
    
    def build_instances(
        self,
        configs: list[Any],
        to_dict: bool = False,
        **kwargs
    ) -> list[Any] | dict[str, Any] | None:
        """Builds multiple model instances from a list of configs.

        Args:
            configs: List of configs (str or dict with ``name`` key).
            to_dict: If ``True``, returns dict of ``{name: instance}``.
                Default is ``False``.

        Returns:
            List or dict of instances, or ``None`` if no valid instances.

        Raises:
            ValueError: If ``configs`` is not a list or items are invalid.
        """
        if not isinstance(configs, list):
            raise ValueError(f"[configs] must be a list, got {type(configs).__name__}.")
        
        result = {} if to_dict else []
        for config in configs:
            if isinstance(config, str):
                name, arch, args = config, None, {}
            elif isinstance(config, dict):
                config = copy.deepcopy(config)
                name   = config.pop("name", None)
                arch   = config.pop("arch", None)
                args   = config
            else:
                raise ValueError(f"[configs] items must be str or dict, "
                                 f"got {type(config).__name__}.")
            
            instance = self.build(name=name, arch=arch, to_dict=to_dict, **args, **kwargs)
            if instance:
                if to_dict:
                    result.update(instance)
                else:
                    result.append(next(iter(instance.values())) if isinstance(instance, dict) else instance)
        
        return result if result else None
