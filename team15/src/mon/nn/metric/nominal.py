#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implement nominal metrics."""

__all__ = [
    "CramersV",
    "FleissKappa",
	"PearsonsContingencyCoefficient",
	"TheilsU",
	"TschuprowsT",
]

from torchmetrics.nominal import (
    CramersV, FleissKappa, PearsonsContingencyCoefficient, TheilsU, TschuprowsT,
)

from mon.constants import METRICS

# ----- Registering -----
METRICS.register(name="cramers_v",                        module=CramersV)
METRICS.register(name="fleiss_kappa",                     module=FleissKappa)
METRICS.register(name="pearsons_contingency_coefficient", module=PearsonsContingencyCoefficient)
METRICS.register(name="theils_u",                         module=TheilsU)
METRICS.register(name="tschuprows_t",                     module=TschuprowsT)
