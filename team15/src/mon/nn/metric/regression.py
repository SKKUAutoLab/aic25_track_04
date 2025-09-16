#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implement regression metrics."""

__all__ = [
    "ConcordanceCorrCoef",
    "CosineSimilarity",
    "CriticalSuccessIndex",
    "ExplainedVariance",
    "KLDivergence",
    "KendallRankCorrCoef",
    "LogCoshError",
    "MeanAbsoluteError",
    "MeanAbsolutePercentageError",
    "MeanSquaredError",
    "MeanSquaredLogError",
    "MinkowskiDistance",
    "PearsonCorrCoef",
    "R2Score",
    "RelativeSquaredError",
    "SpearmanCorrCoef",
    "SymmetricMeanAbsolutePercentageError",
    "TweedieDevianceScore",
    "WeightedMeanAbsolutePercentageError",
]

from torchmetrics.regression import (
    ConcordanceCorrCoef, CosineSimilarity, CriticalSuccessIndex, ExplainedVariance,
    KendallRankCorrCoef, KLDivergence, LogCoshError, MeanAbsoluteError,
    MeanAbsolutePercentageError, MeanSquaredError, MeanSquaredLogError,
    MinkowskiDistance, PearsonCorrCoef, R2Score, RelativeSquaredError, SpearmanCorrCoef,
    SymmetricMeanAbsolutePercentageError, TweedieDevianceScore,
    WeightedMeanAbsolutePercentageError,
)

from mon.constants import METRICS

# ----- Registering -----
METRICS.register(name="concordance_corr_coef",                    module=ConcordanceCorrCoef)
METRICS.register(name="cosine_similarity",                        module=CosineSimilarity)
METRICS.register(name="critical_success_index",                   module=CriticalSuccessIndex)
METRICS.register(name="explained_variance",                       module=ExplainedVariance)
METRICS.register(name="kendall_rank_corr_coef",                   module=KendallRankCorrCoef)
METRICS.register(name="kl_divergence",                            module=KLDivergence)
METRICS.register(name="log_cosh_error",                           module=LogCoshError)
METRICS.register(name="mae",                                      module=MeanAbsoluteError)
METRICS.register(name="mean_absolute_error",                      module=MeanAbsoluteError)
METRICS.register(name="mean_absolute_percentage_error",           module=MeanAbsolutePercentageError)
METRICS.register(name="mean_squared_error",                       module=MeanSquaredError)
METRICS.register(name="mean_squared_log_error",                   module=MeanSquaredLogError)
METRICS.register(name="minkowski_distance",                       module=MinkowskiDistance)
METRICS.register(name="mse",                                      module=MeanSquaredError)
METRICS.register(name="pearson_corr_coef",                        module=PearsonCorrCoef)
METRICS.register(name="r2_score",                                 module=R2Score)
METRICS.register(name="relative_squared_error",                   module=RelativeSquaredError)
METRICS.register(name="spearman_corr_coef",                       module=SpearmanCorrCoef)
METRICS.register(name="symmetric_mean_absolute_percentage_error", module=SymmetricMeanAbsolutePercentageError)
METRICS.register(name="tweedie_deviance_score",                   module=TweedieDevianceScore)
METRICS.register(name="weighted_mean_absolute_percentage_error",  module=WeightedMeanAbsolutePercentageError)
