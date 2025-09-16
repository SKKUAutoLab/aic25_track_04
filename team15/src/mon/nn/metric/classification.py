#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implement classification metrics."""

__all__ = [
	"Accuracy",
	"AveragePrecision",
	"CalibrationError",
	"CohenKappa",
	"ConfusionMatrix",
	"ExactMatch",
	"F1Score",
	"FBetaScore",
	"HammingDistance",
	"HingeLoss",
	"JaccardIndex",
	"MatthewsCorrCoef",
	"Precision",
	"PrecisionAtFixedRecall",
	"PrecisionRecallCurve",
	"ROC",
	"Recall",
	"RecallAtFixedPrecision",
	"Specificity",
	"SpecificityAtSensitivity",
	"StatScores",
  	"AUROC",
]

from torchmetrics.classification import (
	Accuracy, AUROC, AveragePrecision, CalibrationError, CohenKappa, ConfusionMatrix,
	ExactMatch, F1Score, FBetaScore, HammingDistance, HingeLoss, JaccardIndex,
	MatthewsCorrCoef, Precision, PrecisionAtFixedRecall, PrecisionRecallCurve, Recall,
	RecallAtFixedPrecision, ROC, Specificity, SpecificityAtSensitivity, StatScores,
)

from mon.constants import METRICS

# ----- Registering -----
METRICS.register(name="auroc",                      module=AUROC)
METRICS.register(name="accuracy",                   module=Accuracy)
METRICS.register(name="average_precision",          module=AveragePrecision)
METRICS.register(name="calibration_error",          module=CalibrationError)
METRICS.register(name="cohen_kappa",                module=CohenKappa)
METRICS.register(name="confusion_matrix",           module=ConfusionMatrix)
METRICS.register(name="exact_match",                module=ExactMatch)
METRICS.register(name="f1_score ",                  module=F1Score)
METRICS.register(name="f_beta_score",               module=FBetaScore)
METRICS.register(name="hamming_distance",           module=HammingDistance)
METRICS.register(name="hinge_loss",                 module=HingeLoss)
METRICS.register(name="jaccard_index",              module=JaccardIndex)
METRICS.register(name="matthews_corr_coef",         module=MatthewsCorrCoef)
METRICS.register(name="precision",                  module=Precision)
METRICS.register(name="precision_at_fixed_recall",  module=PrecisionAtFixedRecall)
METRICS.register(name="precision_recall_curve",     module=PrecisionRecallCurve)
METRICS.register(name="roc",                        module=ROC)
METRICS.register(name="recall",                     module=Recall)
METRICS.register(name="recall_at_fixed_precision",  module=RecallAtFixedPrecision)
METRICS.register(name="specificity",                module=Specificity)
METRICS.register(name="specificity_at_sensitivity", module=SpecificityAtSensitivity)
METRICS.register(name="stat_scores",                module=StatScores)
