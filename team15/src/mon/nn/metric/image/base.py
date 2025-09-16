#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implement image quality assessment metrics."""

__all__ = [
    "ErrorRelativeGlobalDimensionlessSynthesis",
    "InceptionScore",
    "KernelInceptionDistance",
    "LearnedPerceptualImagePatchSimilarity",
    "MemorizationInformedFrechetInceptionDistance",
    "MultiScaleStructuralSimilarityIndexMeasure",
    "PeakSignalNoiseRatio",
    "PeakSignalNoiseRatioWithBlockedEffect",
    "PerceptualPathLength",
    "RelativeAverageSpectralError",
    "RootMeanSquaredErrorUsingSlidingWindow",
    "SpatialCorrelationCoefficient",
    "SpatialDistortionIndex",
    "SpectralAngleMapper",
    "SpectralDistortionIndex",
    "StructuralSimilarityIndexMeasure",
    "TotalVariation",
    "UniversalImageQualityIndex",
    "VisualInformationFidelity",
]

from torchmetrics.image import (
    ErrorRelativeGlobalDimensionlessSynthesis, InceptionScore, KernelInceptionDistance,
    LearnedPerceptualImagePatchSimilarity, MemorizationInformedFrechetInceptionDistance,
    MultiScaleStructuralSimilarityIndexMeasure, PeakSignalNoiseRatio, PeakSignalNoiseRatioWithBlockedEffect,
    PerceptualPathLength, RelativeAverageSpectralError, RootMeanSquaredErrorUsingSlidingWindow,
    SpatialCorrelationCoefficient, SpatialDistortionIndex, SpectralAngleMapper, SpectralDistortionIndex,
    StructuralSimilarityIndexMeasure, TotalVariation, UniversalImageQualityIndex, VisualInformationFidelity,
)

from mon.constants import METRICS

# ----- Registering -----
METRICS.register(name="error_relative_global_dimensionless_synthesis",    module=ErrorRelativeGlobalDimensionlessSynthesis)
METRICS.register(name="inception_score",                                  module=InceptionScore)
METRICS.register(name="kernel_inception_distance",                        module=KernelInceptionDistance)
METRICS.register(name="learned_perceptual_image_patch_similarity",        module=LearnedPerceptualImagePatchSimilarity)
METRICS.register(name="lpips",                                            module=LearnedPerceptualImagePatchSimilarity)
METRICS.register(name="memorization_informed_frechet_inception_distance", module=MemorizationInformedFrechetInceptionDistance)
METRICS.register(name="multiscale_ssim",                                  module=MultiScaleStructuralSimilarityIndexMeasure)
METRICS.register(name="multiscale_structural_similarity_index_measure",   module=MultiScaleStructuralSimilarityIndexMeasure)
METRICS.register(name="peak_signal_noise_ratio",                          module=PeakSignalNoiseRatio)
METRICS.register(name="peak_signal_noise_ratio_with_blocked_effect",      module=PeakSignalNoiseRatioWithBlockedEffect)
METRICS.register(name="perceptual_path_length",                           module=PerceptualPathLength)
METRICS.register(name="psnr",                                             module=PeakSignalNoiseRatio)
METRICS.register(name="relative_average_spectral_error",                  module=RelativeAverageSpectralError)
METRICS.register(name="root_mean_squared_error_using_sliding_window",     module=RootMeanSquaredErrorUsingSlidingWindow)
METRICS.register(name="spatial_correlation_coefficient",                  module=SpatialCorrelationCoefficient)
METRICS.register(name="spatial_distortion_index",                         module=SpatialDistortionIndex)
METRICS.register(name="spectral_angle_mapper",                            module=SpectralAngleMapper)
METRICS.register(name="spectral_distortion_index",                        module=SpectralDistortionIndex)
METRICS.register(name="ssim",                                             module=StructuralSimilarityIndexMeasure)
METRICS.register(name="structural_similarity_index_measure",              module=StructuralSimilarityIndexMeasure)
METRICS.register(name="total_variation",                                  module=TotalVariation)
METRICS.register(name="universal_image_quality_index",                    module=UniversalImageQualityIndex)
METRICS.register(name="visual_information_fidelity",                      module=VisualInformationFidelity)
