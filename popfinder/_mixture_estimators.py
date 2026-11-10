"""Reference-frequency estimators for sets of unknown SNP samples."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.model_selection import StratifiedKFold


def _dosage_matrix(data, name):
    if data is None or data.empty:
        raise ValueError(f"{name} must contain at least one sample")
    if "alleles" not in data.columns:
        raise ValueError(f"{name} must have an 'alleles' column")

    try:
        matrix = np.stack(data["alleles"].to_numpy()).astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Could not create a rectangular dosage matrix for {name}") from exc

    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one SNP")

    observed = matrix[np.isfinite(matrix)]
    if observed.size == 0:
        raise ValueError(f"{name} has no observed genotype dosages")
    if np.any((observed < 0) | (observed > 2)):
        raise ValueError("Allele dosages must be between 0 and 2")
    if not np.allclose(observed, np.round(observed)):
        raise ValueError("Allele dosages must be integer counts in {0, 1, 2}")

    return matrix


def _reference_frequencies(reference_data, populations, pseudocount):
    if not isinstance(pseudocount, (int, float)) or not np.isfinite(pseudocount):
        raise TypeError("pseudocount must be a finite number")
    if pseudocount < 0:
        raise ValueError("pseudocount must be greater than or equal to 0")
    if "pop" not in reference_data.columns:
        raise ValueError("reference_data must have a 'pop' column")
    if reference_data["pop"].isna().any():
        raise ValueError("reference_data cannot contain unknown population labels")

    reference = _dosage_matrix(reference_data, "reference_data")
    if populations is None:
        populations = pd.unique(reference_data["pop"]).tolist()
    else:
        populations = list(populations)

    if not populations:
        raise ValueError("At least one reference population is required")

    frequencies = []
    keep_loci = np.ones(reference.shape[1], dtype=bool)
    for population in populations:
        pop_mask = reference_data["pop"].to_numpy() == population
        if not np.any(pop_mask):
            raise ValueError(f"No reference samples found for population {population!r}")

        pop_genotypes = reference[pop_mask]
        called = np.isfinite(pop_genotypes)
        allele_count = 2 * called.sum(axis=0)
        keep_loci &= allele_count > 0
        alt_count = np.nansum(pop_genotypes, axis=0)
        frequencies.append(
            (alt_count + pseudocount) / (allele_count + 2 * pseudocount)
        )

    if not np.any(keep_loci):
        raise ValueError("No SNPs have observed genotypes in every reference population")

    frequencies = np.asarray(frequencies)[:, keep_loci]
    return frequencies, populations, keep_loci


def estimate_mixture_proportions_em(
    reference_data,
    unknown_data,
    populations=None,
    pseudocount=0.5,
    prior_strength=0.0,
    tolerance=1e-7,
    max_iter=500,
):
    """Estimate proportions of discrete source-population members in a set.

    Each unknown individual is modeled as originating from one source
    population. Genotypes are conditionally independent across SNPs under a
    binomial allele-frequency likelihood; LD pruning is recommended.
    """
    if not isinstance(prior_strength, (int, float)) or not np.isfinite(prior_strength):
        raise TypeError("prior_strength must be a finite number")
    if prior_strength < 0:
        raise ValueError("prior_strength must be greater than or equal to 0")
    if not isinstance(tolerance, (int, float)) or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be a positive finite number")
    if not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")

    frequencies, populations, keep_loci = _reference_frequencies(
        reference_data, populations, pseudocount
    )
    unknown_matrix = _dosage_matrix(unknown_data, "unknown_data")
    if unknown_matrix.shape[1] != len(keep_loci):
        raise ValueError(
            "reference_data and unknown_data must contain the same number of SNPs "
            "in the same order"
        )
    unknown = unknown_matrix[:, keep_loci]
    if np.any(np.isfinite(unknown).sum(axis=1) == 0):
        raise ValueError("Every unknown sample must have at least one called SNP")

    # Binomial(2, p) log likelihood, omitting the dosage-dependent coefficient
    # because it is identical across source populations for each sample.
    safe_frequency = np.clip(frequencies, 1e-12, 1 - 1e-12)
    log_p = np.log(safe_frequency)
    log_ref = np.log1p(-safe_frequency)
    log_likelihood = np.empty((len(unknown), len(populations)), dtype=float)
    for k in range(len(populations)):
        terms = unknown * log_p[k] + (2 - unknown) * log_ref[k]
        log_likelihood[:, k] = np.nansum(terms, axis=1)

    q = np.full(len(populations), 1 / len(populations), dtype=float)
    converged = False
    for iteration in range(1, max_iter + 1):
        log_joint = log_likelihood + np.log(np.maximum(q, 1e-300))[None, :]
        responsibilities = np.exp(log_joint - logsumexp(log_joint, axis=1, keepdims=True))
        counts = responsibilities.sum(axis=0)
        q_new = (counts + prior_strength / len(populations)) / (
            len(unknown) + prior_strength
        )
        if np.max(np.abs(q_new - q)) < tolerance:
            q = q_new
            converged = True
            break
        q = q_new

    result = pd.Series(q, index=populations, name="estimated_fraction")
    result.attrs["converged"] = converged
    result.attrs["iterations"] = iteration
    result.attrs["log_likelihood"] = float(
        np.sum(logsumexp(log_likelihood + np.log(np.maximum(q, 1e-300))[None, :], axis=1))
    )
    return result


def estimate_average_ancestry(
    reference_data,
    unknown_data,
    populations=None,
    pseudocount=0.5,
    ridge=1e-6,
    return_individual=False,
):
    """Estimate per-sample ancestry by constrained least squares, then average.

    Returns mean ancestry fractions across the unknown set. If
    ``return_individual=True``, returns ``(mean_ancestry, individual_estimates)``.
    """
    if not isinstance(ridge, (int, float)) or not np.isfinite(ridge):
        raise TypeError("ridge must be a finite number")
    if ridge < 0:
        raise ValueError("ridge must be greater than or equal to 0")

    frequencies, populations, keep_loci = _reference_frequencies(
        reference_data, populations, pseudocount
    )
    unknown_matrix = _dosage_matrix(unknown_data, "unknown_data")
    if unknown_matrix.shape[1] != len(keep_loci):
        raise ValueError(
            "reference_data and unknown_data must contain the same number of SNPs "
            "in the same order"
        )
    unknown = unknown_matrix[:, keep_loci] / 2
    prior = np.full(len(populations), 1 / len(populations), dtype=float)
    individual_estimates = np.empty((len(unknown), len(populations)), dtype=float)

    for i, sample in enumerate(unknown):
        called = np.isfinite(sample)
        if not np.any(called):
            raise ValueError("Every unknown sample must have at least one called SNP")

        population_freqs = frequencies[:, called]
        observed_freqs = sample[called]
        n_markers = len(observed_freqs)

        def objective(q):
            residual = q @ population_freqs - observed_freqs
            return (residual @ residual) / n_markers + ridge * np.sum((q - prior) ** 2)

        def gradient(q):
            residual = q @ population_freqs - observed_freqs
            return (2 / n_markers) * (population_freqs @ residual) + 2 * ridge * (q - prior)

        fit = minimize(
            objective,
            prior.copy(),
            jac=gradient,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * len(populations),
            constraints={
                "type": "eq",
                "fun": lambda q: np.sum(q) - 1,
                "jac": lambda q: np.ones_like(q),
            },
            options={"ftol": 1e-10, "maxiter": 300},
        )
        if not fit.success:
            raise RuntimeError(f"Ancestry optimization failed for sample {i}: {fit.message}")
        estimate = np.clip(fit.x, 0, 1)
        individual_estimates[i] = estimate / estimate.sum()

    individual_df = pd.DataFrame(
        individual_estimates,
        index=(unknown_data["sampleID"].to_numpy()
               if "sampleID" in unknown_data.columns else unknown_data.index),
        columns=populations,
    )
    individual_df.index.name = "sampleID"
    mean_ancestry = individual_df.mean(axis=0).rename("mean_ancestry")
    if return_individual:
        return mean_ancestry, individual_df
    return mean_ancestry


def cross_validate_mixture_estimators(
    labeled_data,
    n_splits=5,
    random_state=123,
    pseudocount=0.5,
    prior_strength=0.0,
    tolerance=1e-7,
    max_iter=500,
    ridge=1e-6,
):
    """Stratified k-fold validation of both reference-frequency estimators.

    Each fold estimates reference frequencies from the training samples, then
    predicts the composition and ancestry of the held-out, labeled samples.
    The ancestry scores therefore validate pure-population references against
    one-hot truth; they are not a validation of admixed individuals.

    EM is scored by population-fraction MAE/RMSE and total variation. Average
    ancestry is scored by individual ancestry MAE/RMSE and argmax accuracy, as
    well as error in the mean fractions for the held-out set. Fold metrics also
    include training-population-frequency baselines. For unbiased validation,
    supervised SNP filtering must be performed within each fold.

    Returns a dictionary containing per-fold metrics, their mean and standard
    deviation, held-out population fractions, and out-of-fold ancestry estimates.
    """
    if not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("n_splits must be an integer greater than or equal to 2")
    if labeled_data is None or labeled_data.empty:
        raise ValueError("labeled_data must contain at least one sample")
    if "pop" not in labeled_data.columns:
        raise ValueError("labeled_data must have a 'pop' column")
    if labeled_data["pop"].isna().any():
        raise ValueError("labeled_data cannot contain unknown population labels")

    # Validate the genotype vectors once and fail before starting any folds.
    _dosage_matrix(labeled_data, "labeled_data")
    labels = labeled_data["pop"].to_numpy()
    populations = pd.unique(labels).tolist()
    if len(populations) < 2:
        raise ValueError("At least two populations are required for validation")
    counts = pd.Series(labels).value_counts()
    smallest_population = int(counts.min())
    if smallest_population < n_splits:
        raise ValueError(
            f"Each population needs at least n_splits ({n_splits}) samples for "
            f"stratified k-fold validation; the smallest population has "
            f"{smallest_population}."
        )

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    fold_metrics = []
    fraction_rows = []
    ancestry_rows = []

    for fold, (train_indices, test_indices) in enumerate(
        splitter.split(np.zeros(len(labels)), labels), start=1
    ):
        reference = labeled_data.iloc[train_indices]
        held_out = labeled_data.iloc[test_indices]
        held_out_labels = held_out["pop"].to_numpy()
        true_fractions = (
            held_out["pop"].value_counts(normalize=True)
            .reindex(populations, fill_value=0)
            .to_numpy(dtype=float)
        )
        reference_prior = (
            reference["pop"].value_counts(normalize=True)
            .reindex(populations, fill_value=0)
            .to_numpy(dtype=float)
        )
        prior_error = reference_prior - true_fractions

        em_estimate = estimate_mixture_proportions_em(
            reference_data=reference,
            unknown_data=held_out,
            populations=populations,
            pseudocount=pseudocount,
            prior_strength=prior_strength,
            tolerance=tolerance,
            max_iter=max_iter,
        )
        em_fractions = em_estimate.reindex(populations).to_numpy(dtype=float)
        em_error = em_fractions - true_fractions

        ancestry_mean, ancestry_individual = estimate_average_ancestry(
            reference_data=reference,
            unknown_data=held_out,
            populations=populations,
            pseudocount=pseudocount,
            ridge=ridge,
            return_individual=True,
        )
        ancestry_fractions = ancestry_mean.reindex(populations).to_numpy(dtype=float)
        ancestry_values = ancestry_individual.loc[:, populations].to_numpy(dtype=float)
        true_individual = (
            held_out_labels[:, None] == np.asarray(populations, dtype=object)[None, :]
        ).astype(float)
        ancestry_error = ancestry_values - true_individual
        ancestry_mean_error = ancestry_fractions - true_fractions
        prior_individual = np.broadcast_to(reference_prior, true_individual.shape)
        ancestry_prior_error = prior_individual - true_individual

        # Fraction errors average across populations, while sample ancestry
        # errors average across held-out individuals and populations.
        fold_metrics.append({
            "fold": fold,
            "em_fraction_mae": float(np.mean(np.abs(em_error))),
            "em_fraction_rmse": float(np.sqrt(np.mean(em_error ** 2))),
            "em_total_variation": float(0.5 * np.sum(np.abs(em_error))),
            "reference_prior_fraction_mae": float(np.mean(np.abs(prior_error))),
            "reference_prior_fraction_rmse": float(
                np.sqrt(np.mean(prior_error ** 2))
            ),
            "em_converged": bool(em_estimate.attrs["converged"]),
            "em_iterations": int(em_estimate.attrs["iterations"]),
            "ancestry_sample_mae": float(np.mean(np.abs(ancestry_error))),
            "ancestry_sample_rmse": float(np.sqrt(np.mean(ancestry_error ** 2))),
            "ancestry_sample_argmax_accuracy": float(
                np.mean(np.argmax(ancestry_values, axis=1) == np.argmax(true_individual, axis=1))
            ),
            "ancestry_prior_sample_mae": float(
                np.mean(np.abs(ancestry_prior_error))
            ),
            "ancestry_prior_sample_rmse": float(
                np.sqrt(np.mean(ancestry_prior_error ** 2))
            ),
            "ancestry_mean_fraction_mae": float(np.mean(np.abs(ancestry_mean_error))),
            "ancestry_mean_fraction_rmse": float(
                np.sqrt(np.mean(ancestry_mean_error ** 2))
            ),
        })

        for index, population in enumerate(populations):
            fraction_rows.append({
                "fold": fold,
                "population": population,
                "true_fraction": true_fractions[index],
                "reference_prior_fraction": reference_prior[index],
                "em_estimated_fraction": em_fractions[index],
                "ancestry_mean_fraction": ancestry_fractions[index],
            })

        sample_ids = (
            held_out["sampleID"].to_numpy()
            if "sampleID" in held_out.columns
            else held_out.index.to_numpy()
        )
        ancestry_fold = pd.DataFrame({
            "fold": fold,
            "sampleID": sample_ids,
            "true_pop": held_out_labels,
        })
        for index, population in enumerate(populations):
            ancestry_fold[f"estimated_{population}"] = ancestry_values[:, index]
        ancestry_rows.append(ancestry_fold)

    fold_metrics = pd.DataFrame(fold_metrics)
    summary = fold_metrics.drop(columns="fold").agg(["mean", "std"]).T
    summary.index.name = "metric"
    summary = summary.reset_index()

    return {
        "fold_metrics": fold_metrics,
        "summary": summary,
        "fold_fractions": pd.DataFrame(fraction_rows),
        "individual_ancestry": pd.concat(ancestry_rows, ignore_index=True),
    }
