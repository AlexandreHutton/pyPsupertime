from .preprocessing import Preprocessing, transform_labels
from .model import SGDModel, ThresholdSGDModel, CumulativePenaltySGDModel, PsupertimeBaseModel
from .parameter_search import RegularizationSearchCV
from .plots import (plot_grid_search,
                    plot_identified_gene_coefficients, 
                    plot_labels_over_psupertime, 
                    plot_model_perf, 
                    plot_identified_genes_over_psupertime)

import datetime
import sys
import warnings
from typing import Iterable, Union

import numpy as np
from sklearn import metrics
from sklearn.base import TransformerMixin
import anndata as ad
from scanpy import read_h5ad


class Psupertime:
    """
    Main class for the pypsupertime analysis.
    
    This class handles the end-to-end process of preprocessing data, 
    running regularization search for model parameters, and fitting the 
    final psupertime model.

    Attributes
    ----------
    method : str
        Statistical model for ordinal regression.
    verbosity : int
        Level of output detail.
    random_state : int
        Seed for reproducibility.
    n_jobs : int
        Number of parallel jobs for grid search.
    n_folds : int
        Number of folds for cross-validation.
    n_batches : int
        Number of batches for training.
    preprocessing : Preprocessing
        Preprocessing transformer instance.
    estimator_params : dict
        Parameters for the estimator.
    model : PsupertimeBaseModel
        The fitted model instance.
    grid_search : RegularizationSearchCV
        Grid search instance for regularization.
    regularization_params : dict
        Parameters for regularization search.
    """

    def __init__(self,
                 method="proportional",
                 max_memory=None,
                 n_folds=5,
                 n_jobs=5,
                 n_batches=1,
                 random_state=1234,
                 verbosity=1,
                 regularization_params=dict(),
                 preprocessing_class=Preprocessing,
                 preprocessing_params=dict(),
                 estimator_class=CumulativePenaltySGDModel,
                 estimator_params=dict()):
        """
        Initializes the Psupertime object with provided parameters.

        Parameters
        ----------
        method : str, optional
            Statistical model used for ordinal logistic regression. 
            Options: "proportional", "forward", "backward". Defaults to "proportional".
        max_memory : int, optional
            Currently not implemented. Defaults to None.
        n_folds : int, optional
            Number of cross-validation folds. Defaults to 5.
        n_jobs : int, optional
            Number of parallel jobs. Defaults to 5.
        n_batches : int, optional
            Number of batches for SGD training. Defaults to 1.
        random_state : int, optional
            Seed for random number generator. Defaults to 1234.
        verbosity : int, optional
            Verbosity level. Defaults to 1.
        regularization_params : dict, optional
            Parameters for RegularizationSearchCV. Defaults to empty dict.
        preprocessing_class : class, optional
            Class used for preprocessing. Defaults to Preprocessing.
        preprocessing_params : dict, optional
            Parameters for preprocessing_class. Defaults to empty dict.
        estimator_class : class, optional
            Class used as estimator. Defaults to CumulativePenaltySGDModel.
        estimator_params : dict, optional
            Parameters for estimator_class. Defaults to empty dict.
        """

        self.verbosity = verbosity
        self.random_state = random_state

        # statistical method   
        self.method = method
    
        # grid search params
        self.n_jobs = n_jobs
        self.n_folds = n_folds
        
        # model params
        self.max_memory = max_memory
        if self.max_memory is not None:
            warnings.warn("Parameter `max_memory` is currently not implemented. Try setting n_batches directly to control the memory usage.")
        self.n_batches = n_batches
        
        # TODO: Implement preprocessing as Pipeline with GridSearch -> requires fit functions to take anndata.AnnData
        if not isinstance(preprocessing_params, dict):
            raise ValueError("Parameter estimator_params is not of type dict. Received: ", preprocessing_params)
        
        if preprocessing_class is None:
            self.preprocessing = None
        else:
            if not issubclass(preprocessing_class, TransformerMixin):
                raise ValueError("Parameter preprocessing_class must be None or of type sklearn.base.TransformerMixin. Received: %s" % preprocessing_class)
            self.preprocessing = preprocessing_class(**preprocessing_params)

        # Validate estimator params and instantiate model
        if not isinstance(estimator_params, dict):
            raise ValueError("Parameter estimator_params is not of type dict. Received: %s" % estimator_params)
        
        self.estimator_params = estimator_params
        self.estimator_params["n_batches"] = self.n_batches
        self.estimator_params["method"] = self.method
        self.estimator_params["random_state"] = self.random_state
        self.model = None  # not fitted yet

        if not issubclass(estimator_class, PsupertimeBaseModel):
            raise ValueError("Parameter estimator_class does not inherit PsupertimeBaseModel. Received: ", estimator_class)

        if not isinstance(regularization_params, dict):
            raise ValueError("Parameter estimator_params is not of type dict. Received: ", regularization_params)

        self.regularization_params = regularization_params 
        self.regularization_params["n_jobs"] = regularization_params.get("n_jobs", self.n_jobs)
        self.regularization_params["n_folds"] = regularization_params.get("n_folds", self.n_folds)
        self.regularization_params["estimator"] = estimator_class
        self.grid_search = None  # not fitted yet

    def check_is_fitted(self, raise_error=False):
        """
        Checks if the model has been fitted.

        Parameters
        ----------
        raise_error : bool, optional
            If True, raises a ValueError if not fitted. Defaults to False.

        Returns
        -------
        bool
            True if fitted, False otherwise.
        """
        is_fitted = isinstance(self.model, PsupertimeBaseModel) and self.model.is_fitted_
        
        if raise_error and not is_fitted:
            ValueError("Invalid estimator class or model not fitted yet. Did you call run() already?")
        else:
            return is_fitted
    
    def run(self, adata: Union[ad.AnnData, str], ordinal_data: Union[Iterable, str], copy=True):
        """
        Runs the full psupertime analysis pipeline: preprocessing, grid search, and model fitting.

        Parameters
        ----------
        adata : Union[ad.AnnData, str]
            AnnData object or path to .h5ad file.
        ordinal_data : Union[Iterable, str]
            Labels for the samples. Either an iterable of labels 
            or a string representing a column name in `adata.obs`.
        copy : bool, optional
            Whether to operate on a copy of adata. Currently only True is supported. Defaults to True.

        Returns
        -------
        ad.AnnData
            The annotated AnnData object containing psupertime predictions and gene weights.
        """
        
        if not copy:
            warnings.warn("Setting parameter copy=False is not supported, yet. Returning a copy of adata ...")

        start_time = datetime.datetime.now()

        # TODO: respect verbosity setting everywhere

        # Validate adata or load the filename
        if isinstance(adata, ad.AnnData):
            adata = adata.copy()

        elif isinstance(adata, str):
            filename = adata
            adata = read_h5ad(filename)

        else:
            raise ValueError("Parameter adata must be a filename or anndata.AnnData object. Received: ", adata)

        print("Input Data: n_genes=%s, n_cells=%s" % (adata.n_vars, adata.n_obs))

        # Validate the ordinal data
        if isinstance(ordinal_data, str):
            column_name = ordinal_data
            if column_name not in adata.obs.columns:
                raise ValueError("Parameter ordinal_data is not a valid column in adata.obs. Received: ", ordinal_data)

            ordinal_data = adata.obs.get(column_name)
        
        elif isinstance(ordinal_data, Iterable):
            if len(ordinal_data) != adata.n_obs:
                raise ValueError("Parameter ordinal_data has invalid length. Expected: %s Received: %s" % (len(ordinal_data), adata.n_obs))

        adata.obs["ordinal_label"] = transform_labels(ordinal_data)

        # Run Preprocessing
        if self.preprocessing is not None:
            print("Preprocessing", end="\r")
            adata = self.preprocessing.fit_transform(adata)
            print("Preprocessing: done. mode='%s', n_genes=%s, n_cells=%s" % (self.preprocessing.select_genes, adata.n_vars, adata.n_obs))

        # TODO: Test / Train split required? -> produce two index arrays, to avoid copying the data?

        # heuristic for setting reg_low based on the number of genes and cells in the data, if it has not been specified
        if not (self.regularization_params.get("n_params", False) 
                or self.regularization_params.get("reg_low", False)
                or self.regularization_params.get("reg_high", False)):
            self.regularization_params["reg_low"] = 0.1 if adata.n_obs > adata.n_vars else 0.0001
        self.grid_search = RegularizationSearchCV(**self.regularization_params)
        
        # Run Grid Search
        print("Grid Search CV: CPUs=%s, n_folds=%s" % (self.grid_search.n_jobs, self.grid_search.n_folds))
        self.grid_search.fit(adata.X, adata.obs.ordinal_label, estimator_params=self.estimator_params)

        # Refit Model on _all_ data
        print("Refit on all data", end="\r")
        self.model = self.grid_search.get_optimal_model("1se")
        self.model.track_scores = True
        self.model.fit(adata.X, adata.obs.ordinal_label)
        acc = metrics.accuracy_score(self.model.predict(adata.X), adata.obs.ordinal_label)
        dof = np.count_nonzero(self.model.coef_)
        print("Refit on all data: done. accuracy=%f.02, n_genes=%s" % (acc, dof))

        # Annotate the data
        self.model.predict_psuper(adata, inplace=True)
        self.model.gene_weights(adata, inplace=True)

        self.is_fitted_ = True
        print("Total elapsed time: ", str(datetime.datetime.now() - start_time))

        return adata
    
    def refit_and_predict(self, adata, *args, **kwargs):
        """
        Refits the model with a specific regularization parameter and updates predictions.

        Args:
            adata (ad.AnnData): AnnData object to predict on.
            *args: Arguments passed to `grid_search.get_optimal_model`.
            **kwargs: Keyword arguments passed to `grid_search.get_optimal_model`.
        """
        self.check_is_fitted(raise_error=True)
        print("Input Data: n_genes=%s, n_cells=%s" % (adata.n_vars, adata.n_obs))
        print("Refit on all data", end="\r")
        self.model = self.grid_search.get_optimal_model(*args, **kwargs)
        self.model.track_scores = True
        self.model.fit(adata.X, adata.obs.ordinal_label)
        acc = metrics.accuracy_score(self.model.predict(adata.X), adata.obs.ordinal_label)
        dof = np.count_nonzero(self.model.coef_)
        print("Refit on all data: done. accuracy=%f.02, n_genes=%s" % (acc, dof))

        self.model.predict_psuper(adata, inplace=True)

    def predict_psuper(self, *args, **kwargs):
        """
        Predicts psupertime for the given data using the fitted model.

        Parameters
        ----------
        *args
            Arguments passed to `self.model.predict_psuper`.
        **kwargs
            Keyword arguments passed to `self.model.predict_psuper`.

        Returns
        -------
        pd.DataFrame or None
            The output of `self.model.predict_psuper`.
        """
        self.check_is_fitted(raise_error=True)
        return self.model.predict_psuper(*args, **kwargs)

    def plot_grid_search(self, *args, **kwargs):
        """
        Plots the results of the regularization grid search.

        Parameters
        ----------
        *args
            Arguments passed to `plot_grid_search`.
        **kwargs
            Keyword arguments passed to `plot_grid_search`.

        Returns
        -------
        matplotlib.figure.Figure
            The output of `plot_grid_search`.
        """
        self.check_is_fitted(raise_error=True)
        return plot_grid_search(self.grid_search, *args, **kwargs)
        
    def plot_model_perf(self, *args, **kwargs):
        """
        Plots the performance of the fitted model.

        Parameters
        ----------
        *args
            Arguments passed to `plot_model_perf`.
        **kwargs
            Keyword arguments passed to `plot_model_perf`.

        Returns
        -------
        matplotlib.figure.Figure
            The output of `plot_model_perf`.
        """
        self.check_is_fitted(raise_error=True)
        return plot_model_perf(self.model, *args, **kwargs)

    def plot_identified_gene_coefficients(self, *args, **kwargs):
        """
        Plots the coefficients of the genes identified by the model.

        Parameters
        ----------
        *args
            Arguments passed to `plot_identified_gene_coefficients`.
        **kwargs
            Keyword arguments passed to `plot_identified_gene_coefficients`.

        Returns
        -------
        matplotlib.figure.Figure
            The output of `plot_identified_gene_coefficients`.
        """
        self.check_is_fitted(raise_error=True)
        return plot_identified_gene_coefficients(self.model, *args, **kwargs)
    
    def plot_identified_genes_over_psupertime(self, *args, **kwargs):
        """
        Plots the expression of identified genes over psupertime.
        (Currently raises NotImplementedError)

        Args:
            *args: Arguments passed to `plot_identified_genes_over_psupertime`.
            **kwargs: Keyword arguments passed to `plot_identified_genes_over_psupertime`.
        """
        raise NotImplementedError()
        self.check_is_fitted(raise_error=True)
        return plot_identified_genes_over_psupertime(*args, **kwargs)

    def plot_labels_over_psupertime(self, *args, **kwargs):
        """
        Plots the original labels over the predicted psupertime.

        Parameters
        ----------
        *args
            Arguments passed to `plot_labels_over_psupertime`.
        **kwargs
            Keyword arguments passed to `plot_labels_over_psupertime`.

        Returns
        -------
        matplotlib.figure.Figure
            The output of `plot_labels_over_psupertime`.
        """
        self.check_is_fitted(raise_error=True)
        return plot_labels_over_psupertime(self.model, *args, **kwargs)

    def plot_model_training(self):
        """
        Plots the training progress of the model.

        Raises
        ------
        ValueError
            If `track_scores` was not set to True in the model.
        NotImplementedError
            Currently not implemented.
        """
        self.check_is_fitted(raise_error=True)

        if not self.model.track_scores:
            raise ValueError("Cannot plot model training if 'track_scores' is set to false in self.model: %s" % self.model)

        raise NotImplemented()
