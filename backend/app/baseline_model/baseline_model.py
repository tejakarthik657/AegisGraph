import pandas as pd

from sklearn.ensemble import RandomForestClassifier


class BaselineFraudModel:
    """
    Random Forest baseline model for transaction fraud detection.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 12,
        min_samples_leaf: int = 2,
        random_state: int = 42,
    ):
        """
        Initialize the Random Forest model.

        Parameters
        ----------
        n_estimators:
            Number of decision trees.

        max_depth:
            Maximum depth of each tree.

        min_samples_leaf:
            Minimum number of samples required
            in a leaf.

        random_state:
            Ensures reproducible results.
        """

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        )

    # -----------------------------------------------------
    # Train
    # -----------------------------------------------------
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ):
        """
        Train the Random Forest model.
        """

        self.model.fit(
            X,
            y,
        )

        return self

    # -----------------------------------------------------
    # Predict class
    # -----------------------------------------------------
    def predict(
        self,
        X: pd.DataFrame,
    ):
        """
        Predict fraud class.

        Returns:
            0 = normal
            1 = fraud
        """

        return self.model.predict(X)

    # -----------------------------------------------------
    # Predict probability
    # -----------------------------------------------------
    def predict_proba(
        self,
        X: pd.DataFrame,
    ):
        """
        Predict probability of fraud.

        Returns:
            Probability for each class.
        """

        return self.model.predict_proba(X)

    # -----------------------------------------------------
    # Fraud probability
    # -----------------------------------------------------
    def fraud_probability(
        self,
        X: pd.DataFrame,
    ):
        """
        Return probability that each transaction
        is fraudulent.
        """

        probabilities = (
            self.model.predict_proba(X)
        )

        fraud_class_index = list(
            self.model.classes_
        ).index(1)

        return probabilities[
            :,
            fraud_class_index,
        ]

    # -----------------------------------------------------
    # Feature importance
    # -----------------------------------------------------
    def feature_importance(
        self,
        feature_names,
    ) -> pd.DataFrame:
        """
        Return feature importance from the
        trained Random Forest.
        """

        importance = (
            self.model.feature_importances_
        )

        result = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importance,
            }
        )

        return (
            result
            .sort_values(
                "importance",
                ascending=False,
            )
            .reset_index(drop=True)
        )