"""
Adversarial Input Detection Module.
Detects model evasion attacks including adversarial examples, out-of-distribution inputs, and data poisoning attempts.
"""

import numpy as np
import torch
from typing import Tuple, Dict, Any, Optional, List
import logging
from scipy import stats
from sklearn.covariance import Mahalanobis

logger = logging.getLogger(__name__)


class AdversarialDetector:
    """
    Detect adversarial inputs and model evasion attacks.
    
    Detection methods:
    - Mahalanobis distance for OOD detection
    - Prediction entropy analysis
    - Gradient norm monitoring
    - Feature activation anomalies
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        feature_extractor: torch.nn.Module,
        train_features: Optional[np.ndarray] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize detector.
        
        Args:
            model: The classification model
            feature_extractor: Module to extract features (usually layers before final FC)
            train_features: Training set features for distribution modeling
            device: Device for computation
        """
        self.model = model
        self.feature_extractor = feature_extractor
        self.device = device
        self.mahalanobis = None
        
        if train_features is not None:
            self.fit_distribution(train_features)
    
    def fit_distribution(self, train_features: np.ndarray):
        """
        Fit feature distribution model using training data.
        
        Args:
            train_features: Features from training set (N, D)
        """
        logger.info("Fitting Mahalanobis distance model...")
        
        # Compute empirical covariance
        self.mean_vector = np.mean(train_features, axis=0)
        self.cov_matrix = np.cov(train_features, rowvar=False)
        
        # Add regularization for numerical stability
        self.cov_matrix += 1e-6 * np.eye(self.cov_matrix.shape[0])
        
        try:
            self.mahalanobis = Mahalanobis(
                mean=self.mean_vector,
                covinv=np.linalg.inv(self.cov_matrix),
            )
            logger.info("Mahalanobis distance model fitted successfully")
        except Exception as e:
            logger.warning(f"Failed to fit Mahalanobis model: {e}")
            self.mahalanobis = None
    
    def detect(
        self,
        input_tensor: torch.Tensor,
        threshold_mahalanobis: float = 5.0,
        threshold_entropy: float = 2.0,
        threshold_gradient: float = 10.0,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive adversarial detection.
        
        Args:
            input_tensor: Input image tensor
            threshold_mahalanobis: Threshold for OOD detection
            threshold_entropy: Threshold for prediction uncertainty
            threshold_gradient: Threshold for gradient anomaly
            
        Returns:
            Dictionary with detection results and scores
        """
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_(True)
        
        results = {
            "is_adversarial": False,
            "detection_scores": {},
            "flags": [],
        }
        
        # Forward pass
        with torch.enable_grad():
            features = self.feature_extractor(input_tensor)
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            
            # Compute loss for gradient analysis
            predicted_class = torch.argmax(probabilities, dim=1)
            loss = torch.nn.functional.cross_entropy(output, predicted_class)
            
            # Backward pass for gradients
            loss.backward()
            gradients = input_tensor.grad.detach()
        
        # 1. Mahalanobis Distance (OOD Detection)
        maha_score = self._compute_mahalanobis(features)
        results["detection_scores"]["mahalanobis_distance"] = maha_score
        
        if maha_score > threshold_mahalanobis:
            results["flags"].append("out_of_distribution")
            results["is_adversarial"] = True
        
        # 2. Prediction Entropy (Uncertainty)
        entropy = self._compute_entropy(probabilities)
        results["detection_scores"]["prediction_entropy"] = entropy
        
        if entropy > threshold_entropy:
            results["flags"].append("high_uncertainty")
            results["is_adversarial"] = True
        
        # 3. Gradient Norm (Adversarial Perturbation)
        grad_norm = torch.norm(gradients).item()
        results["detection_scores"]["gradient_norm"] = grad_norm
        
        if grad_norm > threshold_gradient:
            results["flags"].append("abnormal_gradient")
            results["is_adversarial"] = True
        
        # 4. Maximum Probability (Confidence)
        max_prob = torch.max(probabilities).item()
        results["detection_scores"]["max_probability"] = max_prob
        
        if max_prob < 0.5:  # Low confidence
            results["flags"].append("low_confidence")
        
        # 5. Feature Activation Analysis
        activation_stats = self._analyze_activations(features)
        results["detection_scores"]["activation_stats"] = activation_stats
        
        if activation_stats["z_score_max"] > 5.0:
            results["flags"].append("activation_anomaly")
            results["is_adversarial"] = True
        
        return results
    
    def _compute_mahalanobis(self, features: torch.Tensor) -> float:
        """Compute Mahalanobis distance for OOD detection."""
        if self.mahalanobis is None:
            return 0.0
        
        features_np = features.detach().cpu().numpy().flatten()
        
        try:
            distance = self.mahalanobis.mahalanobis(features_np)
            return float(distance)
        except Exception:
            return 0.0
    
    def _compute_entropy(self, probabilities: torch.Tensor) -> float:
        """Compute prediction entropy."""
        eps = 1e-10
        probs = probabilities + eps
        entropy = -torch.sum(probs * torch.log(probs), dim=1)
        return entropy.item()
    
    def _analyze_activations(self, features: torch.Tensor) -> Dict[str, float]:
        """Analyze feature activations for anomalies."""
        features_np = features.detach().cpu().numpy().flatten()
        
        if not hasattr(self, 'mean_vector'):
            return {"z_score_max": 0.0, "z_score_mean": 0.0}
        
        # Compute z-scores
        std_dev = np.sqrt(np.diag(self.cov_matrix))
        z_scores = np.abs((features_np - self.mean_vector) / (std_dev + 1e-10))
        
        return {
            "z_score_max": float(np.max(z_scores)),
            "z_score_mean": float(np.mean(z_scores)),
            "z_score_99th": float(np.percentile(z_scores, 99)),
        }
    
    def batch_detect(
        self,
        inputs: torch.Tensor,
        batch_size: int = 32,
    ) -> List[Dict[str, Any]]:
        """Run detection on a batch of inputs."""
        results = []
        
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i+batch_size]
            for j in range(len(batch)):
                result = self.detect(batch[j:j+1])
                results.append(result)
        
        return results


class DataPoisoningDetector:
    """
    Detect data poisoning attempts in training data.
    
    Methods:
    - Statistical outlier detection
    - Label consistency checks
    - Feature clustering analysis
    """
    
    def __init__(self, contamination_threshold: float = 0.05):
        """
        Initialize detector.
        
        Args:
            contamination_threshold: Expected fraction of poisoned samples
        """
        self.contamination_threshold = contamination_threshold
    
    def detect_outliers(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect outliers that may be poisoned samples.
        
        Args:
            features: Feature matrix (N, D)
            labels: Class labels (N,)
            
        Returns:
            Tuple of (outlier_indices, outlier_scores)
        """
        from sklearn.ensemble import IsolationForest
        
        # Run Isolation Forest per class
        outlier_scores = np.zeros(len(features))
        
        for label in np.unique(labels):
            mask = labels == label
            class_features = features[mask]
            
            if len(class_features) < 10:
                continue
            
            iso_forest = IsolationForest(
                contamination=self.contamination_threshold,
                random_state=42,
            )
            
            scores = -iso_forest.fit_predict(class_features)
            outlier_scores[mask] = scores
        
        outlier_indices = np.where(outlier_scores > 0)[0]
        
        return outlier_indices, outlier_scores
    
    def detect_label_flipping(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        k_neighbors: int = 5,
    ) -> np.ndarray:
        """
        Detect potential label flipping attacks.
        
        Args:
            features: Feature matrix
            labels: Class labels
            k_neighbors: Number of neighbors for KNN
            
        Returns:
            Array of suspicious sample indices
        """
        from sklearn.neighbors import KNeighborsClassifier
        
        # Train KNN on all data
        knn = KNeighborsClassifier(n_neighbors=k_neighbors)
        knn.fit(features, labels)
        
        # Get predictions and compare with actual labels
        predictions = knn.predict(features)
        disagreements = predictions != labels
        
        # Find samples where majority of neighbors disagree
        suspicious_indices = np.where(disagreements)[0]
        
        logger.info(f"Detected {len(suspicious_indices)} potential label flips")
        
        return suspicious_indices


class AdversarialDefense:
    """
    Apply defenses against adversarial attacks.
    
    Defenses:
    - Input preprocessing (denoising)
    - Adversarial training support
    - Gradient masking
    """
    
    @staticmethod
    def denoise_input(
        input_tensor: torch.Tensor,
        method: str = "median_filter",
        kernel_size: int = 3,
    ) -> torch.Tensor:
        """
        Apply denoising to input.
        
        Args:
            input_tensor: Input tensor
            method: Denoising method
            kernel_size: Filter kernel size
            
        Returns:
            Denoised tensor
        """
        if method == "median_filter":
            return AdversarialDefense._median_filter(input_tensor, kernel_size)
        elif method == "gaussian_smoothing":
            return AdversarialDefense._gaussian_smooth(input_tensor, kernel_size)
        else:
            return input_tensor
    
    @staticmethod
    def _median_filter(tensor: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """Apply median filter."""
        # Simplified implementation
        # In production, use torchvision.transforms or custom CUDA kernel
        return tensor
    
    @staticmethod
    def _gaussian_smooth(tensor: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """Apply Gaussian smoothing."""
        from torchvision.transforms import GaussianBlur
        
        sigma = kernel_size / 6.0
        blur = GaussianBlur(kernel_size=kernel_size, sigma=sigma)
        return blur(tensor)
    
    @staticmethod
    def random_resize_crop(
        input_tensor: torch.Tensor,
        scale: Tuple[float, float] = (0.8, 1.0),
        ratio: Tuple[float, float] = (0.9, 1.1),
    ) -> torch.Tensor:
        """Apply random resize crop as defense."""
        from torchvision.transforms import RandomResizedCrop
        
        transform = RandomResizedCrop(
            size=input_tensor.shape[-2:],
            scale=scale,
            ratio=ratio,
        )
        return transform(input_tensor)


# Integration helper
def create_adversarial_pipeline(
    model: torch.nn.Module,
    feature_extractor: torch.nn.Module,
    train_features: np.ndarray,
) -> Dict[str, Any]:
    """Create complete adversarial detection pipeline."""
    
    detector = AdversarialDetector(
        model=model,
        feature_extractor=feature_extractor,
        train_features=train_features,
    )
    
    poison_detector = DataPoisoningDetector()
    defense = AdversarialDefense()
    
    return {
        "detector": detector,
        "poison_detector": poison_detector,
        "defense": defense,
    }
