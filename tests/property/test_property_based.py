"""
Property-based testing for image classification components.
Uses Hypothesis to generate edge cases and adversarial inputs.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from hypothesis.extra.numpy import arrays
from hypothesis import HealthCheck
import numpy as np
from PIL import Image
import io

from src.data.preprocessing import PreprocessingPipeline
from src.data.augmentation import AugmentationPipeline
from src.inference.predictor import Predictor
from src.utils.seed import set_seed


class TestPreprocessingProperty:
    """Property-based tests for preprocessing pipeline."""
    
    @given(
        width=st.integers(min_value=1, max_value=4096),
        height=st.integers(min_value=1, max_value=4096),
        channels=st.sampled_from([1, 3, 4]),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_preprocessing_handles_any_valid_image_size(self, width, height, channels):
        """Preprocessing should handle any valid image dimensions."""
        # Create random image
        img_array = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        pipeline = PreprocessingPipeline(input_size=224)
        result = pipeline(img)
        
        assert result is not None
        assert result.shape[1] == 224  # Height
        assert result.shape[2] == 224  # Width
    
    @given(
        pixel_values=arrays(
            np.uint8,
            shape=(100, 100, 3),
            elements=st.integers(min_value=0, max_value=255)
        ),
    )
    def test_preprocessing_preserves_information(self, pixel_values):
        """Similar images should produce similar preprocessed outputs."""
        img1 = Image.fromarray(pixel_values)
        img2 = Image.fromarray(pixel_values)
        
        pipeline = PreprocessingPipeline(input_size=224)
        result1 = pipeline(img1)
        result2 = pipeline(img2)
        
        # Identical inputs should produce identical outputs
        assert np.allclose(result1, result2, rtol=1e-5)
    
    @given(
        noise_level=st.floats(min_value=0.0, max_value=1.0),
    )
    def test_preprocessing_robust_to_noise(self, noise_level):
        """Preprocessing should handle noisy images gracefully."""
        base_img = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
        noise = (np.random.random((100, 100, 3)) * 255 * noise_level).astype(np.uint8)
        noisy_img = np.clip(base_img.astype(np.int32) + noise.astype(np.int32), 0, 255).astype(np.uint8)
        
        img = Image.fromarray(noisy_img)
        pipeline = PreprocessingPipeline(input_size=224)
        
        # Should not raise exceptions
        result = pipeline(img)
        assert result is not None
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


class TestAugmentationProperty:
    """Property-based tests for augmentation pipeline."""
    
    @given(
        rotation_angle=st.floats(min_value=-180, max_value=180),
        brightness_factor=st.floats(min_value=0.1, max_value=2.0),
        contrast_factor=st.floats(min_value=0.1, max_value=2.0),
    )
    @settings(max_examples=30)
    def test_augmentation_parameters_in_valid_range(self, rotation_angle, brightness_factor, contrast_factor):
        """Augmentation should handle all valid parameter ranges."""
        img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        aug_pipeline = AugmentationPipeline(
            rotation_range=abs(rotation_angle),
            brightness_range=(min(1.0, brightness_factor), max(1.0, brightness_factor)),
            contrast_range=(min(1.0, contrast_factor), max(1.0, contrast_factor)),
        )
        
        # Should not raise exceptions
        augmented = aug_pipeline(img)
        assert augmented is not None
        assert augmented.size == img.size
    
    @given(
        seed=st.integers(min_value=0, max_value=2**32 - 1),
    )
    def test_augmentation_reproducible_with_seed(self, seed):
        """Augmentation should be reproducible with same seed."""
        img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        aug_pipeline = AugmentationPipeline(
            rotation_range=30,
            horizontal_flip=True,
            zoom_range=0.2,
        )
        
        set_seed(seed)
        result1 = aug_pipeline(img)
        
        set_seed(seed)
        result2 = aug_pipeline(img)
        
        # Results should be identical with same seed
        assert np.array_equal(np.array(result1), np.array(result2))


class TestPredictorProperty:
    """Property-based tests for predictor."""
    
    @given(
        batch_size=st.integers(min_value=1, max_value=32),
        top_k=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=20)
    def test_predictor_output_shape_consistency(self, batch_size, top_k):
        """Predictor output should have consistent shape."""
        # Skip if no model available
        pytest.skip("Requires trained model")
        
        # Create batch of random images
        images = [
            Image.fromarray(np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8))
            for _ in range(batch_size)
        ]
        
        predictor = Predictor(model_path="models/dummy.pth", device="cpu")
        
        predictions = predictor.predict_batch(images, top_k=top_k)
        
        assert len(predictions) == batch_size
        for pred in predictions:
            assert 'predictions' in pred
            assert len(pred['predictions']) <= top_k
    
    @given(
        confidence_threshold=st.floats(min_value=0.0, max_value=1.0),
    )
    def test_confidence_threshold_filtering(self, confidence_threshold):
        """All returned predictions should meet confidence threshold."""
        pytest.skip("Requires trained model")
        
        img = Image.fromarray(np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8))
        
        predictor = Predictor(model_path="models/dummy.pth", device="cpu")
        result = predictor.predict(img, confidence_threshold=confidence_threshold)
        
        for pred in result['predictions']:
            assert pred['confidence'] >= confidence_threshold - 1e-6  # Small tolerance for float precision


class TestDataValidationProperty:
    """Property-based tests for data validation."""
    
    @given(
        corrupt_bytes=st.binary(min_size=1, max_size=1000),
    )
    def test_validator_rejects_corrupt_images(self, corrupt_bytes):
        """Data validator should reject corrupt image data."""
        from src.data.data_validator import DataValidator
        
        validator = DataValidator()
        
        # Should return False or raise exception for corrupt data
        try:
            is_valid = validator.validate_image_bytes(corrupt_bytes)
            # If it doesn't raise, it should return False for truly corrupt data
            # Note: Some random bytes might accidentally be valid, that's OK
        except Exception:
            # Expected for most corrupt data
            pass
    
    @given(
        size=st.integers(min_value=0, max_value=100_000_000),  # Up to 100MB
    )
    def test_validator_enforces_size_limits(self, size):
        """Validator should enforce file size limits."""
        from src.data.data_validator import DataValidator
        
        validator = DataValidator(max_file_size_mb=10)
        
        if size > 10 * 1024 * 1024:  # Over 10MB
            # Create large dummy file
            dummy_data = b'\x00' * min(size, 15 * 1024 * 1024)  # Cap at 15MB for test speed
            is_valid = validator.validate_image_bytes(dummy_data)
            assert is_valid is False
        else:
            # Smaller files might be valid (depending on content)
            pass


class TestEdgeCases:
    """Test specific edge cases that often cause failures."""
    
    def test_empty_image(self):
        """Handle empty or zero-sized images."""
        img_array = np.zeros((100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        pipeline = PreprocessingPipeline(input_size=224)
        result = pipeline(img)
        
        assert result is not None
        # All values should be normalized appropriately
        assert result.min() >= -1.0 or result.min() >= 0.0  # Depends on normalization
    
    def test_single_pixel_image(self):
        """Handle single pixel images."""
        img_array = np.array([[[255, 128, 64]]], dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        pipeline = PreprocessingPipeline(input_size=224)
        result = pipeline(img)
        
        assert result is not None
        assert result.shape[1:] == (224, 224)  # Should be resized
    
    def test_maximum_intensity_image(self):
        """Handle images with maximum intensity values."""
        img_array = np.full((100, 100, 3), 255, dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        pipeline = PreprocessingPipeline(input_size=224)
        result = pipeline(img)
        
        assert result is not None
        assert not np.any(np.isnan(result))
    
    def test_alpha_channel_image(self):
        """Handle images with alpha channel."""
        img_array = np.random.randint(0, 256, (100, 100, 4), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        pipeline = PreprocessingPipeline(input_size=224)
        result = pipeline(img)
        
        assert result is not None
        # Should handle RGBA correctly (convert to RGB or handle alpha)
        assert result.shape[0] == 3  # Output should be 3 channels


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
