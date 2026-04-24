"""
Data quality validation utilities.
"""

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from PIL import Image
import numpy as np
from tqdm import tqdm

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataValidator:
    """
    Validate image data quality and integrity.
    """

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        self.min_image_size = 32  # Minimum dimension in pixels
        self.max_aspect_ratio = 10.0  # Maximum width/height ratio

    def validate_directory_structure(self) -> Dict[str, any]:
        """
        Validate that the directory structure follows expected format.

        Expected structure:
            root_dir/
                class1/
                    image1.jpg
                    image2.jpg
                class2/
                    image3.jpg

        Returns:
            Validation report dictionary
        """
        report = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "num_classes": 0,
            "classes": [],
        }

        if not self.root_dir.exists():
            report["valid"] = False
            report["errors"].append(f"Directory does not exist: {self.root_dir}")
            return report

        class_dirs = [d for d in self.root_dir.iterdir() if d.is_dir()]

        if not class_dirs:
            report["valid"] = False
            report["errors"].append("No class directories found")
            return report

        report["num_classes"] = len(class_dirs)
        report["classes"] = [d.name for d in class_dirs]

        # Check for empty classes
        for class_dir in class_dirs:
            images = list(class_dir.glob("*"))
            if not images:
                report["warnings"].append(f"Empty class directory: {class_dir.name}")

        return report

    def validate_images(
        self,
        check_duplicates: bool = True,
        verbose: bool = True,
    ) -> Dict[str, any]:
        """
        Validate individual images in the dataset.

        Args:
            check_duplicates: Whether to check for duplicate images
            verbose: Show progress bar

        Returns:
            Validation report dictionary
        """
        report = {
            "total_images": 0,
            "valid_images": 0,
            "invalid_images": 0,
            "corrupted": [],
            "too_small": [],
            "wrong_format": [],
            "duplicates": [],
            "hashes": {},
        }

        image_hashes: Dict[str, List[Path]] = {}

        # Get all image files
        image_files = []
        for ext in self.valid_extensions:
            image_files.extend(self.root_dir.rglob(f"*{ext}"))
            image_files.extend(self.root_dir.rglob(f"*{ext.upper()}"))

        report["total_images"] = len(image_files)

        iterator = tqdm(image_files, desc="Validating images") if verbose else image_files

        for img_path in iterator:
            try:
                # Try to open image
                with Image.open(img_path) as img:
                    # Verify it's actually an image
                    img.verify()

                # Re-open for additional checks (verify() can corrupt the image object)
                with Image.open(img_path) as img:
                    # Check dimensions
                    width, height = img.size
                    min_dim = min(width, height)

                    if min_dim < self.min_image_size:
                        report["too_small"].append(str(img_path))
                        report["invalid_images"] += 1
                        continue

                    # Check aspect ratio
                    aspect_ratio = max(width, height) / min(width, height)
                    if aspect_ratio > self.max_aspect_ratio:
                        report["warnings"] = report.get("warnings", [])
                        report["warnings"].append(
                            f"Extreme aspect ratio {aspect_ratio:.2f}: {img_path}"
                        )

                    # Check for duplicates using perceptual hash
                    if check_duplicates:
                        img_hash = self._compute_hash(img_path)
                        if img_hash in image_hashes:
                            report["duplicates"].append({
                                "file": str(img_path),
                                "duplicate_of": str(image_hashes[img_hash][0]),
                            })
                        else:
                            image_hashes[img_hash] = []
                        image_hashes[img_hash].append(img_path)

                report["valid_images"] += 1

            except Exception as e:
                report["corrupted"].append({
                    "file": str(img_path),
                    "error": str(e),
                })
                report["invalid_images"] += 1

        report["hashes"] = {k: [str(p) for p in v] for k, v in image_hashes.items()}

        return report

    def _compute_hash(self, image_path: Path) -> str:
        """
        Compute a perceptual hash of an image.

        Args:
            image_path: Path to the image file

        Returns:
            Hash string
        """
        try:
            with Image.open(image_path) as img:
                # Resize to small fixed size
                img = img.convert("L").resize((8, 8), Image.Resampling.LANCZOS)
                pixels = list(img.getdata())

                # Compute average
                avg = sum(pixels) / len(pixels)

                # Create hash based on whether each pixel is above/below average
                bits = "".join("1" if pixel > avg else "0" for pixel in pixels)

                # Convert to hex
                return hex(int(bits, 2))[2:].zfill(16)

        except Exception:
            return hashlib.md5(open(image_path, "rb").read()).hexdigest()[:16]

    def validate_labels(
        self,
        expected_classes: Optional[Set[str]] = None,
    ) -> Dict[str, any]:
        """
        Validate class labels and naming consistency.

        Args:
            expected_classes: Set of expected class names (optional)

        Returns:
            Validation report dictionary
        """
        report = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "found_classes": [],
            "missing_classes": [],
            "unexpected_classes": [],
        }

        class_dirs = [d.name for d in self.root_dir.iterdir() if d.is_dir()]
        report["found_classes"] = class_dirs

        # Check for problematic naming patterns
        for class_name in class_dirs:
            if class_name.startswith("_"):
                report["warnings"].append(
                    f"Class name starts with underscore: {class_name}"
                )

            if " " in class_name:
                report["warnings"].append(
                    f"Class name contains spaces: {class_name}"
                )

            if class_name.lower() != class_name:
                report["warnings"].append(
                    f"Class name is not lowercase: {class_name}"
                )

        # Check against expected classes
        if expected_classes:
            report["missing_classes"] = list(expected_classes - set(class_dirs))
            report["unexpected_classes"] = list(set(class_dirs) - expected_classes)

            if report["missing_classes"]:
                report["valid"] = False
                report["errors"].append(
                    f"Missing expected classes: {report['missing_classes']}"
                )

        return report

    def generate_report(self, output_path: str) -> None:
        """
        Generate a comprehensive validation report.

        Args:
            output_path: Path to save the report
        """
        import json

        full_report = {
            "directory_structure": self.validate_directory_structure(),
            "image_validation": self.validate_images(),
            "label_validation": self.validate_labels(),
        }

        # Calculate overall validity
        full_report["overall_valid"] = (
            full_report["directory_structure"]["valid"]
            and full_report["label_validation"]["valid"]
            and full_report["image_validation"]["invalid_images"] == 0
        )

        with open(output_path, "w") as f:
            json.dump(full_report, f, indent=2)

        logger.info(f"Validation report saved to {output_path}")


def validate_dataset(data_dir: str, output_report: Optional[str] = None) -> Dict:
    """
    Convenience function to validate a dataset.

    Args:
        data_dir: Path to the dataset directory
        output_report: Optional path to save the report

    Returns:
        Validation report dictionary
    """
    validator = DataValidator(data_dir)

    if output_report:
        validator.generate_report(output_report)

    return {
        "structure": validator.validate_directory_structure(),
        "images": validator.validate_images(verbose=False),
        "labels": validator.validate_labels(),
    }
