"""
Model Caching System for ML Models
Caches trained models to improve performance and reduce training time.
"""

import os
import pickle
import hashlib
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd

logger = logging.getLogger(__name__)

# Check for TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    HAS_TF = True
except ImportError:
    HAS_TF = False

# Cache directory
CACHE_DIR = "data/models/cache"
CACHE_EXPIRY_DAYS = 7  # Models expire after 7 days


class ModelCache:
    """Model caching wrapper with get/set interface."""

    def __init__(self):
        self.cache_dir = CACHE_DIR
        self.expiry_days = CACHE_EXPIRY_DAYS

    def get(self, model_type: str, cache_key: str, stock_data: pd.DataFrame = None) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached model if available.

        Args:
            model_type: Type of model ('random_forest', 'svm', etc.)
            cache_key: Cache key (usually symbol_hash)
            stock_data: Stock data (unused, for compatibility)

        Returns:
            Cached model data or None
        """
        return get_cached_model(model_type, cache_key)

    def set(self, model_type: str, cache_key: str, model_data: Dict[str, Any],
            stock_data: pd.DataFrame = None, metadata: Dict = None) -> bool:
        """
        Cache a trained model.

        Args:
            model_type: Type of model
            cache_key: Cache key
            model_data: Model data to cache
            stock_data: Stock data (unused, for compatibility)
            metadata: Additional metadata (unused, for compatibility)

        Returns:
            True if caching succeeded
        """
        return cache_model(model_type, cache_key, model_data)


def get_model_cache() -> ModelCache:
    """
    Get model cache instance.

    Returns:
        ModelCache instance
    """
    return ModelCache()


def _ensure_cache_dir():
    """Ensure cache directory exists."""
    os.makedirs(CACHE_DIR, exist_ok=True)


def _get_cache_path(model_type: str, symbol_hash: str) -> str:
    """Get file path for cached model."""
    return os.path.join(CACHE_DIR, f"{model_type}_{symbol_hash}.pkl")


def _is_cache_valid(cache_path: str) -> bool:
    """Check if cache file is still valid (not expired)."""
    if not os.path.exists(cache_path):
        return False

    # Check file age
    file_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
    age = datetime.now() - file_time

    return age < timedelta(days=CACHE_EXPIRY_DAYS)


def get_cached_model(model_type: str, symbol_hash: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve cached model if available and valid.

    Args:
        model_type: Type of model ('random_forest', 'svm', etc.)
        symbol_hash: Hash of symbol/data to identify cache

    Returns:
        Cached model data or None if not available
    """
    try:
        _ensure_cache_dir()
        cache_path = _get_cache_path(model_type, symbol_hash)

        if not _is_cache_valid(cache_path):
            return None

        with open(cache_path, 'rb') as f:
            data = pickle.load(f)

        # Handle Keras models stored separately
        if 'model_path' in data and HAS_TF:
            keras_path = data['model_path']
            if os.path.exists(keras_path):
                try:
                    # Generic load with safe_mode=False (Keras 3+)
                    try:
                        model = keras.models.load_model(keras_path, safe_mode=False)
                    except TypeError:
                         # Fallback for Keras versions without safe_mode arg
                        model = keras.models.load_model(keras_path)
                    
                    data['model'] = model
                except Exception as e:
                    logger.warning(
                        f"Failed to load Keras model from {keras_path}: {e}. Removing cache.")
                    try:
                        os.remove(cache_path)
                        if os.path.exists(keras_path):
                            os.remove(keras_path)
                    except:
                        pass
                    return None
            else:
                logger.warning(
                    f"Keras model file missing at {keras_path}. Removing cache.")
                try:
                    os.remove(cache_path)
                except:
                    pass
                return None

        # Validate model structure
        if 'model' not in data and 'model_path' not in data: # model_path check for safety
             if 'timestamp' not in data: # Basic check
                logger.warning(
                    f"Invalid cache structure for {model_type}_{symbol_hash}")
                os.remove(cache_path)
                return None

        # Additional validation for ensemble models (estimators_ check)
        model = data.get('model')
        if model is not None and not (HAS_TF and isinstance(model, (keras.Model, tf.keras.Model))):
            if hasattr(model, '__class__'):
                model_class = model.__class__.__name__
                if 'Random' in model_class or 'Forest' in model_class:
                    if not hasattr(model, 'estimators_') or model.estimators_ is None or len(model.estimators_) == 0:
                        logger.warning(
                            f"Cached {model_type} model is invalid (missing estimators_). Removing cache.")
                        os.remove(cache_path)
                        return None

        logger.info(f"Retrieved cached {model_type} model from {cache_path}")
        return data

    except (EOFError, pickle.UnpicklingError, AttributeError) as e:
        logger.warning(
            f"Failed to load cached model {model_type}_{symbol_hash}: {e}. Removing corrupt cache.")
        try:
            os.remove(cache_path)
        except:
            pass
        return None
    except Exception as e:
        logger.error(f"Error retrieving cached model: {e}")
        return None


def cache_model(model_type: str, symbol_hash: str, model_data: Dict[str, Any]) -> bool:
    """
    Cache a trained model for future use.

    Args:
        model_type: Type of model ('random_forest', 'svm', etc.)
        symbol_hash: Hash of symbol/data to identify cache
        model_data: Model data to cache (must include 'model' key)

    Returns:
        True if caching succeeded, False otherwise
    """
    try:
        _ensure_cache_dir()

        if 'model' not in model_data:
            logger.warning(f"Cannot cache model without 'model' key")
            return False

        model = model_data.get('model')
        cache_path = _get_cache_path(model_type, symbol_hash)
        
        # Prepare data to cache
        cache_data = {
            **model_data,
            'timestamp': datetime.now().isoformat(),
            'model_type': model_type,
            'symbol_hash': symbol_hash
        }

        # Check if it's a Keras model
        is_keras = False
        if HAS_TF and model is not None:
             if isinstance(model, (keras.Model, tf.keras.Model)):
                 is_keras = True
        
        if is_keras:
            # Save Keras model to separate file
            keras_path = cache_path.replace('.pkl', '.keras')
            try:
                # Save in Keras format (v3 compatible)
                model.save(keras_path)
                
                # Update cache data to reference the file instead of the object
                cache_data['model_path'] = keras_path
                cache_data['model'] = None # Remove actual model object
                
            except Exception as e:
                logger.error(f"Failed to save Keras model for {model_type}: {e}")
                return False
        else:
             # Validate regular models (e.g. sklearn)
            if hasattr(model, '__class__'):
                model_class = model.__class__.__name__
                if 'Random' in model_class or 'Forest' in model_class:
                    if not hasattr(model, 'estimators_') or model.estimators_ is None or len(model.estimators_) == 0:
                        logger.warning(
                            f"Cannot cache invalid {model_type} model (missing estimators_)")
                        return False

        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)

        logger.info(f"Cached {model_type} model to {cache_path}")
        return True

    except Exception as e:
        logger.error(f"Error caching model: {e}")
        return False


def clear_expired_cache():
    """Remove expired cache files."""
    try:
        _ensure_cache_dir()

        removed_count = 0
        for filename in os.listdir(CACHE_DIR):
            if filename.endswith('.pkl'):
                cache_path = os.path.join(CACHE_DIR, filename)
                if not _is_cache_valid(cache_path):
                    os.remove(cache_path)
                    
                    # Also remove associated .keras file if it exists
                    keras_filename = filename.replace('.pkl', '.keras')
                    keras_path = os.path.join(CACHE_DIR, keras_filename)
                    if os.path.exists(keras_path):
                        os.remove(keras_path)
                        
                    removed_count += 1
            
            # Clean up orphaned .keras files (expiry check is same as pkl)
            # This is simplified; we rely on pkl file presence mostly
            # Ideally we check modification time of .keras too if pkl is missing

        if removed_count > 0:
            logger.info(f"Removed {removed_count} expired cache files")

    except Exception as e:
        logger.error(f"Error clearing expired cache: {e}")


def clear_all_cache():
    """Remove all cached models."""
    try:
        _ensure_cache_dir()

        removed_count = 0
        for filename in os.listdir(CACHE_DIR):
            file_path = os.path.join(CACHE_DIR, filename)
            if filename.endswith('.pkl') or filename.endswith('.keras') or filename.endswith('.h5'):
                os.remove(file_path)
                removed_count += 1

        logger.info(f"Cleared {removed_count} cached model files")

    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
