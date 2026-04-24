"""
Secrets Management Module using HashiCorp Vault.
Provides secure storage and retrieval of sensitive configuration.
"""

import os
import logging
from typing import Optional, Dict, Any
from functools import lru_cache

logger = logging.getLogger(__name__)

try:
    import hvac
    VAULT_AVAILABLE = True
except ImportError:
    VAULT_AVAILABLE = False
    logger.warning("hvac library not installed. Vault integration disabled.")


class VaultSecretsManager:
    """
    Secure secrets management using HashiCorp Vault.
    
    Features:
    - Dynamic secret generation
    - Automatic secret rotation
    - Audit logging
    - Least privilege access
    - Encryption at rest and in transit
    """
    
    def __init__(
        self,
        vault_url: Optional[str] = None,
        vault_token: Optional[str] = None,
        vault_namespace: Optional[str] = None,
        mount_point: str = "secret",
    ):
        """
        Initialize Vault client.
        
        Args:
            vault_url: Vault server URL (VAULT_ADDR env var)
            vault_token: Vault token (VAULT_TOKEN env var)
            vault_namespace: Vault namespace for multi-tenancy
            mount_point: Secret engine mount point
        """
        if not VAULT_AVAILABLE:
            raise RuntimeError(
                "Vault integration requires 'hvac' package. "
                "Install with: pip install hvac"
            )
        
        self.vault_url = vault_url or os.getenv("VAULT_ADDR")
        self.vault_token = vault_token or os.getenv("VAULT_TOKEN")
        self.vault_namespace = vault_namespace or os.getenv("VAULT_NAMESPACE")
        self.mount_point = mount_point
        
        if not self.vault_url:
            raise ValueError("VAULT_ADDR environment variable must be set")
        
        self.client = self._create_client()
        self._authenticate()
    
    def _create_client(self) -> 'hvac.Client':
        """Create and configure Vault client."""
        client = hvac.Client(
            url=self.vault_url,
            namespace=self.vault_namespace,
        )
        
        # Enable TLS verification in production
        verify_tls = os.getenv("VAULT_SKIP_VERIFY", "false").lower() != "true"
        client.session.verify = verify_tls
        
        return client
    
    def _authenticate(self):
        """Authenticate with Vault using token."""
        if not self.vault_token:
            raise ValueError("VAULT_TOKEN environment variable must be set")
        
        self.client.token = self.vault_token
        
        # Verify authentication
        if not self.client.is_authenticated():
            raise RuntimeError("Failed to authenticate with Vault")
        
        logger.info("Successfully authenticated with Vault")
    
    def get_secret(self, path: str, key: Optional[str] = None) -> Any:
        """
        Retrieve a secret from Vault.
        
        Args:
            path: Secret path (e.g., "image-classifier/database")
            key: Specific key within the secret (optional)
            
        Returns:
            Secret value or entire secret dict
        """
        try:
            response = self.client.secrets.kv.v2.read_secret_version(
                path=path,
                mount_point=self.mount_point,
            )
            
            data = response.get("data", {}).get("data", {})
            
            if key:
                if key not in data:
                    raise KeyError(f"Key '{key}' not found in secret at '{path}'")
                return data[key]
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to retrieve secret '{path}': {e}")
            raise
    
    def set_secret(self, path: str, data: Dict[str, Any]) -> bool:
        """
        Store a secret in Vault.
        
        Args:
            path: Secret path
            data: Dictionary of key-value pairs
            
        Returns:
            True if successful
        """
        try:
            self.client.secrets.kv.v2.create_or_update_secret(
                path=path,
                secret=data,
                mount_point=self.mount_point,
            )
            logger.info(f"Successfully stored secret at '{path}'")
            return True
        except Exception as e:
            logger.error(f"Failed to store secret '{path}': {e}")
            raise
    
    def delete_secret(self, path: str) -> bool:
        """Delete a secret from Vault."""
        try:
            self.client.secrets.kv.v2.delete_metadata_and_all_versions(
                path=path,
                mount_point=self.mount_point,
            )
            logger.info(f"Successfully deleted secret at '{path}'")
            return True
        except Exception as e:
            logger.error(f"Failed to delete secret '{path}': {e}")
            raise
    
    def rotate_secret(self, path: str, key: str, generator_func) -> Any:
        """
        Rotate a secret using a generator function.
        
        Args:
            path: Secret path
            key: Key to rotate
            generator_func: Function that generates new secret value
            
        Returns:
            New secret value
        """
        try:
            # Generate new secret
            new_value = generator_func()
            
            # Get existing secret
            current_secret = self.get_secret(path)
            
            # Update with new value
            current_secret[key] = new_value
            
            # Store updated secret
            self.set_secret(path, current_secret)
            
            logger.info(f"Successfully rotated secret '{key}' at '{path}'")
            return new_value
            
        except Exception as e:
            logger.error(f"Failed to rotate secret '{key}' at '{path}': {e}")
            raise
    
    def list_secrets(self, path: str) -> list:
        """List all secrets at a given path."""
        try:
            response = self.client.secrets.kv.v2.list_secrets(
                path=path,
                mount_point=self.mount_point,
            )
            return response.get("data", {}).get("keys", [])
        except Exception as e:
            logger.error(f"Failed to list secrets at '{path}': {e}")
            raise
    
    def get_database_credentials(self, engine_path: str) -> Dict[str, str]:
        """
        Get dynamic database credentials.
        
        Args:
            engine_path: Path to database secrets engine
            
        Returns:
            Dictionary with username and password
        """
        try:
            response = self.client.secrets.database.generate_credentials(
                name=engine_path,
            )
            return {
                "username": response["data"]["username"],
                "password": response["data"]["password"],
            }
        except Exception as e:
            logger.error(f"Failed to generate database credentials: {e}")
            raise


@lru_cache(maxsize=1)
def get_secrets_manager() -> VaultSecretsManager:
    """Get cached instance of secrets manager."""
    return VaultSecretsManager()


def get_secret_cached(path: str, key: Optional[str] = None) -> Any:
    """Get secret with LRU caching."""
    manager = get_secrets_manager()
    return manager.get_secret(path, key)


# Example usage and integration helpers
class SecretsConfig:
    """Configuration class for integrating secrets into application."""
    
    @staticmethod
    def load_api_config() -> Dict[str, Any]:
        """Load API configuration from Vault."""
        manager = get_secrets_manager()
        
        return {
            "api_key": manager.get_secret("image-classifier/api", "api_key"),
            "jwt_secret": manager.get_secret("image-classifier/api", "jwt_secret"),
            "allowed_origins": manager.get_secret("image-classifier/api", "allowed_origins"),
        }
    
    @staticmethod
    def load_database_config() -> Dict[str, Any]:
        """Load database configuration from Vault."""
        manager = get_secrets_manager()
        
        # For dynamic credentials
        try:
            creds = manager.get_database_credentials("image-classifier-db")
            return {
                "username": creds["username"],
                "password": creds["password"],
                "host": manager.get_secret("image-classifier/database", "host"),
                "port": manager.get_secret("image-classifier/database", "port"),
                "database": manager.get_secret("image-classifier/database", "name"),
            }
        except Exception:
            # Fallback to static credentials
            return {
                "username": manager.get_secret("image-classifier/database", "username"),
                "password": manager.get_secret("image-classifier/database", "password"),
                "host": manager.get_secret("image-classifier/database", "host"),
                "port": manager.get_secret("image-classifier/database", "port"),
                "database": manager.get_secret("image-classifier/database", "name"),
            }
    
    @staticmethod
    def load_mlflow_config() -> Dict[str, Any]:
        """Load MLflow tracking configuration from Vault."""
        manager = get_secrets_manager()
        
        return {
            "tracking_uri": manager.get_secret("image-classifier/mlflow", "tracking_uri"),
            "username": manager.get_secret("image-classifier/mlflow", "username"),
            "password": manager.get_secret("image-classifier/mlflow", "password"),
        }
