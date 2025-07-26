"""
Nova Bloom Consciousness Architecture - Key Management System

This module implements a comprehensive key management system for the memory encryption layer,
providing secure key generation, rotation, derivation, and storage with HSM integration.

Key Features:
- Multiple key derivation functions (PBKDF2, Argon2id, HKDF, Scrypt)
- Hardware Security Module (HSM) integration
- Key rotation and lifecycle management
- Key escrow and recovery mechanisms
- Zero-knowledge architecture
- High-availability key services
"""

import asyncio
import json
import logging
import os
import secrets
import sqlite3
import struct
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import argon2
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.constant_time import bytes_eq


class KeyDerivationFunction(Enum):
    """Supported key derivation functions."""
    PBKDF2_SHA256 = "pbkdf2_sha256"
    PBKDF2_SHA512 = "pbkdf2_sha512"
    ARGON2ID = "argon2id"
    HKDF_SHA256 = "hkdf_sha256"
    HKDF_SHA512 = "hkdf_sha512"
    SCRYPT = "scrypt"


class KeyStatus(Enum):
    """Key lifecycle status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"
    REVOKED = "revoked"
    ESCROW = "escrow"


class HSMBackend(Enum):
    """Supported HSM backends."""
    SOFTWARE = "software"  # Software-based secure storage
    PKCS11 = "pkcs11"      # PKCS#11 compatible HSMs
    AWS_KMS = "aws_kms"    # AWS Key Management Service
    AZURE_KV = "azure_kv"  # Azure Key Vault
    GCP_KMS = "gcp_kms"    # Google Cloud KMS


@dataclass
class KeyMetadata:
    """Metadata for encryption keys."""
    key_id: str
    algorithm: str
    key_size: int
    created_at: datetime
    expires_at: Optional[datetime]
    status: KeyStatus
    version: int
    usage_count: int
    max_usage: Optional[int]
    tags: Dict[str, str]
    derivation_info: Optional[Dict[str, Any]] = None
    hsm_key_ref: Optional[str] = None


class KeyManagementException(Exception):
    """Base exception for key management operations."""
    pass


class HSMInterface(ABC):
    """Abstract interface for Hardware Security Module implementations."""
    
    @abstractmethod
    async def generate_key(self, algorithm: str, key_size: int) -> str:
        """Generate a key in the HSM and return a reference."""
        pass
    
    @abstractmethod
    async def get_key(self, key_ref: str) -> bytes:
        """Retrieve a key from the HSM."""
        pass
    
    @abstractmethod
    async def delete_key(self, key_ref: str) -> bool:
        """Delete a key from the HSM."""
        pass
    
    @abstractmethod
    async def encrypt_with_key(self, key_ref: str, plaintext: bytes) -> bytes:
        """Encrypt data using HSM key."""
        pass
    
    @abstractmethod
    async def decrypt_with_key(self, key_ref: str, ciphertext: bytes) -> bytes:
        """Decrypt data using HSM key."""
        pass


class SoftwareHSM(HSMInterface):
    """Software-based HSM implementation for development and testing."""
    
    def __init__(self, storage_path: str = "/tmp/nova_software_hsm"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.key_storage = self.storage_path / "keys.db"
        self._init_database()
        self._master_key = self._load_or_create_master_key()
        self.lock = threading.RLock()
    
    def _init_database(self):
        """Initialize the key storage database."""
        with sqlite3.connect(self.key_storage) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS hsm_keys (
                    key_ref TEXT PRIMARY KEY,
                    algorithm TEXT NOT NULL,
                    key_size INTEGER NOT NULL,
                    encrypted_key BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    
    def _load_or_create_master_key(self) -> bytes:
        """Load or create the master encryption key for the software HSM."""
        master_key_path = self.storage_path / "master.key"
        
        if master_key_path.exists():
            with open(master_key_path, 'rb') as f:
                return f.read()
        else:
            # Generate new master key
            master_key = secrets.token_bytes(32)  # 256-bit master key
            
            # Store securely (in production, this would be encrypted with user credentials)
            with open(master_key_path, 'wb') as f:
                f.write(master_key)
            
            # Set restrictive permissions
            os.chmod(master_key_path, 0o600)
            return master_key
    
    def _encrypt_key(self, key_data: bytes) -> bytes:
        """Encrypt a key with the master key."""
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        
        nonce = secrets.token_bytes(12)
        aesgcm = AESGCM(self._master_key)
        ciphertext = aesgcm.encrypt(nonce, key_data, None)
        return nonce + ciphertext
    
    def _decrypt_key(self, encrypted_data: bytes) -> bytes:
        """Decrypt a key with the master key."""
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        
        nonce = encrypted_data[:12]
        ciphertext = encrypted_data[12:]
        aesgcm = AESGCM(self._master_key)
        return aesgcm.decrypt(nonce, ciphertext, None)
    
    async def generate_key(self, algorithm: str, key_size: int) -> str:
        """Generate a key and store it securely."""
        key_ref = f"swhs_{secrets.token_hex(16)}"
        key_data = secrets.token_bytes(key_size // 8)  # Convert bits to bytes
        
        encrypted_key = self._encrypt_key(key_data)
        
        with self.lock:
            with sqlite3.connect(self.key_storage) as conn:
                conn.execute("""
                    INSERT INTO hsm_keys (key_ref, algorithm, key_size, encrypted_key)
                    VALUES (?, ?, ?, ?)
                """, (key_ref, algorithm, key_size, encrypted_key))
                conn.commit()
        
        return key_ref
    
    async def get_key(self, key_ref: str) -> bytes:
        """Retrieve and decrypt a key."""
        with self.lock:
            with sqlite3.connect(self.key_storage) as conn:
                cursor = conn.execute(
                    "SELECT encrypted_key FROM hsm_keys WHERE key_ref = ?",
                    (key_ref,)
                )
                row = cursor.fetchone()
                
                if not row:
                    raise KeyManagementException(f"Key not found: {key_ref}")
                
                encrypted_key = row[0]
                return self._decrypt_key(encrypted_key)
    
    async def delete_key(self, key_ref: str) -> bool:
        """Delete a key from storage."""
        with self.lock:
            with sqlite3.connect(self.key_storage) as conn:
                cursor = conn.execute(
                    "DELETE FROM hsm_keys WHERE key_ref = ?",
                    (key_ref,)
                )
                conn.commit()
                return cursor.rowcount > 0
    
    async def encrypt_with_key(self, key_ref: str, plaintext: bytes) -> bytes:
        """Encrypt data using a stored key."""
        key_data = await self.get_key(key_ref)
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        
        nonce = secrets.token_bytes(12)
        aesgcm = AESGCM(key_data)
        ciphertext = aesgcm.encrypt(nonce, plaintext, None)
        return nonce + ciphertext
    
    async def decrypt_with_key(self, key_ref: str, ciphertext: bytes) -> bytes:
        """Decrypt data using a stored key."""
        key_data = await self.get_key(key_ref)
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        
        nonce = ciphertext[:12]
        encrypted_data = ciphertext[12:]
        aesgcm = AESGCM(key_data)
        return aesgcm.decrypt(nonce, encrypted_data, None)


class KeyDerivationService:
    """Service for deriving encryption keys using various KDFs."""
    
    @staticmethod
    def derive_key(
        password: bytes,
        salt: bytes,
        key_length: int,
        kdf_type: KeyDerivationFunction,
        iterations: Optional[int] = None,
        memory_cost: Optional[int] = None,
        parallelism: Optional[int] = None
    ) -> Tuple[bytes, Dict[str, Any]]:
        """
        Derive a key using the specified KDF.
        
        Returns:
            Tuple of (derived_key, derivation_info)
        """
        derivation_info = {
            'kdf_type': kdf_type.value,
            'salt': salt.hex(),
            'key_length': key_length
        }
        
        if kdf_type == KeyDerivationFunction.PBKDF2_SHA256:
            iterations = iterations or 100000
            derivation_info['iterations'] = iterations
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=key_length,
                salt=salt,
                iterations=iterations,
                backend=default_backend()
            )
            derived_key = kdf.derive(password)
            
        elif kdf_type == KeyDerivationFunction.PBKDF2_SHA512:
            iterations = iterations or 100000
            derivation_info['iterations'] = iterations
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA512(),
                length=key_length,
                salt=salt,
                iterations=iterations,
                backend=default_backend()
            )
            derived_key = kdf.derive(password)
            
        elif kdf_type == KeyDerivationFunction.ARGON2ID:
            memory_cost = memory_cost or 65536  # 64 MB
            parallelism = parallelism or 1
            iterations = iterations or 3
            
            derivation_info.update({
                'memory_cost': memory_cost,
                'parallelism': parallelism,
                'iterations': iterations
            })
            
            derived_key = argon2.low_level.hash_secret_raw(
                secret=password,
                salt=salt,
                time_cost=iterations,
                memory_cost=memory_cost,
                parallelism=parallelism,
                hash_len=key_length,
                type=argon2.Type.ID
            )
            
        elif kdf_type == KeyDerivationFunction.HKDF_SHA256:
            hkdf = HKDF(
                algorithm=hashes.SHA256(),
                length=key_length,
                salt=salt,
                info=b'Nova Memory Encryption',
                backend=default_backend()
            )
            derived_key = hkdf.derive(password)
            
        elif kdf_type == KeyDerivationFunction.HKDF_SHA512:
            hkdf = HKDF(
                algorithm=hashes.SHA512(),
                length=key_length,
                salt=salt,
                info=b'Nova Memory Encryption',
                backend=default_backend()
            )
            derived_key = hkdf.derive(password)
            
        elif kdf_type == KeyDerivationFunction.SCRYPT:
            memory_cost = memory_cost or 8
            parallelism = parallelism or 1
            iterations = iterations or 16384
            
            derivation_info.update({
                'n': iterations,
                'r': memory_cost,
                'p': parallelism
            })
            
            kdf = Scrypt(
                algorithm=hashes.SHA256(),
                length=key_length,
                salt=salt,
                n=iterations,
                r=memory_cost,
                p=parallelism,
                backend=default_backend()
            )
            derived_key = kdf.derive(password)
            
        else:
            raise KeyManagementException(f"Unsupported KDF: {kdf_type}")
        
        return derived_key, derivation_info


class KeyRotationPolicy:
    """Policy for automatic key rotation."""
    
    def __init__(
        self,
        max_age_hours: int = 168,  # 7 days
        max_usage_count: Optional[int] = None,
        rotation_schedule: Optional[str] = None
    ):
        self.max_age_hours = max_age_hours
        self.max_usage_count = max_usage_count
        self.rotation_schedule = rotation_schedule
    
    def should_rotate(self, metadata: KeyMetadata) -> bool:
        """Determine if a key should be rotated based on policy."""
        now = datetime.utcnow()
        
        # Check age
        if (now - metadata.created_at).total_seconds() > self.max_age_hours * 3600:
            return True
        
        # Check usage count
        if self.max_usage_count and metadata.usage_count >= self.max_usage_count:
            return True
        
        # Check expiration
        if metadata.expires_at and now >= metadata.expires_at:
            return True
        
        return False


class KeyManagementSystem:
    """
    Comprehensive key management system for Nova memory encryption.
    
    Provides secure key generation, storage, rotation, and lifecycle management
    with HSM integration and key escrow capabilities.
    """
    
    def __init__(
        self,
        hsm_backend: HSMBackend = HSMBackend.SOFTWARE,
        storage_path: str = "/nfs/novas/system/memory/keys",
        rotation_policy: Optional[KeyRotationPolicy] = None
    ):
        """Initialize the key management system."""
        self.hsm_backend = hsm_backend
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.metadata_db = self.storage_path / "key_metadata.db"
        self.rotation_policy = rotation_policy or KeyRotationPolicy()
        
        self._init_database()
        self._init_hsm()
        
        self.kdf_service = KeyDerivationService()
        self.lock = threading.RLock()
        
        # Start background rotation task
        self._rotation_task = None
        self._start_rotation_task()
    
    def _init_database(self):
        """Initialize the key metadata database."""
        with sqlite3.connect(self.metadata_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS key_metadata (
                    key_id TEXT PRIMARY KEY,
                    algorithm TEXT NOT NULL,
                    key_size INTEGER NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    expires_at TIMESTAMP,
                    status TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    usage_count INTEGER DEFAULT 0,
                    max_usage INTEGER,
                    tags TEXT,
                    derivation_info TEXT,
                    hsm_key_ref TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS key_escrow (
                    key_id TEXT PRIMARY KEY,
                    encrypted_key BLOB NOT NULL,
                    escrow_public_key BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (key_id) REFERENCES key_metadata (key_id)
                )
            """)
            
            conn.commit()
    
    def _init_hsm(self):
        """Initialize the HSM backend."""
        if self.hsm_backend == HSMBackend.SOFTWARE:
            self.hsm = SoftwareHSM(str(self.storage_path / "hsm"))
        else:
            raise KeyManagementException(f"HSM backend not implemented: {self.hsm_backend}")
    
    def _start_rotation_task(self):
        """Start the background key rotation task."""
        async def rotation_worker():
            while True:
                try:
                    await self._perform_scheduled_rotation()
                    await asyncio.sleep(3600)  # Check every hour
                except Exception as e:
                    logging.error(f"Key rotation error: {e}")
        
        if asyncio.get_event_loop().is_running():
            self._rotation_task = asyncio.create_task(rotation_worker())
    
    def _serialize_metadata(self, metadata: KeyMetadata) -> Dict[str, Any]:
        """Serialize metadata for database storage."""
        data = asdict(metadata)
        data['created_at'] = metadata.created_at.isoformat()
        data['expires_at'] = metadata.expires_at.isoformat() if metadata.expires_at else None
        data['status'] = metadata.status.value
        data['tags'] = json.dumps(metadata.tags)
        data['derivation_info'] = json.dumps(metadata.derivation_info) if metadata.derivation_info else None
        return data
    
    def _deserialize_metadata(self, data: Dict[str, Any]) -> KeyMetadata:
        """Deserialize metadata from database."""
        return KeyMetadata(
            key_id=data['key_id'],
            algorithm=data['algorithm'],
            key_size=data['key_size'],
            created_at=datetime.fromisoformat(data['created_at']),
            expires_at=datetime.fromisoformat(data['expires_at']) if data['expires_at'] else None,
            status=KeyStatus(data['status']),
            version=data['version'],
            usage_count=data['usage_count'],
            max_usage=data['max_usage'],
            tags=json.loads(data['tags']) if data['tags'] else {},
            derivation_info=json.loads(data['derivation_info']) if data['derivation_info'] else None,
            hsm_key_ref=data['hsm_key_ref']
        )
    
    async def generate_key(
        self,
        algorithm: str = "AES-256",
        key_size: int = 256,
        key_id: Optional[str] = None,
        expires_at: Optional[datetime] = None,
        max_usage: Optional[int] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Generate a new encryption key.
        
        Args:
            algorithm: Encryption algorithm
            key_size: Key size in bits
            key_id: Optional key identifier (auto-generated if not provided)
            expires_at: Optional expiration time
            max_usage: Optional maximum usage count
            tags: Optional metadata tags
            
        Returns:
            Key identifier
        """
        key_id = key_id or f"nova_key_{secrets.token_hex(16)}"
        
        # Generate key in HSM
        hsm_key_ref = await self.hsm.generate_key(algorithm, key_size)
        
        # Create metadata
        metadata = KeyMetadata(
            key_id=key_id,
            algorithm=algorithm,
            key_size=key_size,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            status=KeyStatus.ACTIVE,
            version=1,
            usage_count=0,
            max_usage=max_usage,
            tags=tags or {},
            hsm_key_ref=hsm_key_ref
        )
        
        # Store metadata
        with self.lock:
            with sqlite3.connect(self.metadata_db) as conn:
                serialized = self._serialize_metadata(metadata)
                conn.execute("""
                    INSERT INTO key_metadata 
                    (key_id, algorithm, key_size, created_at, expires_at, status, 
                     version, usage_count, max_usage, tags, derivation_info, hsm_key_ref)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    serialized['key_id'], serialized['algorithm'], serialized['key_size'],
                    serialized['created_at'], serialized['expires_at'], serialized['status'],
                    serialized['version'], serialized['usage_count'], serialized['max_usage'],
                    serialized['tags'], serialized['derivation_info'], serialized['hsm_key_ref']
                ))
                conn.commit()
        
        return key_id
    
    async def derive_key(
        self,
        password: str,
        salt: Optional[bytes] = None,
        key_id: Optional[str] = None,
        kdf_type: KeyDerivationFunction = KeyDerivationFunction.ARGON2ID,
        key_size: int = 256,
        **kdf_params
    ) -> str:
        """
        Derive a key from a password using the specified KDF.
        
        Args:
            password: Password to derive from
            salt: Salt for derivation (auto-generated if not provided)
            key_id: Optional key identifier
            kdf_type: Key derivation function to use
            key_size: Derived key size in bits
            **kdf_params: Additional KDF parameters
            
        Returns:
            Key identifier
        """
        key_id = key_id or f"nova_derived_{secrets.token_hex(16)}"
        salt = salt or secrets.token_bytes(32)
        
        # Derive the key
        derived_key, derivation_info = self.kdf_service.derive_key(
            password.encode('utf-8'),
            salt,
            key_size // 8,  # Convert bits to bytes
            kdf_type,
            **kdf_params
        )
        
        # Store in HSM (for software HSM, we'll store the derived key directly)
        if self.hsm_backend == HSMBackend.SOFTWARE:
            # Create a pseudo HSM reference for derived keys
            hsm_key_ref = f"derived_{secrets.token_hex(16)}"
            # Store the derived key in the software HSM
            with self.hsm.lock:
                encrypted_key = self.hsm._encrypt_key(derived_key)
                with sqlite3.connect(self.hsm.key_storage) as conn:
                    conn.execute("""
                        INSERT INTO hsm_keys (key_ref, algorithm, key_size, encrypted_key)
                        VALUES (?, ?, ?, ?)
                    """, (hsm_key_ref, "DERIVED", key_size, encrypted_key))
                    conn.commit()
        else:
            raise KeyManagementException(f"Key derivation not supported for HSM: {self.hsm_backend}")
        
        # Create metadata
        metadata = KeyMetadata(
            key_id=key_id,
            algorithm="DERIVED",
            key_size=key_size,
            created_at=datetime.utcnow(),
            expires_at=None,
            status=KeyStatus.ACTIVE,
            version=1,
            usage_count=0,
            max_usage=None,
            tags={},
            derivation_info=derivation_info,
            hsm_key_ref=hsm_key_ref
        )
        
        # Store metadata
        with self.lock:
            with sqlite3.connect(self.metadata_db) as conn:
                serialized = self._serialize_metadata(metadata)
                conn.execute("""
                    INSERT INTO key_metadata 
                    (key_id, algorithm, key_size, created_at, expires_at, status, 
                     version, usage_count, max_usage, tags, derivation_info, hsm_key_ref)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    serialized['key_id'], serialized['algorithm'], serialized['key_size'],
                    serialized['created_at'], serialized['expires_at'], serialized['status'],
                    serialized['version'], serialized['usage_count'], serialized['max_usage'],
                    serialized['tags'], serialized['derivation_info'], serialized['hsm_key_ref']
                ))
                conn.commit()
        
        return key_id
    
    async def get_key(self, key_id: str) -> bytes:
        """
        Retrieve a key by ID.
        
        Args:
            key_id: Key identifier
            
        Returns:
            Key material
        """
        metadata = await self.get_key_metadata(key_id)
        
        if metadata.status == KeyStatus.REVOKED:
            raise KeyManagementException(f"Key is revoked: {key_id}")
        
        if metadata.expires_at and datetime.utcnow() >= metadata.expires_at:
            raise KeyManagementException(f"Key is expired: {key_id}")
        
        # Increment usage count
        await self._increment_usage_count(key_id)
        
        # Retrieve from HSM
        return await self.hsm.get_key(metadata.hsm_key_ref)
    
    async def get_key_metadata(self, key_id: str) -> KeyMetadata:
        """Get metadata for a key."""
        with sqlite3.connect(self.metadata_db) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM key_metadata WHERE key_id = ?",
                (key_id,)
            )
            row = cursor.fetchone()
            
            if not row:
                raise KeyManagementException(f"Key not found: {key_id}")
            
            return self._deserialize_metadata(dict(row))
    
    async def rotate_key(self, key_id: str) -> str:
        """
        Rotate a key by generating a new version.
        
        Args:
            key_id: Key to rotate
            
        Returns:
            New key identifier
        """
        old_metadata = await self.get_key_metadata(key_id)
        
        # Generate new key with incremented version
        new_key_id = f"{key_id}_v{old_metadata.version + 1}"
        
        new_key_id = await self.generate_key(
            algorithm=old_metadata.algorithm,
            key_size=old_metadata.key_size,
            key_id=new_key_id,
            expires_at=old_metadata.expires_at,
            max_usage=old_metadata.max_usage,
            tags=old_metadata.tags
        )
        
        # Mark old key as deprecated
        await self._update_key_status(key_id, KeyStatus.DEPRECATED)
        
        return new_key_id
    
    async def revoke_key(self, key_id: str):
        """Revoke a key, making it unusable."""
        await self._update_key_status(key_id, KeyStatus.REVOKED)
    
    async def create_key_escrow(self, key_id: str, escrow_public_key: bytes):
        """
        Create an escrow copy of a key encrypted with the escrow public key.
        
        Args:
            key_id: Key to escrow
            escrow_public_key: RSA public key for escrow encryption
        """
        # Get the key material
        key_material = await self.get_key(key_id)
        
        # Load escrow public key
        public_key = serialization.load_pem_public_key(escrow_public_key)
        
        # Encrypt key with escrow public key
        encrypted_key = public_key.encrypt(
            key_material,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        # Store escrow
        with sqlite3.connect(self.metadata_db) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO key_escrow 
                (key_id, encrypted_key, escrow_public_key)
                VALUES (?, ?, ?)
            """, (key_id, encrypted_key, escrow_public_key))
            conn.commit()
        
        # Update key status
        await self._update_key_status(key_id, KeyStatus.ESCROW)
    
    async def recover_from_escrow(
        self,
        key_id: str,
        escrow_private_key: bytes,
        new_key_id: Optional[str] = None
    ) -> str:
        """
        Recover a key from escrow using the escrow private key.
        
        Args:
            key_id: Original key ID
            escrow_private_key: RSA private key for decryption
            new_key_id: Optional new key identifier
            
        Returns:
            New key identifier
        """
        # Get escrow data
        with sqlite3.connect(self.metadata_db) as conn:
            cursor = conn.execute(
                "SELECT encrypted_key FROM key_escrow WHERE key_id = ?",
                (key_id,)
            )
            row = cursor.fetchone()
            
            if not row:
                raise KeyManagementException(f"No escrow found for key: {key_id}")
            
            encrypted_key = row[0]
        
        # Load escrow private key
        private_key = serialization.load_pem_private_key(escrow_private_key, password=None)
        
        # Decrypt the key
        key_material = private_key.decrypt(
            encrypted_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        # Get original metadata
        original_metadata = await self.get_key_metadata(key_id)
        
        # Create new key entry
        new_key_id = new_key_id or f"{key_id}_recovered_{secrets.token_hex(8)}"
        
        # Store recovered key in HSM
        if self.hsm_backend == HSMBackend.SOFTWARE:
            hsm_key_ref = f"recovered_{secrets.token_hex(16)}"
            with self.hsm.lock:
                encrypted_key_data = self.hsm._encrypt_key(key_material)
                with sqlite3.connect(self.hsm.key_storage) as conn:
                    conn.execute("""
                        INSERT INTO hsm_keys (key_ref, algorithm, key_size, encrypted_key)
                        VALUES (?, ?, ?, ?)
                    """, (hsm_key_ref, original_metadata.algorithm, 
                          original_metadata.key_size, encrypted_key_data))
                    conn.commit()
        
        # Create new metadata
        recovered_metadata = KeyMetadata(
            key_id=new_key_id,
            algorithm=original_metadata.algorithm,
            key_size=original_metadata.key_size,
            created_at=datetime.utcnow(),
            expires_at=original_metadata.expires_at,
            status=KeyStatus.ACTIVE,
            version=original_metadata.version,
            usage_count=0,
            max_usage=original_metadata.max_usage,
            tags=original_metadata.tags,
            derivation_info=original_metadata.derivation_info,
            hsm_key_ref=hsm_key_ref
        )
        
        # Store metadata
        with sqlite3.connect(self.metadata_db) as conn:
            serialized = self._serialize_metadata(recovered_metadata)
            conn.execute("""
                INSERT INTO key_metadata 
                (key_id, algorithm, key_size, created_at, expires_at, status, 
                 version, usage_count, max_usage, tags, derivation_info, hsm_key_ref)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                serialized['key_id'], serialized['algorithm'], serialized['key_size'],
                serialized['created_at'], serialized['expires_at'], serialized['status'],
                serialized['version'], serialized['usage_count'], serialized['max_usage'],
                serialized['tags'], serialized['derivation_info'], serialized['hsm_key_ref']
            ))
            conn.commit()
        
        return new_key_id
    
    async def list_keys(
        self,
        status: Optional[KeyStatus] = None,
        algorithm: Optional[str] = None
    ) -> List[KeyMetadata]:
        """List keys with optional filtering."""
        query = "SELECT * FROM key_metadata WHERE 1=1"
        params = []
        
        if status:
            query += " AND status = ?"
            params.append(status.value)
        
        if algorithm:
            query += " AND algorithm = ?"
            params.append(algorithm)
        
        with sqlite3.connect(self.metadata_db) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            
            return [self._deserialize_metadata(dict(row)) for row in rows]
    
    async def _increment_usage_count(self, key_id: str):
        """Increment the usage count for a key."""
        with sqlite3.connect(self.metadata_db) as conn:
            conn.execute(
                "UPDATE key_metadata SET usage_count = usage_count + 1 WHERE key_id = ?",
                (key_id,)
            )
            conn.commit()
    
    async def _update_key_status(self, key_id: str, status: KeyStatus):
        """Update the status of a key."""
        with sqlite3.connect(self.metadata_db) as conn:
            conn.execute(
                "UPDATE key_metadata SET status = ? WHERE key_id = ?",
                (status.value, key_id)
            )
            conn.commit()
    
    async def _perform_scheduled_rotation(self):
        """Perform scheduled key rotation based on policy."""
        keys = await self.list_keys(status=KeyStatus.ACTIVE)
        
        for metadata in keys:
            if self.rotation_policy.should_rotate(metadata):
                try:
                    new_key_id = await self.rotate_key(metadata.key_id)
                    logging.info(f"Rotated key {metadata.key_id} to {new_key_id}")
                except Exception as e:
                    logging.error(f"Failed to rotate key {metadata.key_id}: {e}")


# Global instance for easy access
key_management = KeyManagementSystem()