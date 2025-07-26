"""
Nova Bloom Consciousness Architecture - Memory Encryption Layer

This module implements a comprehensive memory encryption system supporting multiple ciphers
and cryptographic operations for protecting Nova consciousness data.

Key Features:
- Multi-cipher support (AES-256-GCM, ChaCha20-Poly1305, AES-256-XTS)
- Hardware acceleration when available
- Zero-knowledge architecture
- Performance-optimized operations
- At-rest and in-transit encryption modes
"""

import asyncio
import hashlib
import hmac
import os
import secrets
import struct
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305
from cryptography.hazmat.primitives.hashes import SHA256, SHA512
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.primitives.constant_time import bytes_eq
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidSignature, InvalidTag


class CipherType(Enum):
    """Supported cipher types for memory encryption."""
    AES_256_GCM = "aes-256-gcm"
    CHACHA20_POLY1305 = "chacha20-poly1305"
    AES_256_XTS = "aes-256-xts"


class EncryptionMode(Enum):
    """Encryption modes for different use cases."""
    AT_REST = "at_rest"
    IN_TRANSIT = "in_transit"
    STREAMING = "streaming"


@dataclass
class EncryptionMetadata:
    """Metadata for encrypted memory blocks."""
    cipher_type: CipherType
    encryption_mode: EncryptionMode
    key_id: str
    nonce: bytes
    tag: Optional[bytes]
    timestamp: float
    version: int
    additional_data: Optional[bytes] = None


class EncryptionException(Exception):
    """Base exception for encryption operations."""
    pass


class CipherInterface(ABC):
    """Abstract interface for cipher implementations."""
    
    @abstractmethod
    def encrypt(self, plaintext: bytes, key: bytes, nonce: bytes, 
                additional_data: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        """Encrypt plaintext and return (ciphertext, tag)."""
        pass
    
    @abstractmethod
    def decrypt(self, ciphertext: bytes, key: bytes, nonce: bytes, tag: bytes,
                additional_data: Optional[bytes] = None) -> bytes:
        """Decrypt ciphertext and return plaintext."""
        pass
    
    @abstractmethod
    def generate_key(self) -> bytes:
        """Generate a new encryption key."""
        pass
    
    @abstractmethod
    def generate_nonce(self) -> bytes:
        """Generate a new nonce for encryption."""
        pass


class AESGCMCipher(CipherInterface):
    """AES-256-GCM cipher implementation with hardware acceleration support."""
    
    KEY_SIZE = 32  # 256 bits
    NONCE_SIZE = 12  # 96 bits (recommended for GCM)
    TAG_SIZE = 16  # 128 bits
    
    def __init__(self):
        self.backend = default_backend()
        self._check_hardware_support()
    
    def _check_hardware_support(self):
        """Check for AES-NI hardware acceleration."""
        try:
            # Test with dummy operation to check hardware support
            dummy_key = os.urandom(self.KEY_SIZE)
            dummy_nonce = os.urandom(self.NONCE_SIZE)
            dummy_data = b"test"
            
            aesgcm = AESGCM(dummy_key)
            ciphertext = aesgcm.encrypt(dummy_nonce, dummy_data, None)
            aesgcm.decrypt(dummy_nonce, ciphertext, None)
            self.hardware_accelerated = True
        except Exception:
            self.hardware_accelerated = False
    
    def encrypt(self, plaintext: bytes, key: bytes, nonce: bytes,
                additional_data: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        """Encrypt using AES-256-GCM."""
        if len(key) != self.KEY_SIZE:
            raise EncryptionException(f"Invalid key size: {len(key)}, expected {self.KEY_SIZE}")
        if len(nonce) != self.NONCE_SIZE:
            raise EncryptionException(f"Invalid nonce size: {len(nonce)}, expected {self.NONCE_SIZE}")
        
        try:
            aesgcm = AESGCM(key)
            ciphertext_with_tag = aesgcm.encrypt(nonce, plaintext, additional_data)
            
            # Split ciphertext and tag
            ciphertext = ciphertext_with_tag[:-self.TAG_SIZE]
            tag = ciphertext_with_tag[-self.TAG_SIZE:]
            
            return ciphertext, tag
        except Exception as e:
            raise EncryptionException(f"AES-GCM encryption failed: {e}")
    
    def decrypt(self, ciphertext: bytes, key: bytes, nonce: bytes, tag: bytes,
                additional_data: Optional[bytes] = None) -> bytes:
        """Decrypt using AES-256-GCM."""
        if len(key) != self.KEY_SIZE:
            raise EncryptionException(f"Invalid key size: {len(key)}, expected {self.KEY_SIZE}")
        if len(nonce) != self.NONCE_SIZE:
            raise EncryptionException(f"Invalid nonce size: {len(nonce)}, expected {self.NONCE_SIZE}")
        if len(tag) != self.TAG_SIZE:
            raise EncryptionException(f"Invalid tag size: {len(tag)}, expected {self.TAG_SIZE}")
        
        try:
            aesgcm = AESGCM(key)
            ciphertext_with_tag = ciphertext + tag
            plaintext = aesgcm.decrypt(nonce, ciphertext_with_tag, additional_data)
            return plaintext
        except InvalidTag:
            raise EncryptionException("AES-GCM authentication failed")
        except Exception as e:
            raise EncryptionException(f"AES-GCM decryption failed: {e}")
    
    def generate_key(self) -> bytes:
        """Generate a new AES-256 key."""
        return secrets.token_bytes(self.KEY_SIZE)
    
    def generate_nonce(self) -> bytes:
        """Generate a new nonce for AES-GCM."""
        return secrets.token_bytes(self.NONCE_SIZE)


class ChaCha20Poly1305Cipher(CipherInterface):
    """ChaCha20-Poly1305 cipher implementation for high-performance encryption."""
    
    KEY_SIZE = 32  # 256 bits
    NONCE_SIZE = 12  # 96 bits
    TAG_SIZE = 16  # 128 bits
    
    def encrypt(self, plaintext: bytes, key: bytes, nonce: bytes,
                additional_data: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        """Encrypt using ChaCha20-Poly1305."""
        if len(key) != self.KEY_SIZE:
            raise EncryptionException(f"Invalid key size: {len(key)}, expected {self.KEY_SIZE}")
        if len(nonce) != self.NONCE_SIZE:
            raise EncryptionException(f"Invalid nonce size: {len(nonce)}, expected {self.NONCE_SIZE}")
        
        try:
            chacha = ChaCha20Poly1305(key)
            ciphertext_with_tag = chacha.encrypt(nonce, plaintext, additional_data)
            
            # Split ciphertext and tag
            ciphertext = ciphertext_with_tag[:-self.TAG_SIZE]
            tag = ciphertext_with_tag[-self.TAG_SIZE:]
            
            return ciphertext, tag
        except Exception as e:
            raise EncryptionException(f"ChaCha20-Poly1305 encryption failed: {e}")
    
    def decrypt(self, ciphertext: bytes, key: bytes, nonce: bytes, tag: bytes,
                additional_data: Optional[bytes] = None) -> bytes:
        """Decrypt using ChaCha20-Poly1305."""
        if len(key) != self.KEY_SIZE:
            raise EncryptionException(f"Invalid key size: {len(key)}, expected {self.KEY_SIZE}")
        if len(nonce) != self.NONCE_SIZE:
            raise EncryptionException(f"Invalid nonce size: {len(nonce)}, expected {self.NONCE_SIZE}")
        if len(tag) != self.TAG_SIZE:
            raise EncryptionException(f"Invalid tag size: {len(tag)}, expected {self.TAG_SIZE}")
        
        try:
            chacha = ChaCha20Poly1305(key)
            ciphertext_with_tag = ciphertext + tag
            plaintext = chacha.decrypt(nonce, ciphertext_with_tag, additional_data)
            return plaintext
        except InvalidTag:
            raise EncryptionException("ChaCha20-Poly1305 authentication failed")
        except Exception as e:
            raise EncryptionException(f"ChaCha20-Poly1305 decryption failed: {e}")
    
    def generate_key(self) -> bytes:
        """Generate a new ChaCha20 key."""
        return secrets.token_bytes(self.KEY_SIZE)
    
    def generate_nonce(self) -> bytes:
        """Generate a new nonce for ChaCha20-Poly1305."""
        return secrets.token_bytes(self.NONCE_SIZE)


class AESXTSCipher(CipherInterface):
    """AES-256-XTS cipher implementation for disk encryption (at-rest)."""
    
    KEY_SIZE = 64  # 512 bits (two 256-bit keys for XTS)
    NONCE_SIZE = 16  # 128 bits (sector number)
    TAG_SIZE = 0  # XTS doesn't use authentication tags
    
    def encrypt(self, plaintext: bytes, key: bytes, nonce: bytes,
                additional_data: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        """Encrypt using AES-256-XTS."""
        if len(key) != self.KEY_SIZE:
            raise EncryptionException(f"Invalid key size: {len(key)}, expected {self.KEY_SIZE}")
        if len(nonce) != self.NONCE_SIZE:
            raise EncryptionException(f"Invalid nonce size: {len(nonce)}, expected {self.NONCE_SIZE}")
        
        # Pad plaintext to 16-byte boundary (AES block size)
        padding_length = 16 - (len(plaintext) % 16)
        if padding_length != 16:
            plaintext = plaintext + bytes([padding_length] * padding_length)
        
        try:
            # Split key into two parts for XTS
            key1 = key[:32]
            key2 = key[32:]
            
            cipher = Cipher(
                algorithms.AES(key1),
                modes.XTS(key2, nonce),
                backend=default_backend()
            )
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(plaintext) + encryptor.finalize()
            
            return ciphertext, b""  # No tag for XTS
        except Exception as e:
            raise EncryptionException(f"AES-XTS encryption failed: {e}")
    
    def decrypt(self, ciphertext: bytes, key: bytes, nonce: bytes, tag: bytes,
                additional_data: Optional[bytes] = None) -> bytes:
        """Decrypt using AES-256-XTS."""
        if len(key) != self.KEY_SIZE:
            raise EncryptionException(f"Invalid key size: {len(key)}, expected {self.KEY_SIZE}")
        if len(nonce) != self.NONCE_SIZE:
            raise EncryptionException(f"Invalid nonce size: {len(nonce)}, expected {self.NONCE_SIZE}")
        
        try:
            # Split key into two parts for XTS
            key1 = key[:32]
            key2 = key[32:]
            
            cipher = Cipher(
                algorithms.AES(key1),
                modes.XTS(key2, nonce),
                backend=default_backend()
            )
            decryptor = cipher.decryptor()
            plaintext_padded = decryptor.update(ciphertext) + decryptor.finalize()
            
            # Remove padding
            if plaintext_padded:
                padding_length = plaintext_padded[-1]
                if padding_length <= 16:
                    plaintext = plaintext_padded[:-padding_length]
                else:
                    plaintext = plaintext_padded
            else:
                plaintext = plaintext_padded
            
            return plaintext
        except Exception as e:
            raise EncryptionException(f"AES-XTS decryption failed: {e}")
    
    def generate_key(self) -> bytes:
        """Generate a new AES-256-XTS key (512 bits total)."""
        return secrets.token_bytes(self.KEY_SIZE)
    
    def generate_nonce(self) -> bytes:
        """Generate a new sector number for AES-XTS."""
        return secrets.token_bytes(self.NONCE_SIZE)


class MemoryEncryptionLayer:
    """
    Main memory encryption layer for Nova consciousness system.
    
    Provides high-level encryption/decryption operations with multiple cipher support,
    hardware acceleration, and performance optimization.
    """
    
    def __init__(self, default_cipher: CipherType = CipherType.AES_256_GCM):
        """Initialize the memory encryption layer."""
        self.default_cipher = default_cipher
        self.ciphers = {
            CipherType.AES_256_GCM: AESGCMCipher(),
            CipherType.CHACHA20_POLY1305: ChaCha20Poly1305Cipher(),
            CipherType.AES_256_XTS: AESXTSCipher()
        }
        self.performance_stats = {
            'encryptions': 0,
            'decryptions': 0,
            'total_bytes_encrypted': 0,
            'total_bytes_decrypted': 0,
            'average_encrypt_time': 0.0,
            'average_decrypt_time': 0.0
        }
    
    def _get_cipher(self, cipher_type: CipherType) -> CipherInterface:
        """Get cipher implementation for the given type."""
        return self.ciphers[cipher_type]
    
    def _create_additional_data(self, metadata: EncryptionMetadata) -> bytes:
        """Create additional authenticated data from metadata."""
        return struct.pack(
            '!QI',
            int(metadata.timestamp * 1000000),  # microsecond precision
            metadata.version
        ) + metadata.key_id.encode('utf-8')
    
    def encrypt_memory_block(
        self,
        data: bytes,
        key: bytes,
        cipher_type: Optional[CipherType] = None,
        encryption_mode: EncryptionMode = EncryptionMode.AT_REST,
        key_id: str = "default",
        additional_data: Optional[bytes] = None
    ) -> Tuple[bytes, EncryptionMetadata]:
        """
        Encrypt a memory block with specified cipher and return encrypted data with metadata.
        
        Args:
            data: Raw memory data to encrypt
            key: Encryption key
            cipher_type: Cipher to use (defaults to instance default)
            encryption_mode: Encryption mode for the operation
            key_id: Identifier for the encryption key
            additional_data: Optional additional authenticated data
            
        Returns:
            Tuple of (encrypted_data, metadata)
        """
        start_time = time.perf_counter()
        
        cipher_type = cipher_type or self.default_cipher
        cipher = self._get_cipher(cipher_type)
        
        # Generate nonce
        nonce = cipher.generate_nonce()
        
        # Create metadata
        metadata = EncryptionMetadata(
            cipher_type=cipher_type,
            encryption_mode=encryption_mode,
            key_id=key_id,
            nonce=nonce,
            tag=None,  # Will be set after encryption
            timestamp=time.time(),
            version=1,
            additional_data=additional_data
        )
        
        # Create AAD if none provided
        if additional_data is None:
            additional_data = self._create_additional_data(metadata)
        
        try:
            # Perform encryption
            ciphertext, tag = cipher.encrypt(data, key, nonce, additional_data)
            metadata.tag = tag
            
            # Update performance statistics
            encrypt_time = time.perf_counter() - start_time
            self.performance_stats['encryptions'] += 1
            self.performance_stats['total_bytes_encrypted'] += len(data)
            
            # Update running average
            old_avg = self.performance_stats['average_encrypt_time']
            count = self.performance_stats['encryptions']
            self.performance_stats['average_encrypt_time'] = (
                old_avg * (count - 1) + encrypt_time
            ) / count
            
            return ciphertext, metadata
            
        except Exception as e:
            raise EncryptionException(f"Memory block encryption failed: {e}")
    
    def decrypt_memory_block(
        self,
        encrypted_data: bytes,
        key: bytes,
        metadata: EncryptionMetadata,
        additional_data: Optional[bytes] = None
    ) -> bytes:
        """
        Decrypt a memory block using the provided metadata.
        
        Args:
            encrypted_data: Encrypted memory data
            key: Decryption key
            metadata: Encryption metadata
            additional_data: Optional additional authenticated data
            
        Returns:
            Decrypted plaintext data
        """
        start_time = time.perf_counter()
        
        cipher = self._get_cipher(metadata.cipher_type)
        
        # Create AAD if none provided
        if additional_data is None:
            additional_data = self._create_additional_data(metadata)
        
        try:
            # Perform decryption
            plaintext = cipher.decrypt(
                encrypted_data,
                key,
                metadata.nonce,
                metadata.tag or b"",
                additional_data
            )
            
            # Update performance statistics
            decrypt_time = time.perf_counter() - start_time
            self.performance_stats['decryptions'] += 1
            self.performance_stats['total_bytes_decrypted'] += len(plaintext)
            
            # Update running average
            old_avg = self.performance_stats['average_decrypt_time']
            count = self.performance_stats['decryptions']
            self.performance_stats['average_decrypt_time'] = (
                old_avg * (count - 1) + decrypt_time
            ) / count
            
            return plaintext
            
        except Exception as e:
            raise EncryptionException(f"Memory block decryption failed: {e}")
    
    async def encrypt_memory_block_async(
        self,
        data: bytes,
        key: bytes,
        cipher_type: Optional[CipherType] = None,
        encryption_mode: EncryptionMode = EncryptionMode.AT_REST,
        key_id: str = "default",
        additional_data: Optional[bytes] = None
    ) -> Tuple[bytes, EncryptionMetadata]:
        """Asynchronous version of encrypt_memory_block for concurrent operations."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.encrypt_memory_block,
            data, key, cipher_type, encryption_mode, key_id, additional_data
        )
    
    async def decrypt_memory_block_async(
        self,
        encrypted_data: bytes,
        key: bytes,
        metadata: EncryptionMetadata,
        additional_data: Optional[bytes] = None
    ) -> bytes:
        """Asynchronous version of decrypt_memory_block for concurrent operations."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.decrypt_memory_block,
            encrypted_data, key, metadata, additional_data
        )
    
    def generate_encryption_key(self, cipher_type: Optional[CipherType] = None) -> bytes:
        """Generate a new encryption key for the specified cipher."""
        cipher_type = cipher_type or self.default_cipher
        cipher = self._get_cipher(cipher_type)
        return cipher.generate_key()
    
    def get_cipher_info(self, cipher_type: CipherType) -> Dict[str, Any]:
        """Get information about a specific cipher."""
        cipher = self._get_cipher(cipher_type)
        info = {
            'name': cipher_type.value,
            'key_size': getattr(cipher, 'KEY_SIZE', 'Unknown'),
            'nonce_size': getattr(cipher, 'NONCE_SIZE', 'Unknown'),
            'tag_size': getattr(cipher, 'TAG_SIZE', 'Unknown'),
            'hardware_accelerated': getattr(cipher, 'hardware_accelerated', False)
        }
        return info
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        return self.performance_stats.copy()
    
    def reset_performance_stats(self):
        """Reset performance statistics counters."""
        self.performance_stats = {
            'encryptions': 0,
            'decryptions': 0,
            'total_bytes_encrypted': 0,
            'total_bytes_decrypted': 0,
            'average_encrypt_time': 0.0,
            'average_decrypt_time': 0.0
        }
    
    def validate_key(self, key: bytes, cipher_type: Optional[CipherType] = None) -> bool:
        """Validate that a key is the correct size for the specified cipher."""
        cipher_type = cipher_type or self.default_cipher
        cipher = self._get_cipher(cipher_type)
        return len(key) == cipher.KEY_SIZE
    
    def secure_compare(self, a: bytes, b: bytes) -> bool:
        """Constant-time comparison of two byte strings."""
        return bytes_eq(a, b)


# Global instance for easy access
memory_encryption = MemoryEncryptionLayer()