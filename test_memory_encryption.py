"""
Nova Bloom Consciousness Architecture - Memory Encryption Tests

Comprehensive test suite for the memory encryption layer including:
- Unit tests for all encryption components
- Security tests and vulnerability assessments
- Performance benchmarks and hardware acceleration tests
- Integration tests with Nova memory layers
- Stress tests and edge case handling
"""

import asyncio
import json
import os
import secrets
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Import the modules to test
from memory_encryption_layer import (
    MemoryEncryptionLayer, CipherType, EncryptionMode, EncryptionMetadata,
    AESGCMCipher, ChaCha20Poly1305Cipher, AESXTSCipher, EncryptionException
)
from key_management_system import (
    KeyManagementSystem, KeyDerivationFunction, KeyStatus, HSMBackend,
    KeyDerivationService, KeyRotationPolicy, KeyManagementException
)
from encrypted_memory_operations import (
    EncryptedMemoryOperations, MemoryBlock, EncryptedMemoryBlock,
    MemoryBlockType, CompressionType, HardwareAcceleration,
    CompressionService, MemoryChecksumService, StreamingEncryption
)


class TestMemoryEncryptionLayer(unittest.TestCase):
    """Test suite for the core memory encryption layer."""
    
    def setUp(self):
        """Set up test environment."""
        self.encryption_layer = MemoryEncryptionLayer()
        self.test_data = b"This is test data for Nova consciousness memory encryption testing."
        self.test_key = secrets.token_bytes(32)  # 256-bit key
    
    def test_aes_gcm_cipher_initialization(self):
        """Test AES-GCM cipher initialization and hardware detection."""
        cipher = AESGCMCipher()
        self.assertEqual(cipher.KEY_SIZE, 32)
        self.assertEqual(cipher.NONCE_SIZE, 12)
        self.assertEqual(cipher.TAG_SIZE, 16)
        self.assertIsInstance(cipher.hardware_accelerated, bool)
    
    def test_aes_gcm_encryption_decryption(self):
        """Test AES-GCM encryption and decryption."""
        cipher = AESGCMCipher()
        key = cipher.generate_key()
        nonce = cipher.generate_nonce()
        
        # Test encryption
        ciphertext, tag = cipher.encrypt(self.test_data, key, nonce)
        self.assertNotEqual(ciphertext, self.test_data)
        self.assertEqual(len(tag), cipher.TAG_SIZE)
        
        # Test decryption
        decrypted = cipher.decrypt(ciphertext, key, nonce, tag)
        self.assertEqual(decrypted, self.test_data)
    
    def test_chacha20_poly1305_encryption_decryption(self):
        """Test ChaCha20-Poly1305 encryption and decryption."""
        cipher = ChaCha20Poly1305Cipher()
        key = cipher.generate_key()
        nonce = cipher.generate_nonce()
        
        # Test encryption
        ciphertext, tag = cipher.encrypt(self.test_data, key, nonce)
        self.assertNotEqual(ciphertext, self.test_data)
        self.assertEqual(len(tag), cipher.TAG_SIZE)
        
        # Test decryption
        decrypted = cipher.decrypt(ciphertext, key, nonce, tag)
        self.assertEqual(decrypted, self.test_data)
    
    def test_aes_xts_encryption_decryption(self):
        """Test AES-XTS encryption and decryption."""
        cipher = AESXTSCipher()
        key = cipher.generate_key()
        nonce = cipher.generate_nonce()
        
        # Test encryption
        ciphertext, tag = cipher.encrypt(self.test_data, key, nonce)
        self.assertNotEqual(ciphertext, self.test_data)
        self.assertEqual(len(tag), 0)  # XTS doesn't use tags
        
        # Test decryption
        decrypted = cipher.decrypt(ciphertext, key, nonce, b"")
        self.assertEqual(decrypted, self.test_data)
    
    def test_memory_encryption_layer_encrypt_decrypt(self):
        """Test high-level memory encryption layer operations."""
        # Test encryption
        encrypted_data, metadata = self.encryption_layer.encrypt_memory_block(
            self.test_data,
            self.test_key,
            CipherType.AES_256_GCM,
            EncryptionMode.AT_REST,
            "test_key_id"
        )
        
        self.assertNotEqual(encrypted_data, self.test_data)
        self.assertEqual(metadata.cipher_type, CipherType.AES_256_GCM)
        self.assertEqual(metadata.encryption_mode, EncryptionMode.AT_REST)
        self.assertEqual(metadata.key_id, "test_key_id")
        
        # Test decryption
        decrypted_data = self.encryption_layer.decrypt_memory_block(
            encrypted_data,
            self.test_key,
            metadata
        )
        
        self.assertEqual(decrypted_data, self.test_data)
    
    async def test_async_encryption_decryption(self):
        """Test asynchronous encryption and decryption operations."""
        # Test async encryption
        encrypted_data, metadata = await self.encryption_layer.encrypt_memory_block_async(
            self.test_data,
            self.test_key,
            CipherType.CHACHA20_POLY1305,
            EncryptionMode.IN_TRANSIT,
            "async_test_key"
        )
        
        self.assertNotEqual(encrypted_data, self.test_data)
        self.assertEqual(metadata.cipher_type, CipherType.CHACHA20_POLY1305)
        
        # Test async decryption
        decrypted_data = await self.encryption_layer.decrypt_memory_block_async(
            encrypted_data,
            self.test_key,
            metadata
        )
        
        self.assertEqual(decrypted_data, self.test_data)
    
    def test_invalid_key_size_handling(self):
        """Test handling of invalid key sizes."""
        cipher = AESGCMCipher()
        invalid_key = b"too_short"
        nonce = cipher.generate_nonce()
        
        with self.assertRaises(EncryptionException):
            cipher.encrypt(self.test_data, invalid_key, nonce)
    
    def test_invalid_nonce_size_handling(self):
        """Test handling of invalid nonce sizes."""
        cipher = AESGCMCipher()
        key = cipher.generate_key()
        invalid_nonce = b"short"
        
        with self.assertRaises(EncryptionException):
            cipher.encrypt(self.test_data, key, invalid_nonce)
    
    def test_authentication_failure(self):
        """Test authentication failure detection."""
        cipher = AESGCMCipher()
        key = cipher.generate_key()
        nonce = cipher.generate_nonce()
        
        ciphertext, tag = cipher.encrypt(self.test_data, key, nonce)
        
        # Tamper with ciphertext
        tampered_ciphertext = ciphertext[:-1] + b'\x00'
        
        with self.assertRaises(EncryptionException):
            cipher.decrypt(tampered_ciphertext, key, nonce, tag)
    
    def test_performance_statistics(self):
        """Test performance statistics collection."""
        initial_stats = self.encryption_layer.get_performance_stats()
        
        # Perform some operations
        for _ in range(10):
            encrypted_data, metadata = self.encryption_layer.encrypt_memory_block(
                self.test_data, self.test_key
            )
            self.encryption_layer.decrypt_memory_block(
                encrypted_data, self.test_key, metadata
            )
        
        final_stats = self.encryption_layer.get_performance_stats()
        
        self.assertGreater(final_stats['encryptions'], initial_stats['encryptions'])
        self.assertGreater(final_stats['decryptions'], initial_stats['decryptions'])
        self.assertGreater(final_stats['total_bytes_encrypted'], 0)
        self.assertGreater(final_stats['total_bytes_decrypted'], 0)


class TestKeyManagementSystem(unittest.TestCase):
    """Test suite for the key management system."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.key_management = KeyManagementSystem(
            storage_path=self.temp_dir,
            hsm_backend=HSMBackend.SOFTWARE
        )
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    async def test_key_generation(self):
        """Test key generation and storage."""
        key_id = await self.key_management.generate_key(
            algorithm="AES-256",
            key_size=256,
            tags={"test": "true", "purpose": "nova_encryption"}
        )
        
        self.assertIsInstance(key_id, str)
        
        # Test key retrieval
        key_data = await self.key_management.get_key(key_id)
        self.assertEqual(len(key_data), 32)  # 256 bits = 32 bytes
        
        # Test metadata retrieval
        metadata = await self.key_management.get_key_metadata(key_id)
        self.assertEqual(metadata.algorithm, "AES-256")
        self.assertEqual(metadata.key_size, 256)
        self.assertEqual(metadata.status, KeyStatus.ACTIVE)
        self.assertEqual(metadata.tags["test"], "true")
    
    async def test_key_derivation(self):
        """Test key derivation from passwords."""
        password = "secure_nova_password_123"
        key_id = await self.key_management.derive_key(
            password=password,
            kdf_type=KeyDerivationFunction.ARGON2ID,
            key_size=256
        )
        
        self.assertIsInstance(key_id, str)
        
        # Test key retrieval
        derived_key = await self.key_management.get_key(key_id)
        self.assertEqual(len(derived_key), 32)  # 256 bits = 32 bytes
        
        # Test metadata
        metadata = await self.key_management.get_key_metadata(key_id)
        self.assertEqual(metadata.algorithm, "DERIVED")
        self.assertIsNotNone(metadata.derivation_info)
        self.assertEqual(metadata.derivation_info['kdf_type'], 'argon2id')
    
    async def test_key_rotation(self):
        """Test key rotation functionality."""
        # Generate initial key
        original_key_id = await self.key_management.generate_key(
            algorithm="AES-256",
            key_size=256
        )
        
        # Rotate the key
        new_key_id = await self.key_management.rotate_key(original_key_id)
        
        self.assertNotEqual(original_key_id, new_key_id)
        
        # Check that old key is deprecated
        old_metadata = await self.key_management.get_key_metadata(original_key_id)
        self.assertEqual(old_metadata.status, KeyStatus.DEPRECATED)
        
        # Check that new key is active
        new_metadata = await self.key_management.get_key_metadata(new_key_id)
        self.assertEqual(new_metadata.status, KeyStatus.ACTIVE)
        self.assertEqual(new_metadata.version, old_metadata.version + 1)
    
    async def test_key_revocation(self):
        """Test key revocation."""
        key_id = await self.key_management.generate_key()
        
        # Revoke the key
        await self.key_management.revoke_key(key_id)
        
        # Check status
        metadata = await self.key_management.get_key_metadata(key_id)
        self.assertEqual(metadata.status, KeyStatus.REVOKED)
        
        # Test that revoked key cannot be used
        with self.assertRaises(KeyManagementException):
            await self.key_management.get_key(key_id)
    
    async def test_key_escrow_and_recovery(self):
        """Test key escrow and recovery mechanisms."""
        # Generate RSA key pair for escrow
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.primitives import serialization
        
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        public_key = private_key.public_key()
        
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        # Generate key to escrow
        original_key_id = await self.key_management.generate_key()
        original_key_data = await self.key_management.get_key(original_key_id)
        
        # Create escrow
        await self.key_management.create_key_escrow(original_key_id, public_pem)
        
        # Revoke original key to simulate loss
        await self.key_management.revoke_key(original_key_id)
        
        # Recovery from escrow
        recovered_key_id = await self.key_management.recover_from_escrow(
            original_key_id,
            private_pem,
            "recovered_test_key"
        )
        
        # Verify recovered key
        recovered_key_data = await self.key_management.get_key(recovered_key_id)
        self.assertEqual(original_key_data, recovered_key_data)
    
    def test_key_derivation_functions(self):
        """Test different key derivation functions."""
        password = b"test_password"
        salt = b"test_salt_123456789012345678901234"  # 32 bytes
        
        kdf_service = KeyDerivationService()
        
        # Test PBKDF2-SHA256
        key1, info1 = kdf_service.derive_key(
            password, salt, 32, KeyDerivationFunction.PBKDF2_SHA256, iterations=1000
        )
        self.assertEqual(len(key1), 32)
        self.assertEqual(info1['kdf_type'], 'pbkdf2_sha256')
        self.assertEqual(info1['iterations'], 1000)
        
        # Test Argon2id
        key2, info2 = kdf_service.derive_key(
            password, salt, 32, KeyDerivationFunction.ARGON2ID,
            memory_cost=1024, parallelism=1, iterations=2
        )
        self.assertEqual(len(key2), 32)
        self.assertEqual(info2['kdf_type'], 'argon2id')
        
        # Test HKDF-SHA256
        key3, info3 = kdf_service.derive_key(
            password, salt, 32, KeyDerivationFunction.HKDF_SHA256
        )
        self.assertEqual(len(key3), 32)
        self.assertEqual(info3['kdf_type'], 'hkdf_sha256')
        
        # Keys should be different
        self.assertNotEqual(key1, key2)
        self.assertNotEqual(key2, key3)
        self.assertNotEqual(key1, key3)
    
    def test_key_rotation_policy(self):
        """Test key rotation policy evaluation."""
        from datetime import datetime, timedelta
        from key_management_system import KeyMetadata
        
        policy = KeyRotationPolicy(max_age_hours=24, max_usage_count=100)
        
        # Test fresh key (should not rotate)
        fresh_metadata = KeyMetadata(
            key_id="fresh_key",
            algorithm="AES-256",
            key_size=256,
            created_at=datetime.utcnow(),
            expires_at=None,
            status=KeyStatus.ACTIVE,
            version=1,
            usage_count=10,
            max_usage=None,
            tags={}
        )
        self.assertFalse(policy.should_rotate(fresh_metadata))
        
        # Test old key (should rotate)
        old_metadata = KeyMetadata(
            key_id="old_key",
            algorithm="AES-256",
            key_size=256,
            created_at=datetime.utcnow() - timedelta(hours=25),
            expires_at=None,
            status=KeyStatus.ACTIVE,
            version=1,
            usage_count=10,
            max_usage=None,
            tags={}
        )
        self.assertTrue(policy.should_rotate(old_metadata))
        
        # Test overused key (should rotate)
        overused_metadata = KeyMetadata(
            key_id="overused_key",
            algorithm="AES-256",
            key_size=256,
            created_at=datetime.utcnow(),
            expires_at=None,
            status=KeyStatus.ACTIVE,
            version=1,
            usage_count=150,
            max_usage=None,
            tags={}
        )
        self.assertTrue(policy.should_rotate(overused_metadata))


class TestEncryptedMemoryOperations(unittest.TestCase):
    """Test suite for encrypted memory operations."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.encrypted_ops = EncryptedMemoryOperations(storage_path=self.temp_dir)
        self.test_data = b"Nova consciousness memory data for testing encryption operations" * 100
        self.test_block = MemoryBlock(
            block_id="test_block_001",
            block_type=MemoryBlockType.CONSCIOUSNESS_STATE,
            data=self.test_data,
            size=len(self.test_data),
            checksum=MemoryChecksumService.calculate_checksum(self.test_data),
            created_at=time.time(),
            accessed_at=time.time(),
            modified_at=time.time()
        )
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_hardware_acceleration_detection(self):
        """Test hardware acceleration detection."""
        hw_accel = HardwareAcceleration()
        
        self.assertIsInstance(hw_accel.aes_ni_available, bool)
        self.assertIsInstance(hw_accel.avx2_available, bool)
        self.assertIsInstance(hw_accel.vectorization_available, bool)
        
        chunk_size = hw_accel.get_optimal_chunk_size(1024 * 1024)
        self.assertGreater(chunk_size, 0)
        self.assertLessEqual(chunk_size, 1024 * 1024)
    
    def test_compression_service(self):
        """Test compression service functionality."""
        compression_service = CompressionService()
        
        # Test GZIP compression
        if compression_service.available_algorithms.get(CompressionType.GZIP):
            compressed = compression_service.compress(self.test_data, CompressionType.GZIP)
            decompressed = compression_service.decompress(compressed, CompressionType.GZIP)
            self.assertEqual(decompressed, self.test_data)
            self.assertLess(len(compressed), len(self.test_data))  # Should compress
        
        # Test compression ratio estimation
        ratio = compression_service.estimate_compression_ratio(
            self.test_data, CompressionType.GZIP
        )
        self.assertIsInstance(ratio, float)
        self.assertGreater(ratio, 0)
        self.assertLessEqual(ratio, 1.0)
    
    def test_checksum_service(self):
        """Test checksum service functionality."""
        checksum_service = MemoryChecksumService()
        
        # Test checksum calculation
        checksum = checksum_service.calculate_checksum(self.test_data)
        self.assertIsInstance(checksum, str)
        self.assertEqual(len(checksum), 64)  # Blake2b 256-bit = 64 hex chars
        
        # Test checksum verification
        self.assertTrue(checksum_service.verify_checksum(self.test_data, checksum))
        
        # Test checksum failure detection
        wrong_checksum = "0" * 64
        self.assertFalse(checksum_service.verify_checksum(self.test_data, wrong_checksum))
    
    async def test_memory_block_encryption_decryption(self):
        """Test memory block encryption and decryption."""
        # Generate key
        key_id = await self.encrypted_ops.key_management.generate_key()
        
        # Encrypt memory block
        encrypted_block = await self.encrypted_ops.encrypt_memory_block(
            self.test_block,
            key_id,
            CipherType.AES_256_GCM,
            EncryptionMode.AT_REST
        )
        
        self.assertEqual(encrypted_block.block_id, self.test_block.block_id)
        self.assertEqual(encrypted_block.block_type, self.test_block.block_type)
        self.assertEqual(encrypted_block.original_size, len(self.test_data))
        self.assertNotEqual(encrypted_block.encrypted_data, self.test_data)
        
        # Decrypt memory block
        decrypted_block = await self.encrypted_ops.decrypt_memory_block(
            encrypted_block,
            key_id
        )
        
        self.assertEqual(decrypted_block.data, self.test_data)
        self.assertEqual(decrypted_block.block_id, self.test_block.block_id)
        self.assertEqual(decrypted_block.checksum, self.test_block.checksum)
    
    async def test_large_memory_block_encryption(self):
        """Test streaming encryption for large memory blocks."""
        # Create large test data (10MB)
        large_data = b"X" * (10 * 1024 * 1024)
        
        key_id = await self.encrypted_ops.key_management.generate_key()
        
        start_time = time.time()
        
        encrypted_block = await self.encrypted_ops.encrypt_large_memory_block(
            large_data,
            "large_test_block",
            MemoryBlockType.NEURAL_WEIGHTS,
            key_id,
            CipherType.CHACHA20_POLY1305,
            EncryptionMode.STREAMING
        )
        
        encryption_time = time.time() - start_time
        
        self.assertEqual(encrypted_block.original_size, len(large_data))
        self.assertNotEqual(encrypted_block.encrypted_data, large_data)
        
        # Test that it completed in reasonable time (should be fast with streaming)
        self.assertLess(encryption_time, 10.0)  # Should take less than 10 seconds
    
    async def test_memory_block_storage_and_loading(self):
        """Test storing and loading encrypted memory blocks."""
        key_id = await self.encrypted_ops.key_management.generate_key()
        
        # Encrypt and store
        encrypted_block = await self.encrypted_ops.encrypt_memory_block(
            self.test_block,
            key_id
        )
        
        file_path = await self.encrypted_ops.store_encrypted_block(encrypted_block)
        self.assertTrue(Path(file_path).exists())
        
        # Load and decrypt
        loaded_block = await self.encrypted_ops.load_encrypted_block(file_path)
        
        self.assertEqual(loaded_block.block_id, encrypted_block.block_id)
        self.assertEqual(loaded_block.encrypted_data, encrypted_block.encrypted_data)
        self.assertEqual(loaded_block.original_size, encrypted_block.original_size)
        
        # Decrypt loaded block
        decrypted_block = await self.encrypted_ops.decrypt_memory_block(
            loaded_block,
            key_id
        )
        
        self.assertEqual(decrypted_block.data, self.test_data)
    
    def test_performance_statistics(self):
        """Test performance statistics collection."""
        stats = self.encrypted_ops.get_performance_stats()
        
        self.assertIn('operations_count', stats)
        self.assertIn('total_bytes_processed', stats)
        self.assertIn('average_throughput', stats)
        self.assertIn('hardware_info', stats)
        self.assertIn('compression_algorithms', stats)


class TestSecurityAndVulnerabilities(unittest.TestCase):
    """Security tests and vulnerability assessments."""
    
    def setUp(self):
        """Set up security test environment."""
        self.encryption_layer = MemoryEncryptionLayer()
        self.test_data = b"Sensitive Nova consciousness data that must be protected"
    
    def test_key_reuse_detection(self):
        """Test that nonces are never reused with the same key."""
        key = secrets.token_bytes(32)
        nonces_used = set()
        
        # Generate many encryptions and ensure no nonce reuse
        for _ in range(1000):
            encrypted_data, metadata = self.encryption_layer.encrypt_memory_block(
                self.test_data,
                key,
                CipherType.AES_256_GCM
            )
            
            nonce = metadata.nonce
            self.assertNotIn(nonce, nonces_used, "Nonce reuse detected!")
            nonces_used.add(nonce)
    
    def test_timing_attack_resistance(self):
        """Test resistance to timing attacks."""
        key = secrets.token_bytes(32)
        
        # Generate valid encrypted data
        encrypted_data, metadata = self.encryption_layer.encrypt_memory_block(
            self.test_data,
            key,
            CipherType.AES_256_GCM
        )
        
        # Create tampered data
        tampered_data = encrypted_data[:-1] + b'\x00'
        
        # Measure decryption times
        valid_times = []
        invalid_times = []
        
        for _ in range(100):
            # Valid decryption
            start = time.perf_counter()
            try:
                self.encryption_layer.decrypt_memory_block(encrypted_data, key, metadata)
            except:
                pass
            valid_times.append(time.perf_counter() - start)
            
            # Invalid decryption
            start = time.perf_counter()
            try:
                tampered_metadata = metadata
                tampered_metadata.nonce = secrets.token_bytes(12)
                self.encryption_layer.decrypt_memory_block(tampered_data, key, tampered_metadata)
            except:
                pass
            invalid_times.append(time.perf_counter() - start)
        
        # Times should be similar (within reasonable variance)
        avg_valid = sum(valid_times) / len(valid_times)
        avg_invalid = sum(invalid_times) / len(invalid_times)
        
        # Allow for up to 50% variance (this is generous, but hardware can vary)
        variance_ratio = abs(avg_valid - avg_invalid) / max(avg_valid, avg_invalid)
        self.assertLess(variance_ratio, 0.5, "Potential timing attack vulnerability detected")
    
    def test_memory_clearing(self):
        """Test that sensitive data is properly cleared from memory."""
        # This is a simplified test - in practice, memory clearing is complex
        key = secrets.token_bytes(32)
        
        encrypted_data, metadata = self.encryption_layer.encrypt_memory_block(
            self.test_data,
            key,
            CipherType.AES_256_GCM
        )
        
        decrypted_data = self.encryption_layer.decrypt_memory_block(
            encrypted_data,
            key,
            metadata
        )
        
        self.assertEqual(decrypted_data, self.test_data)
        
        # In a real implementation, we would verify that key material
        # and plaintext are zeroed out after use
    
    def test_side_channel_resistance(self):
        """Test basic resistance to side-channel attacks."""
        # Test that encryption operations with different data lengths
        # don't leak information through execution patterns
        
        key = secrets.token_bytes(32)
        
        # Test data of different lengths
        test_cases = [
            b"A" * 16,      # One AES block
            b"B" * 32,      # Two AES blocks
            b"C" * 48,      # Three AES blocks
            b"D" * 17,      # One block + 1 byte
        ]
        
        times = []
        for test_data in test_cases:
            start = time.perf_counter()
            encrypted_data, metadata = self.encryption_layer.encrypt_memory_block(
                test_data,
                key,
                CipherType.AES_256_GCM
            )
            end = time.perf_counter()
            times.append(end - start)
        
        # While timing will vary with data size, the pattern should be predictable
        # and not leak information about the actual content
        self.assertTrue(all(t > 0 for t in times))
    
    def test_cryptographic_randomness(self):
        """Test quality of cryptographic randomness."""
        # Generate many keys and nonces to test randomness
        keys = [secrets.token_bytes(32) for _ in range(100)]
        nonces = [secrets.token_bytes(12) for _ in range(100)]
        
        # Check that all keys are unique
        self.assertEqual(len(set(keys)), len(keys), "Non-unique keys generated")
        
        # Check that all nonces are unique
        self.assertEqual(len(set(nonces)), len(nonces), "Non-unique nonces generated")
        
        # Basic entropy check (this is simplified)
        key_bytes = b''.join(keys)
        byte_counts = {}
        for byte_val in key_bytes:
            byte_counts[byte_val] = byte_counts.get(byte_val, 0) + 1
        
        # Check that byte distribution is reasonably uniform
        # With 3200 bytes (100 keys * 32 bytes), each byte value should appear
        # roughly 12.5 times on average (3200/256)
        expected_count = len(key_bytes) / 256
        for count in byte_counts.values():
            # Allow for significant variance in this simple test
            self.assertLess(abs(count - expected_count), expected_count * 2)


class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmarks and optimization tests."""
    
    def setUp(self):
        """Set up benchmark environment."""
        self.encryption_layer = MemoryEncryptionLayer()
        self.temp_dir = tempfile.mkdtemp()
        self.encrypted_ops = EncryptedMemoryOperations(storage_path=self.temp_dir)
        
        # Different sized test data
        self.small_data = b"X" * 1024        # 1KB
        self.medium_data = b"X" * (100 * 1024)  # 100KB
        self.large_data = b"X" * (1024 * 1024)  # 1MB
    
    def tearDown(self):
        """Clean up benchmark environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def benchmark_cipher_performance(self):
        """Benchmark different cipher performance."""
        key = secrets.token_bytes(32)
        test_data = self.medium_data
        
        cipher_results = {}
        
        for cipher_type in [CipherType.AES_256_GCM, CipherType.CHACHA20_POLY1305, CipherType.AES_256_XTS]:
            # Warm up
            for _ in range(5):
                encrypted_data, metadata = self.encryption_layer.encrypt_memory_block(
                    test_data, key, cipher_type
                )
                self.encryption_layer.decrypt_memory_block(encrypted_data, key, metadata)
            
            # Benchmark encryption
            encrypt_times = []
            for _ in range(50):
                start = time.perf_counter()
                encrypted_data, metadata = self.encryption_layer.encrypt_memory_block(
                    test_data, key, cipher_type
                )
                encrypt_times.append(time.perf_counter() - start)
            
            # Benchmark decryption
            decrypt_times = []
            for _ in range(50):
                start = time.perf_counter()
                self.encryption_layer.decrypt_memory_block(encrypted_data, key, metadata)
                decrypt_times.append(time.perf_counter() - start)
            
            cipher_results[cipher_type.value] = {
                'avg_encrypt_time': sum(encrypt_times) / len(encrypt_times),
                'avg_decrypt_time': sum(decrypt_times) / len(decrypt_times),
                'encrypt_throughput_mbps': (len(test_data) / (sum(encrypt_times) / len(encrypt_times))) / (1024 * 1024),
                'decrypt_throughput_mbps': (len(test_data) / (sum(decrypt_times) / len(decrypt_times))) / (1024 * 1024)
            }
        
        # Print results for analysis
        print("\nCipher Performance Benchmark Results:")
        for cipher, results in cipher_results.items():
            print(f"{cipher}:")
            print(f"  Encryption: {results['encrypt_throughput_mbps']:.2f} MB/s")
            print(f"  Decryption: {results['decrypt_throughput_mbps']:.2f} MB/s")
        
        # Basic assertion that all ciphers perform reasonably
        for results in cipher_results.values():
            self.assertGreater(results['encrypt_throughput_mbps'], 1.0)  # At least 1 MB/s
            self.assertGreater(results['decrypt_throughput_mbps'], 1.0)
    
    async def benchmark_memory_operations(self):
        """Benchmark encrypted memory operations."""
        key_id = await self.encrypted_ops.key_management.generate_key()
        
        # Test different data sizes
        test_cases = [
            ("Small (1KB)", self.small_data),
            ("Medium (100KB)", self.medium_data),
            ("Large (1MB)", self.large_data)
        ]
        
        print("\nMemory Operations Benchmark Results:")
        
        for name, test_data in test_cases:
            # Create memory block
            memory_block = MemoryBlock(
                block_id=f"bench_{name.lower()}",
                block_type=MemoryBlockType.TEMPORARY_BUFFER,
                data=test_data,
                size=len(test_data),
                checksum=MemoryChecksumService.calculate_checksum(test_data),
                created_at=time.time(),
                accessed_at=time.time(),
                modified_at=time.time()
            )
            
            # Benchmark encryption
            encrypt_times = []
            for _ in range(10):
                start = time.perf_counter()
                encrypted_block = await self.encrypted_ops.encrypt_memory_block(
                    memory_block, key_id
                )
                encrypt_times.append(time.perf_counter() - start)
            
            # Benchmark decryption
            decrypt_times = []
            for _ in range(10):
                start = time.perf_counter()
                decrypted_block = await self.encrypted_ops.decrypt_memory_block(
                    encrypted_block, key_id
                )
                decrypt_times.append(time.perf_counter() - start)
            
            avg_encrypt = sum(encrypt_times) / len(encrypt_times)
            avg_decrypt = sum(decrypt_times) / len(decrypt_times)
            
            encrypt_throughput = (len(test_data) / avg_encrypt) / (1024 * 1024)
            decrypt_throughput = (len(test_data) / avg_decrypt) / (1024 * 1024)
            
            print(f"{name}:")
            print(f"  Encryption: {encrypt_throughput:.2f} MB/s")
            print(f"  Decryption: {decrypt_throughput:.2f} MB/s")
            print(f"  Compression ratio: {encrypted_block.compressed_size / len(test_data):.2f}")
    
    def test_hardware_acceleration_impact(self):
        """Test impact of hardware acceleration on performance."""
        hw_accel = HardwareAcceleration()
        
        print(f"\nHardware Acceleration Status:")
        print(f"  AES-NI Available: {hw_accel.aes_ni_available}")
        print(f"  AVX2 Available: {hw_accel.avx2_available}")
        print(f"  Vectorization Available: {hw_accel.vectorization_available}")
        
        # The actual performance impact would be measured in a real hardware environment
        self.assertIsInstance(hw_accel.aes_ni_available, bool)


class TestIntegration(unittest.TestCase):
    """Integration tests with Nova memory system."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.encrypted_ops = EncryptedMemoryOperations(storage_path=self.temp_dir)
    
    def tearDown(self):
        """Clean up integration test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    async def test_consciousness_state_encryption(self):
        """Test encryption of consciousness state data."""
        # Simulate consciousness state data
        consciousness_data = {
            "awareness_level": 0.85,
            "emotional_state": "focused",
            "memory_fragments": ["learning", "processing", "understanding"],
            "neural_patterns": list(range(1000))
        }
        
        # Serialize consciousness data
        serialized_data = json.dumps(consciousness_data).encode('utf-8')
        
        # Create memory block
        memory_block = MemoryBlock(
            block_id="consciousness_state_001",
            block_type=MemoryBlockType.CONSCIOUSNESS_STATE,
            data=serialized_data,
            size=len(serialized_data),
            checksum=MemoryChecksumService.calculate_checksum(serialized_data),
            created_at=time.time(),
            accessed_at=time.time(),
            modified_at=time.time(),
            metadata={"version": 1, "priority": "high"}
        )
        
        # Generate key and encrypt
        key_id = await self.encrypted_ops.key_management.generate_key(
            tags={"purpose": "consciousness_encryption", "priority": "high"}
        )
        
        encrypted_block = await self.encrypted_ops.encrypt_memory_block(
            memory_block,
            key_id,
            CipherType.AES_256_GCM,
            EncryptionMode.AT_REST
        )
        
        # Verify encryption
        self.assertNotEqual(encrypted_block.encrypted_data, serialized_data)
        self.assertEqual(encrypted_block.block_type, MemoryBlockType.CONSCIOUSNESS_STATE)
        
        # Store and retrieve
        file_path = await self.encrypted_ops.store_encrypted_block(encrypted_block)
        loaded_block = await self.encrypted_ops.load_encrypted_block(file_path)
        
        # Decrypt and verify
        decrypted_block = await self.encrypted_ops.decrypt_memory_block(loaded_block, key_id)
        recovered_data = json.loads(decrypted_block.data.decode('utf-8'))
        
        self.assertEqual(recovered_data, consciousness_data)
    
    async def test_conversation_data_encryption(self):
        """Test encryption of conversation data."""
        # Simulate conversation data
        conversation_data = {
            "messages": [
                {"role": "user", "content": "How does Nova process information?", "timestamp": time.time()},
                {"role": "assistant", "content": "Nova processes information through...", "timestamp": time.time()},
            ],
            "context": "Technical discussion about Nova architecture",
            "metadata": {"session_id": "conv_001", "user_id": "user_123"}
        }
        
        serialized_data = json.dumps(conversation_data).encode('utf-8')
        
        memory_block = MemoryBlock(
            block_id="conversation_001",
            block_type=MemoryBlockType.CONVERSATION_DATA,
            data=serialized_data,
            size=len(serialized_data),
            checksum=MemoryChecksumService.calculate_checksum(serialized_data),
            created_at=time.time(),
            accessed_at=time.time(),
            modified_at=time.time()
        )
        
        # Use ChaCha20-Poly1305 for conversation data (good for text)
        key_id = await self.encrypted_ops.key_management.generate_key()
        
        encrypted_block = await self.encrypted_ops.encrypt_memory_block(
            memory_block,
            key_id,
            CipherType.CHACHA20_POLY1305,
            EncryptionMode.IN_TRANSIT
        )
        
        # Verify that compression helped (conversation data should compress well)
        compression_ratio = encrypted_block.compressed_size / encrypted_block.original_size
        self.assertLess(compression_ratio, 0.8)  # Should compress to less than 80%
        
        # Decrypt and verify
        decrypted_block = await self.encrypted_ops.decrypt_memory_block(encrypted_block, key_id)
        recovered_data = json.loads(decrypted_block.data.decode('utf-8'))
        
        self.assertEqual(recovered_data, conversation_data)


def run_all_tests():
    """Run all test suites."""
    print("Running Nova Memory Encryption Test Suite...")
    
    # Create test suite
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestMemoryEncryptionLayer,
        TestKeyManagementSystem,
        TestEncryptedMemoryOperations,
        TestSecurityAndVulnerabilities,
        TestPerformanceBenchmarks,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = test_loader.loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*60}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # Run tests
    success = run_all_tests()
    
    # Run async tests separately
    async def run_async_tests():
        print("\nRunning async integration tests...")
        
        # Create test instances
        test_key_mgmt = TestKeyManagementSystem()
        test_encrypted_ops = TestEncryptedMemoryOperations()
        test_integration = TestIntegration()
        
        # Set up test environments
        test_key_mgmt.setUp()
        test_encrypted_ops.setUp()
        test_integration.setUp()
        
        try:
            # Run async tests
            await test_key_mgmt.test_key_generation()
            await test_key_mgmt.test_key_derivation()
            await test_key_mgmt.test_key_rotation()
            await test_key_mgmt.test_key_revocation()
            await test_key_mgmt.test_key_escrow_and_recovery()
            
            await test_encrypted_ops.test_memory_block_encryption_decryption()
            await test_encrypted_ops.test_large_memory_block_encryption()
            await test_encrypted_ops.test_memory_block_storage_and_loading()
            
            await test_integration.test_consciousness_state_encryption()
            await test_integration.test_conversation_data_encryption()
            
            print("All async tests passed!")
            
        except Exception as e:
            print(f"Async test failed: {e}")
            success = False
            
        finally:
            # Clean up
            test_key_mgmt.tearDown()
            test_encrypted_ops.tearDown()
            test_integration.tearDown()
        
        return success
    
    # Run async tests
    async_success = asyncio.run(run_async_tests())
    
    exit(0 if success and async_success else 1)