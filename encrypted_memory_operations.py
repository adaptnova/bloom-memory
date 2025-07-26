"""
Nova Bloom Consciousness Architecture - Encrypted Memory Operations

This module implements high-performance encrypted memory operations with hardware acceleration,
streaming support, and integration with the Nova memory layer architecture.

Key Features:
- Performance-optimized encryption/decryption operations
- Hardware acceleration detection and utilization (AES-NI, etc.)
- Streaming encryption for large memory blocks
- At-rest and in-transit encryption modes
- Memory-mapped file encryption
- Integration with Nova memory layers
"""

import asyncio
import mmap
import os
import struct
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
from memory_encryption_layer import (
    MemoryEncryptionLayer, CipherType, EncryptionMode, EncryptionMetadata
)
from key_management_system import KeyManagementSystem


class MemoryBlockType(Enum):
    """Types of memory blocks for encryption."""
    CONSCIOUSNESS_STATE = "consciousness_state"
    MEMORY_LAYER = "memory_layer"
    CONVERSATION_DATA = "conversation_data"
    NEURAL_WEIGHTS = "neural_weights"
    TEMPORARY_BUFFER = "temporary_buffer"
    PERSISTENT_STORAGE = "persistent_storage"


class CompressionType(Enum):
    """Compression algorithms for memory blocks."""
    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"
    ZSTD = "zstd"


@dataclass
class MemoryBlock:
    """Represents a memory block with metadata."""
    block_id: str
    block_type: MemoryBlockType
    data: bytes
    size: int
    checksum: str
    created_at: float
    accessed_at: float
    modified_at: float
    compression: CompressionType = CompressionType.NONE
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EncryptedMemoryBlock:
    """Represents an encrypted memory block."""
    block_id: str
    block_type: MemoryBlockType
    encrypted_data: bytes
    encryption_metadata: EncryptionMetadata
    original_size: int
    compressed_size: int
    compression: CompressionType
    checksum: str
    created_at: float
    accessed_at: float
    modified_at: float
    metadata: Optional[Dict[str, Any]] = None


class HardwareAcceleration:
    """Hardware acceleration detection and management."""
    
    def __init__(self):
        self.aes_ni_available = self._check_aes_ni()
        self.avx2_available = self._check_avx2()
        self.vectorization_available = self._check_vectorization()
    
    def _check_aes_ni(self) -> bool:
        """Check for AES-NI hardware acceleration."""
        try:
            import cpuinfo
            cpu_info = cpuinfo.get_cpu_info()
            return 'aes' in cpu_info.get('flags', [])
        except ImportError:
            # Fallback: try to detect through /proc/cpuinfo
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    content = f.read()
                    return 'aes' in content
            except:
                return False
    
    def _check_avx2(self) -> bool:
        """Check for AVX2 support."""
        try:
            import cpuinfo
            cpu_info = cpuinfo.get_cpu_info()
            return 'avx2' in cpu_info.get('flags', [])
        except ImportError:
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    content = f.read()
                    return 'avx2' in content
            except:
                return False
    
    def _check_vectorization(self) -> bool:
        """Check if NumPy is compiled with vectorization support."""
        try:
            return hasattr(np.core._multiarray_umath, 'hardware_detect')
        except:
            return False
    
    def get_optimal_chunk_size(self, data_size: int) -> int:
        """Calculate optimal chunk size for the given data size and hardware."""
        base_chunk = 64 * 1024  # 64KB base
        
        if self.avx2_available:
            # AVX2 can process 32 bytes at a time
            return min(data_size, base_chunk * 4)
        elif self.aes_ni_available:
            # AES-NI processes 16 bytes at a time
            return min(data_size, base_chunk * 2)
        else:
            return min(data_size, base_chunk)


class CompressionService:
    """Service for compressing memory blocks before encryption."""
    
    def __init__(self):
        self.available_algorithms = self._check_available_algorithms()
    
    def _check_available_algorithms(self) -> Dict[CompressionType, bool]:
        """Check which compression algorithms are available."""
        available = {CompressionType.NONE: True}
        
        try:
            import gzip
            available[CompressionType.GZIP] = True
        except ImportError:
            available[CompressionType.GZIP] = False
        
        try:
            import lz4.frame
            available[CompressionType.LZ4] = True
        except ImportError:
            available[CompressionType.LZ4] = False
        
        try:
            import zstandard as zstd
            available[CompressionType.ZSTD] = True
        except ImportError:
            available[CompressionType.ZSTD] = False
        
        return available
    
    def compress(self, data: bytes, algorithm: CompressionType) -> bytes:
        """Compress data using the specified algorithm."""
        if algorithm == CompressionType.NONE:
            return data
        
        if not self.available_algorithms.get(algorithm, False):
            raise ValueError(f"Compression algorithm not available: {algorithm}")
        
        if algorithm == CompressionType.GZIP:
            import gzip
            return gzip.compress(data, compresslevel=6)
        
        elif algorithm == CompressionType.LZ4:
            import lz4.frame
            return lz4.frame.compress(data, compression_level=1)
        
        elif algorithm == CompressionType.ZSTD:
            import zstandard as zstd
            cctx = zstd.ZstdCompressor(level=3)
            return cctx.compress(data)
        
        else:
            raise ValueError(f"Unsupported compression algorithm: {algorithm}")
    
    def decompress(self, data: bytes, algorithm: CompressionType) -> bytes:
        """Decompress data using the specified algorithm."""
        if algorithm == CompressionType.NONE:
            return data
        
        if not self.available_algorithms.get(algorithm, False):
            raise ValueError(f"Compression algorithm not available: {algorithm}")
        
        if algorithm == CompressionType.GZIP:
            import gzip
            return gzip.decompress(data)
        
        elif algorithm == CompressionType.LZ4:
            import lz4.frame
            return lz4.frame.decompress(data)
        
        elif algorithm == CompressionType.ZSTD:
            import zstandard as zstd
            dctx = zstd.ZstdDecompressor()
            return dctx.decompress(data)
        
        else:
            raise ValueError(f"Unsupported compression algorithm: {algorithm}")
    
    def estimate_compression_ratio(self, data: bytes, algorithm: CompressionType) -> float:
        """Estimate compression ratio for the data and algorithm."""
        if algorithm == CompressionType.NONE:
            return 1.0
        
        # Sample-based estimation for performance
        sample_size = min(4096, len(data))
        sample_data = data[:sample_size]
        
        try:
            compressed_sample = self.compress(sample_data, algorithm)
            return len(compressed_sample) / len(sample_data)
        except:
            return 1.0  # Fallback to no compression


class MemoryChecksumService:
    """Service for calculating and verifying memory block checksums."""
    
    @staticmethod
    def calculate_checksum(data: bytes, algorithm: str = "blake2b") -> str:
        """Calculate checksum for data."""
        if algorithm == "blake2b":
            import hashlib
            return hashlib.blake2b(data, digest_size=32).hexdigest()
        elif algorithm == "sha256":
            import hashlib
            return hashlib.sha256(data).hexdigest()
        else:
            raise ValueError(f"Unsupported checksum algorithm: {algorithm}")
    
    @staticmethod
    def verify_checksum(data: bytes, expected_checksum: str, algorithm: str = "blake2b") -> bool:
        """Verify data checksum."""
        calculated_checksum = MemoryChecksumService.calculate_checksum(data, algorithm)
        return calculated_checksum == expected_checksum


class StreamingEncryption:
    """Streaming encryption for large memory blocks."""
    
    def __init__(
        self,
        encryption_layer: MemoryEncryptionLayer,
        key_management: KeyManagementSystem,
        chunk_size: int = 64 * 1024  # 64KB chunks
    ):
        self.encryption_layer = encryption_layer
        self.key_management = key_management
        self.chunk_size = chunk_size
        self.hardware_accel = HardwareAcceleration()
    
    async def encrypt_stream(
        self,
        data_stream: AsyncIterator[bytes],
        key_id: str,
        cipher_type: CipherType = CipherType.AES_256_GCM,
        encryption_mode: EncryptionMode = EncryptionMode.STREAMING
    ) -> AsyncIterator[Tuple[bytes, EncryptionMetadata]]:
        """Encrypt a data stream in chunks."""
        key = await self.key_management.get_key(key_id)
        chunk_index = 0
        
        async for chunk in data_stream:
            if not chunk:
                continue
            
            # Create unique additional data for each chunk
            additional_data = struct.pack('!Q', chunk_index)
            
            encrypted_chunk, metadata = self.encryption_layer.encrypt_memory_block(
                chunk,
                key,
                cipher_type,
                encryption_mode,
                key_id,
                additional_data
            )
            
            chunk_index += 1
            yield encrypted_chunk, metadata
    
    async def decrypt_stream(
        self,
        encrypted_stream: AsyncIterator[Tuple[bytes, EncryptionMetadata]],
        key_id: str
    ) -> AsyncIterator[bytes]:
        """Decrypt an encrypted data stream."""
        key = await self.key_management.get_key(key_id)
        chunk_index = 0
        
        async for encrypted_chunk, metadata in encrypted_stream:
            # Reconstruct additional data
            additional_data = struct.pack('!Q', chunk_index)
            
            decrypted_chunk = self.encryption_layer.decrypt_memory_block(
                encrypted_chunk,
                key,
                metadata,
                additional_data
            )
            
            chunk_index += 1
            yield decrypted_chunk


class EncryptedMemoryOperations:
    """
    High-performance encrypted memory operations for Nova consciousness system.
    
    Provides optimized encryption/decryption operations with hardware acceleration,
    compression, streaming support, and integration with the memory layer architecture.
    """
    
    def __init__(
        self,
        encryption_layer: Optional[MemoryEncryptionLayer] = None,
        key_management: Optional[KeyManagementSystem] = None,
        storage_path: str = "/nfs/novas/system/memory/encrypted",
        enable_compression: bool = True,
        default_cipher: CipherType = CipherType.AES_256_GCM
    ):
        """Initialize encrypted memory operations."""
        self.encryption_layer = encryption_layer or MemoryEncryptionLayer(default_cipher)
        self.key_management = key_management or KeyManagementSystem()
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.enable_compression = enable_compression
        self.default_cipher = default_cipher
        
        # Initialize services
        self.compression_service = CompressionService()
        self.checksum_service = MemoryChecksumService()
        self.hardware_accel = HardwareAcceleration()
        self.streaming_encryption = StreamingEncryption(
            self.encryption_layer,
            self.key_management,
            self.hardware_accel.get_optimal_chunk_size(1024 * 1024)  # 1MB base
        )
        
        # Thread pool for parallel operations
        self.thread_pool = ThreadPoolExecutor(max_workers=os.cpu_count())
        
        # Performance statistics
        self.performance_stats = {
            'operations_count': 0,
            'total_bytes_processed': 0,
            'average_throughput': 0.0,
            'compression_ratio': 0.0,
            'hardware_acceleration_used': False
        }
        
        self.lock = threading.RLock()
    
    def _select_optimal_compression(self, data: bytes, block_type: MemoryBlockType) -> CompressionType:
        """Select the optimal compression algorithm for the given data and block type."""
        if not self.enable_compression or len(data) < 1024:  # Don't compress small blocks
            return CompressionType.NONE
        
        # Different block types benefit from different compression algorithms
        if block_type in [MemoryBlockType.NEURAL_WEIGHTS, MemoryBlockType.CONSCIOUSNESS_STATE]:
            # Neural data often compresses well with ZSTD
            if self.compression_service.available_algorithms.get(CompressionType.ZSTD):
                return CompressionType.ZSTD
        
        elif block_type == MemoryBlockType.CONVERSATION_DATA:
            # Text data compresses well with gzip
            if self.compression_service.available_algorithms.get(CompressionType.GZIP):
                return CompressionType.GZIP
        
        elif block_type == MemoryBlockType.TEMPORARY_BUFFER:
            # Fast compression for temporary data
            if self.compression_service.available_algorithms.get(CompressionType.LZ4):
                return CompressionType.LZ4
        
        # Default to LZ4 for speed if available, otherwise gzip
        if self.compression_service.available_algorithms.get(CompressionType.LZ4):
            return CompressionType.LZ4
        elif self.compression_service.available_algorithms.get(CompressionType.GZIP):
            return CompressionType.GZIP
        else:
            return CompressionType.NONE
    
    async def encrypt_memory_block(
        self,
        memory_block: MemoryBlock,
        key_id: str,
        cipher_type: Optional[CipherType] = None,
        encryption_mode: EncryptionMode = EncryptionMode.AT_REST
    ) -> EncryptedMemoryBlock:
        """
        Encrypt a memory block with optimal compression and hardware acceleration.
        
        Args:
            memory_block: Memory block to encrypt
            key_id: Key identifier for encryption
            cipher_type: Cipher to use (defaults to instance default)
            encryption_mode: Encryption mode
            
        Returns:
            Encrypted memory block
        """
        start_time = time.perf_counter()
        cipher_type = cipher_type or self.default_cipher
        
        # Verify checksum
        if not self.checksum_service.verify_checksum(memory_block.data, memory_block.checksum):
            raise ValueError(f"Checksum verification failed for block {memory_block.block_id}")
        
        # Select and apply compression
        compression_type = self._select_optimal_compression(memory_block.data, memory_block.block_type)
        compressed_data = self.compression_service.compress(memory_block.data, compression_type)
        
        # Get encryption key
        key = await self.key_management.get_key(key_id)
        
        # Create additional authenticated data
        aad = self._create_block_aad(memory_block, compression_type)
        
        # Encrypt the compressed data
        encrypted_data, encryption_metadata = await self.encryption_layer.encrypt_memory_block_async(
            compressed_data,
            key,
            cipher_type,
            encryption_mode,
            key_id,
            aad
        )
        
        # Create encrypted memory block
        current_time = time.time()
        encrypted_block = EncryptedMemoryBlock(
            block_id=memory_block.block_id,
            block_type=memory_block.block_type,
            encrypted_data=encrypted_data,
            encryption_metadata=encryption_metadata,
            original_size=len(memory_block.data),
            compressed_size=len(compressed_data),
            compression=compression_type,
            checksum=memory_block.checksum,
            created_at=memory_block.created_at,
            accessed_at=current_time,
            modified_at=current_time,
            metadata=memory_block.metadata
        )
        
        # Update performance statistics
        processing_time = time.perf_counter() - start_time
        self._update_performance_stats(len(memory_block.data), processing_time)
        
        return encrypted_block
    
    async def decrypt_memory_block(
        self,
        encrypted_block: EncryptedMemoryBlock,
        key_id: str
    ) -> MemoryBlock:
        """
        Decrypt an encrypted memory block.
        
        Args:
            encrypted_block: Encrypted memory block to decrypt
            key_id: Key identifier for decryption
            
        Returns:
            Decrypted memory block
        """
        start_time = time.perf_counter()
        
        # Get decryption key
        key = await self.key_management.get_key(key_id)
        
        # Create additional authenticated data
        aad = self._create_block_aad_from_encrypted(encrypted_block)
        
        # Decrypt the data
        compressed_data = await self.encryption_layer.decrypt_memory_block_async(
            encrypted_block.encrypted_data,
            key,
            encrypted_block.encryption_metadata,
            aad
        )
        
        # Decompress the data
        decrypted_data = self.compression_service.decompress(
            compressed_data,
            encrypted_block.compression
        )
        
        # Verify checksum
        if not self.checksum_service.verify_checksum(decrypted_data, encrypted_block.checksum):
            raise ValueError(f"Checksum verification failed for decrypted block {encrypted_block.block_id}")
        
        # Create memory block
        current_time = time.time()
        memory_block = MemoryBlock(
            block_id=encrypted_block.block_id,
            block_type=encrypted_block.block_type,
            data=decrypted_data,
            size=len(decrypted_data),
            checksum=encrypted_block.checksum,
            created_at=encrypted_block.created_at,
            accessed_at=current_time,
            modified_at=encrypted_block.modified_at,
            compression=encrypted_block.compression,
            metadata=encrypted_block.metadata
        )
        
        # Update performance statistics
        processing_time = time.perf_counter() - start_time
        self._update_performance_stats(len(decrypted_data), processing_time)
        
        return memory_block
    
    async def encrypt_large_memory_block(
        self,
        data: bytes,
        block_id: str,
        block_type: MemoryBlockType,
        key_id: str,
        cipher_type: Optional[CipherType] = None,
        encryption_mode: EncryptionMode = EncryptionMode.STREAMING
    ) -> EncryptedMemoryBlock:
        """
        Encrypt a large memory block using streaming encryption.
        
        Args:
            data: Large data to encrypt
            block_id: Block identifier
            block_type: Type of memory block
            key_id: Key identifier
            cipher_type: Cipher to use
            encryption_mode: Encryption mode
            
        Returns:
            Encrypted memory block
        """
        # Calculate checksum
        checksum = self.checksum_service.calculate_checksum(data)
        
        # Select compression
        compression_type = self._select_optimal_compression(data, block_type)
        compressed_data = self.compression_service.compress(data, compression_type)
        
        # Create memory block
        memory_block = MemoryBlock(
            block_id=block_id,
            block_type=block_type,
            data=compressed_data,
            size=len(data),
            checksum=checksum,
            created_at=time.time(),
            accessed_at=time.time(),
            modified_at=time.time(),
            compression=compression_type
        )
        
        # Use streaming encryption for large blocks
        chunk_size = self.hardware_accel.get_optimal_chunk_size(len(compressed_data))
        
        async def data_chunks():
            for i in range(0, len(compressed_data), chunk_size):
                yield compressed_data[i:i + chunk_size]
        
        encrypted_chunks = []
        encryption_metadata = None
        
        async for encrypted_chunk, metadata in self.streaming_encryption.encrypt_stream(
            data_chunks(), key_id, cipher_type or self.default_cipher, encryption_mode
        ):
            encrypted_chunks.append(encrypted_chunk)
            if encryption_metadata is None:
                encryption_metadata = metadata
        
        # Combine encrypted chunks
        combined_encrypted_data = b''.join(encrypted_chunks)
        
        # Create encrypted block
        encrypted_block = EncryptedMemoryBlock(
            block_id=block_id,
            block_type=block_type,
            encrypted_data=combined_encrypted_data,
            encryption_metadata=encryption_metadata,
            original_size=len(data),
            compressed_size=len(compressed_data),
            compression=compression_type,
            checksum=checksum,
            created_at=memory_block.created_at,
            accessed_at=memory_block.accessed_at,
            modified_at=memory_block.modified_at,
            metadata=memory_block.metadata
        )
        
        return encrypted_block
    
    async def store_encrypted_block(
        self,
        encrypted_block: EncryptedMemoryBlock,
        persistent: bool = True
    ) -> str:
        """
        Store an encrypted memory block to disk.
        
        Args:
            encrypted_block: Block to store
            persistent: Whether to store persistently
            
        Returns:
            File path where the block was stored
        """
        # Create storage path
        storage_dir = self.storage_path / encrypted_block.block_type.value
        storage_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = storage_dir / f"{encrypted_block.block_id}.encrypted"
        
        # Serialize block metadata and data
        metadata_dict = {
            'block_id': encrypted_block.block_id,
            'block_type': encrypted_block.block_type.value,
            'encryption_metadata': {
                'cipher_type': encrypted_block.encryption_metadata.cipher_type.value,
                'encryption_mode': encrypted_block.encryption_metadata.encryption_mode.value,
                'key_id': encrypted_block.encryption_metadata.key_id,
                'nonce': encrypted_block.encryption_metadata.nonce.hex(),
                'tag': encrypted_block.encryption_metadata.tag.hex() if encrypted_block.encryption_metadata.tag else None,
                'timestamp': encrypted_block.encryption_metadata.timestamp,
                'version': encrypted_block.encryption_metadata.version,
                'additional_data': encrypted_block.encryption_metadata.additional_data.hex() if encrypted_block.encryption_metadata.additional_data else None
            },
            'original_size': encrypted_block.original_size,
            'compressed_size': encrypted_block.compressed_size,
            'compression': encrypted_block.compression.value,
            'checksum': encrypted_block.checksum,
            'created_at': encrypted_block.created_at,
            'accessed_at': encrypted_block.accessed_at,
            'modified_at': encrypted_block.modified_at,
            'metadata': encrypted_block.metadata
        }
        
        # Store using memory-mapped file for efficiency
        with open(file_path, 'wb') as f:
            # Write metadata length
            metadata_json = json.dumps(metadata_dict).encode('utf-8')
            f.write(struct.pack('!I', len(metadata_json)))
            
            # Write metadata
            f.write(metadata_json)
            
            # Write encrypted data
            f.write(encrypted_block.encrypted_data)
        
        return str(file_path)
    
    async def load_encrypted_block(self, file_path: str) -> EncryptedMemoryBlock:
        """Load an encrypted memory block from disk."""
        import json
        from memory_encryption_layer import EncryptionMetadata, CipherType, EncryptionMode
        
        with open(file_path, 'rb') as f:
            # Read metadata length
            metadata_length = struct.unpack('!I', f.read(4))[0]
            
            # Read metadata
            metadata_json = f.read(metadata_length)
            metadata_dict = json.loads(metadata_json.decode('utf-8'))
            
            # Read encrypted data
            encrypted_data = f.read()
        
        # Reconstruct encryption metadata
        enc_meta_dict = metadata_dict['encryption_metadata']
        encryption_metadata = EncryptionMetadata(
            cipher_type=CipherType(enc_meta_dict['cipher_type']),
            encryption_mode=EncryptionMode(enc_meta_dict['encryption_mode']),
            key_id=enc_meta_dict['key_id'],
            nonce=bytes.fromhex(enc_meta_dict['nonce']),
            tag=bytes.fromhex(enc_meta_dict['tag']) if enc_meta_dict['tag'] else None,
            timestamp=enc_meta_dict['timestamp'],
            version=enc_meta_dict['version'],
            additional_data=bytes.fromhex(enc_meta_dict['additional_data']) if enc_meta_dict['additional_data'] else None
        )
        
        # Create encrypted block
        encrypted_block = EncryptedMemoryBlock(
            block_id=metadata_dict['block_id'],
            block_type=MemoryBlockType(metadata_dict['block_type']),
            encrypted_data=encrypted_data,
            encryption_metadata=encryption_metadata,
            original_size=metadata_dict['original_size'],
            compressed_size=metadata_dict['compressed_size'],
            compression=CompressionType(metadata_dict['compression']),
            checksum=metadata_dict['checksum'],
            created_at=metadata_dict['created_at'],
            accessed_at=metadata_dict['accessed_at'],
            modified_at=metadata_dict['modified_at'],
            metadata=metadata_dict.get('metadata')
        )
        
        return encrypted_block
    
    def _create_block_aad(self, memory_block: MemoryBlock, compression_type: CompressionType) -> bytes:
        """Create additional authenticated data for a memory block."""
        return struct.pack(
            '!QQI',
            int(memory_block.created_at * 1000000),
            int(memory_block.modified_at * 1000000),
            compression_type.value.encode('utf-8').__hash__() & 0xffffffff
        ) + memory_block.block_id.encode('utf-8')
    
    def _create_block_aad_from_encrypted(self, encrypted_block: EncryptedMemoryBlock) -> bytes:
        """Create additional authenticated data from encrypted block."""
        return struct.pack(
            '!QQI',
            int(encrypted_block.created_at * 1000000),
            int(encrypted_block.modified_at * 1000000),
            encrypted_block.compression.value.encode('utf-8').__hash__() & 0xffffffff
        ) + encrypted_block.block_id.encode('utf-8')
    
    def _update_performance_stats(self, bytes_processed: int, processing_time: float):
        """Update performance statistics."""
        with self.lock:
            self.performance_stats['operations_count'] += 1
            self.performance_stats['total_bytes_processed'] += bytes_processed
            
            # Update running average throughput (MB/s)
            throughput = bytes_processed / (processing_time * 1024 * 1024)
            count = self.performance_stats['operations_count']
            old_avg = self.performance_stats['average_throughput']
            self.performance_stats['average_throughput'] = (
                old_avg * (count - 1) + throughput
            ) / count
            
            # Update hardware acceleration usage
            self.performance_stats['hardware_acceleration_used'] = (
                self.hardware_accel.aes_ni_available or self.hardware_accel.avx2_available
            )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        with self.lock:
            stats = self.performance_stats.copy()
            stats.update({
                'hardware_info': {
                    'aes_ni_available': self.hardware_accel.aes_ni_available,
                    'avx2_available': self.hardware_accel.avx2_available,
                    'vectorization_available': self.hardware_accel.vectorization_available
                },
                'compression_algorithms': self.compression_service.available_algorithms
            })
            return stats
    
    def reset_performance_stats(self):
        """Reset performance statistics."""
        with self.lock:
            self.performance_stats = {
                'operations_count': 0,
                'total_bytes_processed': 0,
                'average_throughput': 0.0,
                'compression_ratio': 0.0,
                'hardware_acceleration_used': False
            }


# Global instance for easy access
encrypted_memory_ops = EncryptedMemoryOperations()