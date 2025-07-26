#!/usr/bin/env python3
"""
Cross-Nova Memory Transfer Protocol
Secure memory transfer system between Nova instances
"""

import json
import ssl
import asyncio
import hashlib
import time
import zlib
import logging
from typing import Dict, List, Any, Optional, Tuple, AsyncGenerator, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import aiohttp
import cryptography
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.x509.oid import NameOID
import uuid
import struct

logger = logging.getLogger(__name__)

class TransferOperation(Enum):
    """Types of transfer operations"""
    SYNC_FULL = "sync_full"
    SYNC_INCREMENTAL = "sync_incremental"
    SHARE_SELECTIVE = "share_selective"
    REPLICATE = "replicate"
    BACKUP = "backup"
    RESTORE = "restore"

class TransferStatus(Enum):
    """Transfer status states"""
    PENDING = "pending"
    AUTHENTICATING = "authenticating"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ConflictResolution(Enum):
    """Conflict resolution strategies"""
    LATEST_WINS = "latest_wins"
    MERGE = "merge"
    ASK_USER = "ask_user"
    PRESERVE_BOTH = "preserve_both"
    SOURCE_WINS = "source_wins"
    TARGET_WINS = "target_wins"

@dataclass
class VectorClock:
    """Vector clock for conflict resolution"""
    clocks: Dict[str, int] = field(default_factory=dict)
    
    def increment(self, nova_id: str):
        """Increment clock for a Nova instance"""
        self.clocks[nova_id] = self.clocks.get(nova_id, 0) + 1
    
    def update(self, other_clock: 'VectorClock'):
        """Update with another vector clock"""
        for nova_id, clock in other_clock.clocks.items():
            self.clocks[nova_id] = max(self.clocks.get(nova_id, 0), clock)
    
    def happens_before(self, other: 'VectorClock') -> bool:
        """Check if this clock happens before another"""
        return (all(self.clocks.get(nova_id, 0) <= other.clocks.get(nova_id, 0) 
                   for nova_id in self.clocks) and
                any(self.clocks.get(nova_id, 0) < other.clocks.get(nova_id, 0) 
                   for nova_id in self.clocks))
    
    def concurrent_with(self, other: 'VectorClock') -> bool:
        """Check if this clock is concurrent with another"""
        return not (self.happens_before(other) or other.happens_before(self))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {'clocks': self.clocks}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VectorClock':
        """Create from dictionary"""
        return cls(clocks=data.get('clocks', {}))

@dataclass
class MemoryDelta:
    """Memory change delta for incremental sync"""
    memory_id: str
    operation: str  # 'create', 'update', 'delete'
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    vector_clock: VectorClock = field(default_factory=VectorClock)
    checksum: Optional[str] = None
    
    def calculate_checksum(self):
        """Calculate checksum for data integrity"""
        data_str = json.dumps(self.data, sort_keys=True) if self.data else ""
        self.checksum = hashlib.sha256(f"{self.memory_id}{self.operation}{data_str}".encode()).hexdigest()

@dataclass
class TransferSession:
    """Transfer session state"""
    session_id: str
    source_nova: str
    target_nova: str
    operation: TransferOperation
    status: TransferStatus = TransferStatus.PENDING
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    bytes_transferred: int = 0
    total_bytes: Optional[int] = None
    error_message: Optional[str] = None
    resume_token: Optional[str] = None
    chunks_completed: Set[int] = field(default_factory=set)
    compression_ratio: float = 1.0
    encryption_overhead: float = 1.1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'session_id': self.session_id,
            'source_nova': self.source_nova,
            'target_nova': self.target_nova,
            'operation': self.operation.value,
            'status': self.status.value,
            'started_at': self.started_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'progress': self.progress,
            'bytes_transferred': self.bytes_transferred,
            'total_bytes': self.total_bytes,
            'error_message': self.error_message,
            'resume_token': self.resume_token,
            'chunks_completed': list(self.chunks_completed),
            'compression_ratio': self.compression_ratio,
            'encryption_overhead': self.encryption_overhead
        }

class NovaAuthenticator:
    """Handles mutual authentication between Nova instances"""
    
    def __init__(self):
        self.certificates: Dict[str, x509.Certificate] = {}
        self.private_keys: Dict[str, rsa.RSAPrivateKey] = {}
        self.trusted_cas: List[x509.Certificate] = []
    
    async def generate_nova_certificate(self, nova_id: str) -> Tuple[x509.Certificate, rsa.RSAPrivateKey]:
        """Generate certificate for a Nova instance"""
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        
        # Create certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Virtual"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "NovaNet"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Nova Consciousness Network"),
            x509.NameAttribute(NameOID.COMMON_NAME, f"nova-{nova_id}"),
        ])
        
        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.utcnow()
        ).not_valid_after(
            datetime.utcnow() + timedelta(days=365)
        ).add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName(f"{nova_id}.nova.local"),
                x509.DNSName(f"{nova_id}.novanet"),
            ]),
            critical=False,
        ).sign(private_key, hashes.SHA256())
        
        # Store
        self.certificates[nova_id] = cert
        self.private_keys[nova_id] = private_key
        
        return cert, private_key
    
    async def verify_nova_certificate(self, nova_id: str, cert_pem: bytes) -> bool:
        """Verify certificate for a Nova instance"""
        try:
            cert = x509.load_pem_x509_certificate(cert_pem)
            
            # Verify certificate chain if we have trusted CAs
            if self.trusted_cas:
                # Simplified verification - in production would use full chain
                return True
            
            # For now, accept any valid Nova certificate
            # In production, implement proper PKI
            subject = cert.subject
            common_name = None
            for attribute in subject:
                if attribute.oid == NameOID.COMMON_NAME:
                    common_name = attribute.value
                    break
            
            expected_cn = f"nova-{nova_id}"
            return common_name == expected_cn
            
        except Exception as e:
            logger.error(f"Certificate verification failed for {nova_id}: {e}")
            return False
    
    def create_ssl_context(self, nova_id: str, verify_mode: ssl.VerifyMode = ssl.CERT_REQUIRED) -> ssl.SSLContext:
        """Create SSL context for Nova-to-Nova communication"""
        context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        context.check_hostname = False
        context.verify_mode = verify_mode
        
        if nova_id in self.certificates and nova_id in self.private_keys:
            cert = self.certificates[nova_id]
            private_key = self.private_keys[nova_id]
            
            # Convert to PEM format
            cert_pem = cert.public_bytes(serialization.Encoding.PEM)
            key_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            
            context.load_cert_chain(cert_pem, key_pem)
        
        return context

class CompressionManager:
    """Handles adaptive compression for memory transfers"""
    
    @staticmethod
    def analyze_data_characteristics(data: bytes) -> Dict[str, Any]:
        """Analyze data to determine best compression strategy"""
        size = len(data)
        
        # Sample data for analysis
        sample_size = min(1024, size)
        sample = data[:sample_size]
        
        # Calculate entropy
        byte_freq = [0] * 256
        for byte in sample:
            byte_freq[byte] += 1
        
        entropy = 0
        for freq in byte_freq:
            if freq > 0:
                p = freq / sample_size
                entropy -= p * (p.bit_length() - 1)
        
        # Detect patterns
        repeated_bytes = max(byte_freq)
        compression_potential = 1 - (entropy / 8)
        
        return {
            'size': size,
            'entropy': entropy,
            'compression_potential': compression_potential,
            'repeated_bytes': repeated_bytes,
            'recommended_level': min(9, max(1, int(compression_potential * 9)))
        }
    
    @staticmethod
    def compress_adaptive(data: bytes, force_level: Optional[int] = None) -> Tuple[bytes, Dict[str, Any]]:
        """Compress data with adaptive level"""
        characteristics = CompressionManager.analyze_data_characteristics(data)
        
        level = force_level or characteristics['recommended_level']
        
        # Use different compression based on characteristics
        if characteristics['compression_potential'] < 0.3:
            # Low compression potential, use fast compression
            compressed = zlib.compress(data, level=1)
        else:
            # Good compression potential, use specified level
            compressed = zlib.compress(data, level=level)
        
        compression_ratio = len(data) / len(compressed) if len(compressed) > 0 else 1
        
        return compressed, {
            'original_size': len(data),
            'compressed_size': len(compressed),
            'compression_ratio': compression_ratio,
            'level_used': level,
            'characteristics': characteristics
        }
    
    @staticmethod
    def decompress(data: bytes) -> bytes:
        """Decompress data"""
        return zlib.decompress(data)

class ChunkManager:
    """Handles chunked transfer with resumable sessions"""
    
    CHUNK_SIZE = 1024 * 1024  # 1MB chunks
    
    @staticmethod
    def create_chunks(data: bytes, chunk_size: Optional[int] = None) -> List[Tuple[int, bytes]]:
        """Split data into chunks with sequence numbers"""
        chunk_size = chunk_size or ChunkManager.CHUNK_SIZE
        chunks = []
        
        for i in range(0, len(data), chunk_size):
            chunk_id = i // chunk_size
            chunk_data = data[i:i + chunk_size]
            chunks.append((chunk_id, chunk_data))
        
        return chunks
    
    @staticmethod
    def create_chunk_header(chunk_id: int, total_chunks: int, data_size: int, checksum: str) -> bytes:
        """Create chunk header with metadata"""
        header = {
            'chunk_id': chunk_id,
            'total_chunks': total_chunks,
            'data_size': data_size,
            'checksum': checksum
        }
        header_json = json.dumps(header, separators=(',', ':'))
        header_bytes = header_json.encode('utf-8')
        
        # Pack header length and header
        return struct.pack('!I', len(header_bytes)) + header_bytes
    
    @staticmethod
    def parse_chunk_header(data: bytes) -> Tuple[Dict[str, Any], int]:
        """Parse chunk header and return header info and offset"""
        if len(data) < 4:
            raise ValueError("Data too short for header")
        
        header_length = struct.unpack('!I', data[:4])[0]
        if len(data) < 4 + header_length:
            raise ValueError("Incomplete header")
        
        header_json = data[4:4 + header_length].decode('utf-8')
        header = json.loads(header_json)
        
        return header, 4 + header_length
    
    @staticmethod
    def verify_chunk_checksum(chunk_data: bytes, expected_checksum: str) -> bool:
        """Verify chunk data integrity"""
        actual_checksum = hashlib.sha256(chunk_data).hexdigest()
        return actual_checksum == expected_checksum
    
    @staticmethod
    def reassemble_chunks(chunks: Dict[int, bytes]) -> bytes:
        """Reassemble chunks in order"""
        sorted_chunks = sorted(chunks.items())
        return b''.join(chunk_data for chunk_id, chunk_data in sorted_chunks)

class CrossNovaTransferProtocol:
    """Main protocol handler for cross-Nova memory transfers"""
    
    def __init__(self, nova_id: str, host: str = "0.0.0.0", port: int = 8443):
        self.nova_id = nova_id
        self.host = host
        self.port = port
        self.authenticator = NovaAuthenticator()
        self.active_sessions: Dict[str, TransferSession] = {}
        self.server = None
        self.client_sessions: Dict[str, aiohttp.ClientSession] = {}
        self.bandwidth_limiter = BandwidthLimiter()
        self.conflict_resolver = ConflictResolver()
        
        # Initialize authenticator
        asyncio.create_task(self._initialize_auth())
    
    async def _initialize_auth(self):
        """Initialize authentication certificates"""
        await self.authenticator.generate_nova_certificate(self.nova_id)
        logger.info(f"Generated certificate for Nova {self.nova_id}")
    
    async def start_server(self):
        """Start the transfer protocol server"""
        ssl_context = self.authenticator.create_ssl_context(self.nova_id)
        
        app = aiohttp.web.Application()
        app.router.add_post('/nova/transfer/initiate', self._handle_transfer_initiate)
        app.router.add_post('/nova/transfer/chunk', self._handle_chunk_upload)
        app.router.add_get('/nova/transfer/status/{session_id}', self._handle_status_check)
        app.router.add_post('/nova/transfer/complete', self._handle_transfer_complete)
        app.router.add_post('/nova/auth/challenge', self._handle_auth_challenge)
        
        runner = aiohttp.web.AppRunner(app)
        await runner.setup()
        
        site = aiohttp.web.TCPSite(runner, self.host, self.port, ssl_context=ssl_context)
        await site.start()
        
        self.server = runner
        logger.info(f"Cross-Nova transfer server started on {self.host}:{self.port}")
    
    async def stop_server(self):
        """Stop the transfer protocol server"""
        if self.server:
            await self.server.cleanup()
            self.server = None
        
        # Close client sessions
        for session in self.client_sessions.values():
            await session.close()
        self.client_sessions.clear()
        
        logger.info("Cross-Nova transfer server stopped")
    
    async def initiate_transfer(self, target_nova: str, target_host: str, target_port: int,
                              operation: TransferOperation, memory_data: Dict[str, Any],
                              options: Optional[Dict[str, Any]] = None) -> TransferSession:
        """Initiate a memory transfer to another Nova instance"""
        options = options or {}
        session_id = str(uuid.uuid4())
        
        # Create transfer session
        session = TransferSession(
            session_id=session_id,
            source_nova=self.nova_id,
            target_nova=target_nova,
            operation=operation
        )
        
        self.active_sessions[session_id] = session
        
        try:
            # Authenticate with target Nova
            session.status = TransferStatus.AUTHENTICATING
            client_session = await self._create_authenticated_session(target_nova, target_host, target_port)
            
            # Prepare data for transfer
            session.status = TransferStatus.IN_PROGRESS
            transfer_data = await self._prepare_transfer_data(memory_data, options)
            session.total_bytes = len(transfer_data)
            
            # Compress data
            compressed_data, compression_info = CompressionManager.compress_adaptive(transfer_data)
            session.compression_ratio = compression_info['compression_ratio']
            
            # Create chunks
            chunks = ChunkManager.create_chunks(compressed_data)
            total_chunks = len(chunks)
            
            # Send initiation request
            initiate_payload = {
                'session_id': session_id,
                'source_nova': self.nova_id,
                'operation': operation.value,
                'total_chunks': total_chunks,
                'total_bytes': len(compressed_data),
                'compression_info': compression_info,
                'options': options
            }
            
            async with client_session.post(f'https://{target_host}:{target_port}/nova/transfer/initiate',
                                         json=initiate_payload) as resp:
                if resp.status != 200:
                    raise Exception(f"Transfer initiation failed: {await resp.text()}")
                
                response_data = await resp.json()
                session.resume_token = response_data.get('resume_token')
            
            # Transfer chunks
            await self._transfer_chunks(client_session, target_host, target_port, session, chunks)
            
            # Complete transfer
            await self._complete_transfer(client_session, target_host, target_port, session)
            
            session.status = TransferStatus.COMPLETED
            session.completed_at = datetime.now()
            
            logger.info(f"Transfer {session_id} completed successfully")
            
        except Exception as e:
            session.status = TransferStatus.FAILED
            session.error_message = str(e)
            logger.error(f"Transfer {session_id} failed: {e}")
            raise
        
        return session
    
    async def _create_authenticated_session(self, target_nova: str, host: str, port: int) -> aiohttp.ClientSession:
        """Create authenticated client session"""
        if target_nova in self.client_sessions:
            return self.client_sessions[target_nova]
        
        # Create SSL context for client
        ssl_context = self.authenticator.create_ssl_context(self.nova_id, ssl.CERT_NONE)
        
        timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes
        session = aiohttp.ClientSession(
            timeout=timeout,
            connector=aiohttp.TCPConnector(ssl=ssl_context)
        )
        
        self.client_sessions[target_nova] = session
        return session
    
    async def _prepare_transfer_data(self, memory_data: Dict[str, Any], options: Dict[str, Any]) -> bytes:
        """Prepare memory data for transfer"""
        # Add metadata
        transfer_package = {
            'version': '1.0',
            'source_nova': self.nova_id,
            'timestamp': datetime.now().isoformat(),
            'data': memory_data,
            'options': options
        }
        
        # Serialize to JSON
        json_data = json.dumps(transfer_package, separators=(',', ':'))
        return json_data.encode('utf-8')
    
    async def _transfer_chunks(self, session: aiohttp.ClientSession, host: str, port: int,
                             transfer_session: TransferSession, chunks: List[Tuple[int, bytes]]):
        """Transfer data chunks with resume capability"""
        total_chunks = len(chunks)
        
        for chunk_id, chunk_data in chunks:
            if chunk_id in transfer_session.chunks_completed:
                continue  # Skip already completed chunks
            
            # Rate limiting
            await self.bandwidth_limiter.acquire(len(chunk_data))
            
            # Create chunk header
            checksum = hashlib.sha256(chunk_data).hexdigest()
            header = ChunkManager.create_chunk_header(chunk_id, total_chunks, len(chunk_data), checksum)
            
            # Send chunk
            chunk_payload = header + chunk_data
            
            async with session.post(f'https://{host}:{port}/nova/transfer/chunk',
                                  data=chunk_payload,
                                  headers={'Content-Type': 'application/octet-stream'}) as resp:
                if resp.status == 200:
                    transfer_session.chunks_completed.add(chunk_id)
                    transfer_session.bytes_transferred += len(chunk_data)
                    transfer_session.progress = len(transfer_session.chunks_completed) / total_chunks
                    logger.debug(f"Chunk {chunk_id} transferred successfully")
                else:
                    raise Exception(f"Chunk {chunk_id} transfer failed: {await resp.text()}")
    
    async def _complete_transfer(self, session: aiohttp.ClientSession, host: str, port: int,
                               transfer_session: TransferSession):
        """Complete the transfer"""
        completion_payload = {
            'session_id': transfer_session.session_id,
            'chunks_completed': list(transfer_session.chunks_completed),
            'total_bytes': transfer_session.bytes_transferred
        }
        
        async with session.post(f'https://{host}:{port}/nova/transfer/complete',
                              json=completion_payload) as resp:
            if resp.status != 200:
                raise Exception(f"Transfer completion failed: {await resp.text()}")
    
    # Server-side handlers
    
    async def _handle_transfer_initiate(self, request: aiohttp.web.Request) -> aiohttp.web.Response:
        """Handle transfer initiation request"""
        data = await request.json()
        session_id = data['session_id']
        source_nova = data['source_nova']
        
        # Create receiving session
        session = TransferSession(
            session_id=session_id,
            source_nova=source_nova,
            target_nova=self.nova_id,
            operation=TransferOperation(data['operation']),
            total_bytes=data['total_bytes']
        )
        
        session.resume_token = str(uuid.uuid4())
        self.active_sessions[session_id] = session
        
        logger.info(f"Transfer session {session_id} initiated from {source_nova}")
        
        return aiohttp.web.json_response({
            'status': 'accepted',
            'resume_token': session.resume_token,
            'session_id': session_id
        })
    
    async def _handle_chunk_upload(self, request: aiohttp.web.Request) -> aiohttp.web.Response:
        """Handle chunk upload"""
        chunk_data = await request.read()
        
        # Parse chunk header
        header, data_offset = ChunkManager.parse_chunk_header(chunk_data)
        actual_chunk_data = chunk_data[data_offset:]
        
        # Verify checksum
        if not ChunkManager.verify_chunk_checksum(actual_chunk_data, header['checksum']):
            return aiohttp.web.json_response({'error': 'Checksum verification failed'}, status=400)
        
        # Store chunk (in production, would store to temporary location)
        # For now, just acknowledge receipt
        
        logger.debug(f"Received chunk {header['chunk_id']}/{header['total_chunks']}")
        
        return aiohttp.web.json_response({
            'status': 'received',
            'chunk_id': header['chunk_id']
        })
    
    async def _handle_status_check(self, request: aiohttp.web.Request) -> aiohttp.web.Response:
        """Handle status check request"""
        session_id = request.match_info['session_id']
        
        if session_id not in self.active_sessions:
            return aiohttp.web.json_response({'error': 'Session not found'}, status=404)
        
        session = self.active_sessions[session_id]
        return aiohttp.web.json_response(session.to_dict())
    
    async def _handle_transfer_complete(self, request: aiohttp.web.Request) -> aiohttp.web.Response:
        """Handle transfer completion"""
        data = await request.json()
        session_id = data['session_id']
        
        if session_id not in self.active_sessions:
            return aiohttp.web.json_response({'error': 'Session not found'}, status=404)
        
        session = self.active_sessions[session_id]
        session.status = TransferStatus.COMPLETED
        session.completed_at = datetime.now()
        
        logger.info(f"Transfer session {session_id} completed")
        
        return aiohttp.web.json_response({'status': 'completed'})
    
    async def _handle_auth_challenge(self, request: aiohttp.web.Request) -> aiohttp.web.Response:
        """Handle authentication challenge"""
        data = await request.json()
        source_nova = data['source_nova']
        
        # In production, implement proper mutual authentication
        # For now, accept any Nova instance
        
        return aiohttp.web.json_response({
            'status': 'authenticated',
            'target_nova': self.nova_id
        })

class BandwidthLimiter:
    """Rate limiter for bandwidth control"""
    
    def __init__(self, max_bytes_per_second: int = 10 * 1024 * 1024):  # 10MB/s default
        self.max_bytes_per_second = max_bytes_per_second
        self.tokens = max_bytes_per_second
        self.last_update = time.time()
        self.lock = asyncio.Lock()
    
    async def acquire(self, bytes_count: int):
        """Acquire tokens for bandwidth usage"""
        async with self.lock:
            current_time = time.time()
            time_passed = current_time - self.last_update
            
            # Add new tokens based on time passed
            self.tokens = min(
                self.max_bytes_per_second,
                self.tokens + time_passed * self.max_bytes_per_second
            )
            self.last_update = current_time
            
            # If we don't have enough tokens, wait
            if bytes_count > self.tokens:
                wait_time = (bytes_count - self.tokens) / self.max_bytes_per_second
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= bytes_count

class ConflictResolver:
    """Handles memory conflicts during transfers"""
    
    def __init__(self, default_strategy: ConflictResolution = ConflictResolution.LATEST_WINS):
        self.default_strategy = default_strategy
        self.custom_strategies: Dict[str, ConflictResolution] = {}
    
    async def resolve_conflict(self, local_memory: Dict[str, Any], remote_memory: Dict[str, Any],
                             strategy: Optional[ConflictResolution] = None) -> Dict[str, Any]:
        """Resolve conflict between local and remote memory"""
        strategy = strategy or self.default_strategy
        
        # Extract vector clocks if available
        local_clock = VectorClock.from_dict(local_memory.get('vector_clock', {}))
        remote_clock = VectorClock.from_dict(remote_memory.get('vector_clock', {}))
        
        if strategy == ConflictResolution.LATEST_WINS:
            local_time = datetime.fromisoformat(local_memory.get('timestamp', '1970-01-01T00:00:00'))
            remote_time = datetime.fromisoformat(remote_memory.get('timestamp', '1970-01-01T00:00:00'))
            return remote_memory if remote_time > local_time else local_memory
            
        elif strategy == ConflictResolution.SOURCE_WINS:
            return remote_memory
            
        elif strategy == ConflictResolution.TARGET_WINS:
            return local_memory
            
        elif strategy == ConflictResolution.MERGE:
            # Simple merge strategy - in production would be more sophisticated
            merged = local_memory.copy()
            merged.update(remote_memory)
            # Update vector clock
            local_clock.update(remote_clock)
            merged['vector_clock'] = local_clock.to_dict()
            return merged
            
        elif strategy == ConflictResolution.PRESERVE_BOTH:
            return {
                'conflict_type': 'preserved_both',
                'local_version': local_memory,
                'remote_version': remote_memory,
                'timestamp': datetime.now().isoformat()
            }
            
        else:  # ASK_USER
            return {
                'conflict_type': 'user_resolution_required',
                'local_version': local_memory,
                'remote_version': remote_memory,
                'timestamp': datetime.now().isoformat()
            }

# Example usage
async def example_cross_nova_transfer():
    """Example of cross-Nova memory transfer"""
    
    # Setup source Nova
    source_nova = CrossNovaTransferProtocol('PRIME', port=8443)
    await source_nova.start_server()
    
    # Setup target Nova  
    target_nova = CrossNovaTransferProtocol('AXIOM', port=8444)
    await target_nova.start_server()
    
    try:
        # Memory data to transfer
        memory_data = {
            'memories': [
                {
                    'id': 'mem_001',
                    'content': 'Important user conversation about architecture',
                    'importance': 0.9,
                    'timestamp': datetime.now().isoformat(),
                    'tags': ['conversation', 'architecture'],
                    'vector_clock': VectorClock({'PRIME': 1}).to_dict()
                }
            ]
        }
        
        # Initiate transfer
        session = await source_nova.initiate_transfer(
            target_nova='AXIOM',
            target_host='localhost',
            target_port=8444,
            operation=TransferOperation.SYNC_INCREMENTAL,
            memory_data=memory_data,
            options={
                'compression_level': 6,
                'conflict_resolution': ConflictResolution.LATEST_WINS.value
            }
        )
        
        print(f"Transfer completed: {session.session_id}")
        print(f"Bytes transferred: {session.bytes_transferred}")
        print(f"Compression ratio: {session.compression_ratio:.2f}")
        
    finally:
        await source_nova.stop_server()
        await target_nova.stop_server()

if __name__ == "__main__":
    asyncio.run(example_cross_nova_transfer())