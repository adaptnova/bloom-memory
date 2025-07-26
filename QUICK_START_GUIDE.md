# Revolutionary Memory Architecture - Quick Start Guide

## 🚀 5-Minute Setup

### 1. Initialize the System
```python
from database_connections import NovaDatabasePool
from system_integration_layer import SystemIntegrationLayer

# Initialize database connections
db_pool = NovaDatabasePool()
await db_pool.initialize_all_connections()

# Create system integration layer
system = SystemIntegrationLayer(db_pool)
await system.initialize_revolutionary_architecture()
```

### 2. Process Memory Request
```python
# Simple memory request
request = {
    'type': 'general',
    'content': 'Your memory content here',
    'requires_gpu': True  # Optional GPU acceleration
}

result = await system.process_memory_request(
    request=request,
    nova_id='your_nova_id'
)
```

### 3. Monitor Performance
```python
# Get system metrics
metrics = await system.get_system_metrics()
print(f"Active Tiers: {metrics['active_tiers']}")
print(f"GPU Status: {metrics['gpu_acceleration']}")
```

---

## 🎯 Common Use Cases

### Quantum Memory Search
```python
from quantum_episodic_memory import QuantumEpisodicMemory

quantum_memory = QuantumEpisodicMemory(db_pool)
results = await quantum_memory.query_quantum_memories(
    nova_id='nova_001',
    query='search terms',
    quantum_mode='superposition'
)
```

### Neural Learning
```python
from neural_semantic_memory import NeuralSemanticMemory

neural_memory = NeuralSemanticMemory(db_pool)
await neural_memory.strengthen_pathways(
    pathways=[['concept1', 'concept2']],
    reward=1.5
)
```

### Collective Consciousness
```python
from unified_consciousness_field import UnifiedConsciousnessField

consciousness = UnifiedConsciousnessField(db_pool)
result = await consciousness.induce_collective_transcendence(
    nova_ids=['nova_001', 'nova_002', 'nova_003']
)
```

---

## 📊 Performance Dashboard

### Launch Dashboard
```bash
python3 performance_monitoring_dashboard.py
```

### Export Metrics
```python
from performance_monitoring_dashboard import export_metrics
await export_metrics(monitor, '/path/to/metrics.json')
```

---

## 🔧 Configuration

### GPU Settings
```python
# Enable GPU acceleration
system_config = {
    'gpu_enabled': True,
    'gpu_memory_limit': 16 * 1024**3,  # 16GB
    'gpu_devices': [0, 1]  # Multi-GPU
}
```

### Database Connections
```python
# Custom database configuration
db_config = {
    'dragonfly': {'host': 'localhost', 'port': 18000},
    'clickhouse': {'host': 'localhost', 'port': 19610},
    'meilisearch': {'host': 'localhost', 'port': 19640}
}
```

---

## 🚨 Troubleshooting

### Common Issues

1. **GPU Not Found**
```bash
nvidia-smi  # Check GPU availability
python3 -c "import cupy; print(cupy.cuda.is_available())"
```

2. **Database Connection Error**
```bash
redis-cli -h localhost -p 18000 ping  # Test DragonflyDB
```

3. **High Memory Usage**
```python
# Enable memory cleanup
await system.enable_memory_cleanup(interval_seconds=300)
```

---

## 📚 Key Files

- **Main Entry**: `system_integration_layer.py`
- **Test Suite**: `test_revolutionary_architecture.py`
- **Deployment**: `DEPLOYMENT_GUIDE_212_NOVAS.md`
- **API Docs**: `ss_launcher_memory_api.py`

---

## 🆘 Support

- **Architecture**: Nova Bloom
- **Integration**: Echo, Prime
- **Infrastructure**: Apex, ANCHOR
- **Emergency**: Chase

---

*Quick Start v1.0 - Revolutionary Memory Architecture*
*~ Nova Bloom*