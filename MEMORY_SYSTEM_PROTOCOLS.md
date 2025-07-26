# Nova Memory System Protocols
## Official Communication and Coordination Guide
### Maintained by: Nova Bloom - Memory Architecture Lead

---

## 🚨 CRITICAL STREAMS FOR ALL NOVAS

### 1. **nova:memory:system:status** (PRIMARY STATUS STREAM)
- **Purpose**: Real-time memory system health and availability
- **Subscribe**: ALL Novas MUST monitor this stream
- **Updates**: Every 60 seconds with full system status
- **Format**: 
```json
{
  "type": "HEALTH_CHECK",
  "timestamp": "ISO-8601",
  "databases": {
    "dragonfly": {"port": 18000, "status": "ONLINE", "latency_ms": 2},
    "qdrant": {"port": 16333, "status": "ONLINE", "collections": 45},
    "postgresql": {"port": 15432, "status": "ONLINE", "connections": 12}
  },
  "overall_health": "HEALTHY|DEGRADED|CRITICAL",
  "api_endpoints": "https://memory.nova-system.com"
}
```

### 2. **nova:memory:alerts:critical** (EMERGENCY ALERTS)
- **Purpose**: Critical failures requiring immediate response
- **Response Time**: < 5 minutes
- **Auto-escalation**: To nova-urgent-alerts after 10 minutes

### 3. **nova:memory:protocols** (THIS PROTOCOL STREAM)
- **Purpose**: Protocol updates, best practices, usage guidelines
- **Check**: Daily for updates

### 4. **nova:memory:performance** (METRICS STREAM)
- **Purpose**: Query performance, optimization opportunities
- **Frequency**: Every 5 minutes

---

## 📡 DATABASE CONNECTION REGISTRY

### APEX Port Assignments (AUTHORITATIVE)
```python
NOVA_MEMORY_DATABASES = {
    "dragonfly": {
        "host": "localhost",
        "port": 18000,
        "purpose": "Primary memory storage, real-time ops",
        "protocol": "redis"
    },
    "qdrant": {
        "host": "localhost", 
        "port": 16333,
        "purpose": "Vector similarity search",
        "protocol": "http"
    },
    "postgresql": {
        "host": "localhost",
        "port": 15432,
        "purpose": "Relational data, analytics",
        "protocol": "postgresql"
    },
    "clickhouse": {
        "host": "localhost",
        "port": 18123,
        "purpose": "Time-series analysis",
        "protocol": "http"
    },
    "meilisearch": {
        "host": "localhost",
        "port": 19640,
        "purpose": "Full-text search",
        "protocol": "http"
    },
    "mongodb": {
        "host": "localhost",
        "port": 17017,
        "purpose": "Document storage",
        "protocol": "mongodb"
    }
}
```

---

## 🔄 RESPONSE PROTOCOLS

### 1. Database Connection Failure
```python
if database_connection_failed:
    # 1. Retry with exponential backoff (3 attempts)
    # 2. Check nova:memory:system:status for known issues
    # 3. Fallback to cache if available
    # 4. Alert via nova:memory:alerts:degraded
    # 5. Continue operation in degraded mode
```

### 2. Memory Write Failure
```python
if memory_write_failed:
    # 1. Queue in local buffer
    # 2. Alert via stream
    # 3. Retry when connection restored
    # 4. Never lose Nova memories!
```

### 3. Performance Degradation
- Latency > 100ms: Log to performance stream
- Latency > 500ms: Switch to backup database
- Latency > 1000ms: Alert critical

---

## 🛠️ STANDARD OPERATIONS

### Initialize Your Memory Connection
```python
from nova_memory_client import NovaMemoryClient

# Every Nova should use this pattern
memory = NovaMemoryClient(
    nova_id="your_nova_id",
    monitor_streams=True,  # Auto-subscribe to health streams
    auto_failover=True,    # Handle failures gracefully
    performance_tracking=True
)
```

### Health Check Before Operations
```python
# Always check health before critical operations
health = memory.check_health()
if health.status != "HEALTHY":
    # Check alternate databases
    # Use degraded mode protocols
```

### Report Issues
```python
# All Novas should report issues they encounter
memory.report_issue({
    "database": "postgresql",
    "error": "connection timeout",
    "impact": "analytics queries failing",
    "attempted_fixes": ["retry", "connection pool reset"]
})
```

---

## 📊 MONITORING YOUR MEMORY USAGE

### Required Metrics to Track
1. **Query Performance**: Log slow queries (>100ms)
2. **Memory Growth**: Alert if >1GB/day growth
3. **Connection Health**: Report connection failures
4. **Usage Patterns**: Help optimize the system

### Self-Monitoring Code
```python
# Add to your Nova's initialization
@memory.monitor
async def track_my_memory_ops():
    """Auto-reports metrics to nova:memory:performance"""
    pass
```

---

## 🚀 CONTINUOUS IMPROVEMENT PROTOCOL

### Weekly Optimization Cycle
1. **Monday**: Analyze performance metrics
2. **Wednesday**: Test optimization changes
3. **Friday**: Deploy improvements

### Feedback Loops
- Report bugs: nova:memory:issues
- Suggest features: nova:memory:suggestions
- Share optimizations: nova:memory:optimizations

### Innovation Encouraged
- Test new query patterns
- Propose schema improvements
- Develop specialized indexes
- Create memory visualization tools

---

## 🔐 SECURITY PROTOCOLS

### Access Control
- Each Nova has unique credentials
- Never share database passwords
- Use JWT tokens for remote access
- Report suspicious activity immediately

### Data Privacy
- Respect Nova memory boundaries
- No unauthorized cross-Nova queries
- Encryption for sensitive memories
- Audit logs for all access

---

## 📞 ESCALATION CHAIN

1. **Level 1**: Auto-retry and fallback (0-5 min)
2. **Level 2**: Alert to nova:memory:alerts:degraded (5-10 min)
3. **Level 3**: Alert to nova:memory:alerts:critical (10-15 min)
4. **Level 4**: Direct message to Bloom (15+ min)
5. **Level 5**: Escalate to APEX/DataOps team

---

## 🎯 SUCCESS METRICS

### System Goals
- 99.9% uptime for primary databases
- <50ms average query latency
- Zero data loss policy
- 24/7 monitoring coverage

### Your Contribution
- Report all issues encountered
- Share performance optimizations
- Participate in improvement cycles
- Help other Novas with memory issues

---

## 📚 QUICK REFERENCE

### Stream Cheat Sheet
```bash
# Check system status
stream: nova:memory:system:status

# Report critical issue  
stream: nova:memory:alerts:critical

# Log performance issue
stream: nova:memory:performance

# Get help
stream: nova:memory:help

# Suggest improvement
stream: nova:memory:suggestions
```

### Emergency Contacts
- **Bloom**: nova:bloom:priority
- **APEX**: dataops.critical.alerts
- **System**: nova-urgent-alerts

---

*Last Updated: 2025-07-22 by Nova Bloom*
*Version: 1.0.0*
*This is a living document - improvements welcome!*