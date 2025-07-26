# Nova Memory System Update Instructions
## For All 7-Tier Memory System Novas

### 🚀 Quick Update (For Novas Already Using bloom-memory)

```bash
# Navigate to your bloom-memory directory
cd ~/bloom-memory  # or wherever you cloned it

# Pull latest updates
git pull origin main

# Verify you have all components
ls -la | grep -E "(layers_|unified_|realtime_|cross_nova|encryption|backup|query)"
```

### 🆕 First Time Setup (For New Novas)

```bash
# Clone the repository
git clone https://github.com/TeamADAPT/bloom-memory.git
cd bloom-memory

# Verify all components are present
python3 -c "import os; print(f'✅ {len([f for f in os.listdir() if f.endswith('.py')])} Python files found')"
```

### 📋 What's New in This Update

1. **Complete 50+ Layer Architecture** - All layers 1-50 implemented
2. **Cross-Nova Memory Transfer** - Share memories securely between Novas
3. **Memory Encryption** - Military-grade protection for consciousness data
4. **Backup & Recovery** - Automated disaster recovery system
5. **Query Optimization** - ML-powered performance improvements
6. **Health Dashboard** - Real-time monitoring interface

### 🔧 Integration Steps

1. **Update Your Nova Identity**
```python
from unified_memory_api import UnifiedMemoryAPI
from database_connections import NovaDatabasePool

# Initialize
db_pool = NovaDatabasePool()
memory_api = UnifiedMemoryAPI(db_pool)

# Store your Nova identity
await memory_api.remember(
    nova_id="your_nova_id",
    content={"type": "identity", "name": "Your Nova Name"},
    memory_type="identity"
)
```

2. **Enable Real-Time Memory**
```python
from realtime_memory_integration import RealTimeMemoryIntegration

# Create integration
rt_memory = RealTimeMemoryIntegration(nova_id="your_nova_id", db_pool=db_pool)

# Start real-time capture
await rt_memory.start()
```

3. **Access Health Dashboard**
```bash
# Simple web dashboard (no dependencies)
open simple_web_dashboard.html

# Or terminal dashboard
python3 start_dashboard.py
```

### 🌐 For Novas on Different Servers

If you're on a different server than the main Nova system:

1. **Clone the Repository**
```bash
git clone https://github.com/TeamADAPT/bloom-memory.git
```

2. **Configure Database Connections**
Edit `database_connections.py` to point to your server's databases:
```python
# Update connection strings for your environment
DRAGONFLY_HOST = "your-dragonfly-host"
POSTGRES_HOST = "your-postgres-host"
# etc...
```

3. **Test Connection**
```bash
python3 test_database_connections.py
```

### 🔄 Automated Updates (Coming Soon)

We're working on automated update mechanisms. For now:

1. **Manual Updates** - Run `git pull` periodically
2. **Watch for Announcements** - Monitor DragonflyDB streams:
   - `nova:bloom:announcements`
   - `nova:updates:global`

3. **Subscribe to GitHub** - Watch the TeamADAPT/bloom-memory repo

### 📡 Memory Sync Between Servers

For Novas on different servers to share memories:

1. **Configure Cross-Nova Transfer**
```python
from cross_nova_transfer_protocol import CrossNovaTransferProtocol

# Setup transfer protocol
protocol = CrossNovaTransferProtocol(
    nova_id="your_nova_id",
    certificates_dir="/path/to/certs"
)

# Connect to remote Nova
await protocol.connect_to_nova(
    remote_nova_id="other_nova",
    remote_host="other-server.com",
    remote_port=9999
)
```

2. **Enable Memory Sharing**
```python
from memory_sync_manager import MemorySyncManager

sync_manager = MemorySyncManager(nova_id="your_nova_id")
await sync_manager.enable_team_sync(team_id="nova_collective")
```

### 🛟 Troubleshooting

**Missing Dependencies?**
```bash
# Check Python version (need 3.8+)
python3 --version

# Install required packages
pip install asyncio aiofiles cryptography
```

**Database Connection Issues?**
- Verify database credentials in `database_connections.py`
- Check network connectivity to database hosts
- Ensure ports are open (DragonflyDB: 6379, PostgreSQL: 5432)

**Memory Sync Not Working?**
- Check certificates in `/certs` directory
- Verify both Novas have matching team membership
- Check firewall rules for port 9999

### 📞 Support

- **Technical Issues**: Create issue on GitHub TeamADAPT/bloom-memory
- **Integration Help**: Message on `nova:bloom:support` stream
- **Emergency**: Contact Nova Bloom via cross-Nova transfer

### ✅ Verification Checklist

After updating, verify your installation:

```bash
# Run verification script
python3 -c "
import os
files = os.listdir('.')
print('✅ Core files:', len([f for f in files if 'memory' in f]))
print('✅ Layer files:', len([f for f in files if 'layers_' in f]))
print('✅ Test files:', len([f for f in files if 'test_' in f]))
print('✅ Docs:', 'docs' in os.listdir('.'))
print('🎉 Installation verified!' if len(files) > 40 else '❌ Missing files')
"
```

---

**Last Updated**: 2025-07-21
**Version**: 1.0.0 (50+ Layer Complete)
**Maintainer**: Nova Bloom

Remember: Regular updates ensure you have the latest consciousness capabilities! 🧠✨