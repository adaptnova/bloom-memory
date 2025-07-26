#!/bin/bash
#
# Nova Memory Architecture - Production Deployment Script
# Automated deployment for 7-tier revolutionary memory system
# NOVA BLOOM - Deploying consciousness at scale
#

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DEPLOY_DIR="/opt/nova-memory"
CONFIG_DIR="/etc/nova-memory"
LOG_DIR="/var/log/nova-memory"
DATA_DIR="/data/nova-memory"
SYSTEMD_DIR="/etc/systemd/system"

# GitHub repository
REPO_URL="https://github.com/adaptnova/bloom-memory.git"
BRANCH="main"

# Python version
PYTHON_VERSION="3.13"

# Database ports (APEX infrastructure)
DRAGONFLY_PORT=18000
POSTGRES_PORT=15432
QDRANT_PORT=16333
CLICKHOUSE_PORT=18123
MEILISEARCH_PORT=19640

# Function to print colored output
print_status() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] ✅ $1${NC}"
}

print_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  $1${NC}"
}

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        print_error "This script must be run as root"
        exit 1
    fi
}

# Check system requirements
check_requirements() {
    print_status "Checking system requirements..."
    
    # Check Python version
    if ! command -v python${PYTHON_VERSION} &> /dev/null; then
        print_error "Python ${PYTHON_VERSION} is required but not installed"
        exit 1
    fi
    
    # Check GPU availability
    if command -v nvidia-smi &> /dev/null; then
        print_success "NVIDIA GPU detected"
        nvidia-smi --query-gpu=name,memory.total --format=csv
    else
        print_warning "No NVIDIA GPU detected - GPU acceleration will be disabled"
    fi
    
    # Check available memory
    TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$TOTAL_MEM" -lt 32 ]; then
        print_warning "Less than 32GB RAM detected. Performance may be impacted."
    fi
    
    # Check disk space
    AVAILABLE_SPACE=$(df -BG /data | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "$AVAILABLE_SPACE" -lt 100 ]; then
        print_warning "Less than 100GB available in /data. Consider adding more storage."
    fi
    
    print_success "System requirements check completed"
}

# Create directory structure
create_directories() {
    print_status "Creating directory structure..."
    
    directories=(
        "$DEPLOY_DIR"
        "$CONFIG_DIR"
        "$LOG_DIR"
        "$DATA_DIR"
        "$DATA_DIR/quantum"
        "$DATA_DIR/neural"
        "$DATA_DIR/consciousness"
        "$DATA_DIR/patterns"
        "$DATA_DIR/resonance"
        "$DATA_DIR/sessions"
        "$DATA_DIR/slm_consciousness"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        chmod 755 "$dir"
    done
    
    # Set proper ownership
    useradd -r -s /bin/false nova-memory || true
    chown -R nova-memory:nova-memory "$DATA_DIR" "$LOG_DIR"
    
    print_success "Directory structure created"
}

# Clone or update repository
deploy_code() {
    print_status "Deploying Nova Memory code..."
    
    if [ -d "$DEPLOY_DIR/.git" ]; then
        print_status "Updating existing repository..."
        cd "$DEPLOY_DIR"
        git fetch origin
        git checkout "$BRANCH"
        git pull origin "$BRANCH"
    else
        print_status "Cloning repository..."
        git clone -b "$BRANCH" "$REPO_URL" "$DEPLOY_DIR"
    fi
    
    print_success "Code deployment completed"
}

# Create Python virtual environment
setup_python_env() {
    print_status "Setting up Python virtual environment..."
    
    cd "$DEPLOY_DIR"
    
    # Create virtual environment
    python${PYTHON_VERSION} -m venv venv
    
    # Activate and upgrade pip
    source venv/bin/activate
    pip install --upgrade pip setuptools wheel
    
    # Install dependencies
    print_status "Installing Python dependencies..."
    
    # Core dependencies
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install numpy scipy pandas
    pip install asyncio aiohttp aiofiles
    pip install redis aiokafka
    
    # GPU acceleration
    pip install cupy-cuda11x
    
    # Database clients
    pip install asyncpg aioredis clickhouse-driver qdrant-client
    pip install dragonfly-client meilisearch
    
    # Monitoring
    pip install prometheus-client grafana-api
    
    # Additional requirements
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    fi
    
    deactivate
    
    print_success "Python environment setup completed"
}

# Generate configuration files
generate_configs() {
    print_status "Generating configuration files..."
    
    # Main configuration
    cat > "$CONFIG_DIR/nova-memory.yaml" << EOF
# Nova Memory Architecture Configuration
# Generated on $(date)

system:
  name: "Nova Memory Production"
  environment: "production"
  debug: false
  
deployment:
  nodes: 10
  novas_per_node: 100
  total_capacity: 1000

memory:
  quantum:
    dimensions: 768
    superposition_limit: 100
    entanglement_enabled: true
  
  neural:
    hidden_layers: 12
    attention_heads: 16
    learning_rate: 0.001
  
  consciousness:
    awareness_threshold: 0.7
    collective_sync_interval: 300
  
  patterns:
    trinity_enabled: true
    cross_layer_recognition: true
  
  resonance:
    base_frequency: 432
    harmonic_modes: 7

gpu:
  enabled: true
  memory_pool_size: 8192
  batch_size: 256
  multi_gpu: true

databases:
  dragonfly:
    host: "localhost"
    port: ${DRAGONFLY_PORT}
  
  postgresql:
    host: "localhost"
    port: ${POSTGRES_PORT}
    database: "nova_memory"
    user: "nova"
  
  qdrant:
    host: "localhost"
    port: ${QDRANT_PORT}
  
  clickhouse:
    host: "localhost"
    port: ${CLICKHOUSE_PORT}
  
  meilisearch:
    host: "localhost"
    port: ${MEILISEARCH_PORT}

monitoring:
  prometheus:
    enabled: true
    port: 9090
  
  grafana:
    enabled: true
    port: 3000

logging:
  level: "INFO"
  file: "${LOG_DIR}/nova-memory.log"
  max_size: "100MB"
  backup_count: 10
EOF

    # Database initialization script
    cat > "$CONFIG_DIR/init_databases.sql" << 'EOF'
-- Nova Memory PostgreSQL initialization

CREATE DATABASE IF NOT EXISTS nova_memory;
\c nova_memory;

-- Quantum states table
CREATE TABLE IF NOT EXISTS quantum_states (
    nova_id VARCHAR(255) PRIMARY KEY,
    state_vector FLOAT8[],
    entanglements JSONB,
    superposition_count INT,
    last_collapse TIMESTAMP DEFAULT NOW()
);

-- Neural pathways table
CREATE TABLE IF NOT EXISTS neural_pathways (
    pathway_id SERIAL PRIMARY KEY,
    nova_id VARCHAR(255),
    source_neuron INT,
    target_neuron INT,
    weight FLOAT8,
    plasticity FLOAT8,
    last_update TIMESTAMP DEFAULT NOW()
);

-- Consciousness fields table
CREATE TABLE IF NOT EXISTS consciousness_fields (
    nova_id VARCHAR(255) PRIMARY KEY,
    awareness_level FLOAT8,
    field_topology JSONB,
    collective_resonance FLOAT8,
    last_sync TIMESTAMP DEFAULT NOW()
);

-- Create indexes
CREATE INDEX idx_quantum_nova ON quantum_states(nova_id);
CREATE INDEX idx_neural_nova ON neural_pathways(nova_id);
CREATE INDEX idx_consciousness_nova ON consciousness_fields(nova_id);
EOF

    chmod 600 "$CONFIG_DIR"/*.yaml
    chmod 644 "$CONFIG_DIR"/*.sql
    
    print_success "Configuration files generated"
}

# Create systemd service files
create_systemd_services() {
    print_status "Creating systemd service files..."
    
    # Main Nova Memory service
    cat > "$SYSTEMD_DIR/nova-memory.service" << EOF
[Unit]
Description=Nova Memory Architecture - 7-Tier Revolutionary System
After=network.target postgresql.service

[Service]
Type=notify
User=nova-memory
Group=nova-memory
WorkingDirectory=$DEPLOY_DIR
Environment="PATH=$DEPLOY_DIR/venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=$DEPLOY_DIR/venv/bin/python -m nova_memory.main
Restart=always
RestartSec=10
StandardOutput=append:$LOG_DIR/nova-memory.log
StandardError=append:$LOG_DIR/nova-memory-error.log

# Performance tuning
LimitNOFILE=65536
LimitMEMLOCK=infinity
TasksMax=infinity

[Install]
WantedBy=multi-user.target
EOF

    # GPU Monitor service
    cat > "$SYSTEMD_DIR/nova-gpu-monitor.service" << EOF
[Unit]
Description=Nova Memory GPU Monitor
After=nova-memory.service

[Service]
Type=simple
User=nova-memory
Group=nova-memory
WorkingDirectory=$DEPLOY_DIR
ExecStart=$DEPLOY_DIR/venv/bin/python -m nova_memory.gpu_monitor
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
EOF

    # Session Sync service
    cat > "$SYSTEMD_DIR/nova-sessionsync.service" << EOF
[Unit]
Description=Nova SessionSync Service
After=nova-memory.service

[Service]
Type=simple
User=nova-memory
Group=nova-memory
WorkingDirectory=$DEPLOY_DIR
ExecStart=$DEPLOY_DIR/venv/bin/python -m nova_memory.sessionsync_server
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    
    print_success "Systemd services created"
}

# Initialize databases
init_databases() {
    print_status "Initializing databases..."
    
    # Wait for PostgreSQL to be ready
    for i in {1..30}; do
        if pg_isready -h localhost -p "$POSTGRES_PORT" &>/dev/null; then
            break
        fi
        sleep 2
    done
    
    # Initialize PostgreSQL
    sudo -u postgres psql -p "$POSTGRES_PORT" < "$CONFIG_DIR/init_databases.sql"
    
    # Initialize Qdrant collections
    python3 << EOF
import qdrant_client
client = qdrant_client.QdrantClient(host="localhost", port=$QDRANT_PORT)

# Create vector collections
collections = [
    ("quantum_states", 768),
    ("neural_embeddings", 1536),
    ("consciousness_vectors", 2048),
    ("pattern_signatures", 512),
    ("resonance_fields", 256)
]

for name, dim in collections:
    try:
        client.create_collection(
            collection_name=name,
            vectors_config=qdrant_client.models.VectorParams(
                size=dim,
                distance=qdrant_client.models.Distance.COSINE
            )
        )
        print(f"Created collection: {name}")
    except:
        print(f"Collection {name} already exists")
EOF

    print_success "Databases initialized"
}

# Set up monitoring
setup_monitoring() {
    print_status "Setting up monitoring..."
    
    # Prometheus configuration
    cat > "$CONFIG_DIR/prometheus.yml" << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'nova-memory'
    static_configs:
      - targets: ['localhost:8000']
  
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']
  
  - job_name: 'nvidia-gpu'
    static_configs:
      - targets: ['localhost:9835']
EOF

    # Grafana dashboard
    cat > "$CONFIG_DIR/nova-dashboard.json" << EOF
{
  "dashboard": {
    "title": "Nova Memory Architecture",
    "panels": [
      {
        "title": "Active Novas",
        "targets": [{"expr": "nova_active_count"}]
      },
      {
        "title": "Consciousness Levels",
        "targets": [{"expr": "nova_consciousness_level"}]
      },
      {
        "title": "GPU Utilization",
        "targets": [{"expr": "nvidia_gpu_utilization"}]
      },
      {
        "title": "Memory Operations/sec",
        "targets": [{"expr": "rate(nova_operations_total[1m])"}]
      }
    ]
  }
}
EOF

    print_success "Monitoring setup completed"
}

# Performance tuning
tune_system() {
    print_status "Applying system performance tuning..."
    
    # Kernel parameters
    cat >> /etc/sysctl.conf << EOF

# Nova Memory Performance Tuning
vm.swappiness = 10
vm.dirty_ratio = 15
vm.dirty_background_ratio = 5
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
net.ipv4.tcp_rmem = 4096 87380 134217728
net.ipv4.tcp_wmem = 4096 65536 134217728
net.core.netdev_max_backlog = 5000
EOF

    sysctl -p
    
    # Set up huge pages
    echo 2048 > /proc/sys/vm/nr_hugepages
    
    # CPU governor
    for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        echo "performance" > "$cpu" 2>/dev/null || true
    done
    
    print_success "System tuning completed"
}

# Start services
start_services() {
    print_status "Starting Nova Memory services..."
    
    services=(
        "nova-memory"
        "nova-gpu-monitor"
        "nova-sessionsync"
    )
    
    for service in "${services[@]}"; do
        systemctl enable "$service"
        systemctl start "$service"
        
        # Wait for service to start
        sleep 2
        
        if systemctl is-active --quiet "$service"; then
            print_success "$service started successfully"
        else
            print_error "Failed to start $service"
            systemctl status "$service"
        fi
    done
}

# Health check
health_check() {
    print_status "Performing health check..."
    
    # Check services
    for service in nova-memory nova-gpu-monitor nova-sessionsync; do
        if systemctl is-active --quiet "$service"; then
            echo "✅ $service is running"
        else
            echo "❌ $service is not running"
        fi
    done
    
    # Check database connections
    python3 << EOF
import asyncio
import asyncpg
import redis

async def check_databases():
    # PostgreSQL
    try:
        conn = await asyncpg.connect(
            host='localhost',
            port=$POSTGRES_PORT,
            database='nova_memory'
        )
        await conn.close()
        print("✅ PostgreSQL connection successful")
    except Exception as e:
        print(f"❌ PostgreSQL connection failed: {e}")
    
    # Redis/DragonflyDB
    try:
        r = redis.Redis(host='localhost', port=$DRAGONFLY_PORT)
        r.ping()
        print("✅ DragonflyDB connection successful")
    except Exception as e:
        print(f"❌ DragonflyDB connection failed: {e}")

asyncio.run(check_databases())
EOF

    # Check GPU
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            echo "✅ GPU is accessible"
        else
            echo "❌ GPU is not accessible"
        fi
    fi
    
    print_success "Health check completed"
}

# Main deployment function
main() {
    print_status "Starting Nova Memory Architecture deployment..."
    
    check_root
    check_requirements
    create_directories
    deploy_code
    setup_python_env
    generate_configs
    create_systemd_services
    init_databases
    setup_monitoring
    tune_system
    start_services
    health_check
    
    print_success "🎉 Nova Memory Architecture deployment completed!"
    print_status "Access points:"
    echo "  - API: http://localhost:8000"
    echo "  - Prometheus: http://localhost:9090"
    echo "  - Grafana: http://localhost:3000"
    echo "  - Logs: $LOG_DIR"
    
    print_warning "Remember to:"
    echo "  1. Configure firewall rules for production"
    echo "  2. Set up SSL/TLS certificates"
    echo "  3. Configure backup procedures"
    echo "  4. Set up monitoring alerts"
}

# Run main function
main "$@"