"""
Memory Health Monitoring Dashboard
Nova Bloom Consciousness Architecture - Real-time Memory Health Monitoring
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import json
import time
import statistics
import sys
import os

sys.path.append('/nfs/novas/system/memory/implementation')

from database_connections import NovaDatabasePool
from unified_memory_api import UnifiedMemoryAPI
from memory_compaction_scheduler import MemoryCompactionScheduler

class HealthStatus(Enum):
    """Health status levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class AlertType(Enum):
    """Types of health alerts"""
    MEMORY_PRESSURE = "memory_pressure"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    STORAGE_CAPACITY = "storage_capacity"
    CONSOLIDATION_BACKLOG = "consolidation_backlog"
    ERROR_RATE = "error_rate"
    DECAY_ACCELERATION = "decay_acceleration"

@dataclass
class HealthMetric:
    """Represents a health metric"""
    name: str
    value: float
    unit: str
    status: HealthStatus
    timestamp: datetime
    threshold_warning: float
    threshold_critical: float
    description: str

@dataclass
class HealthAlert:
    """Represents a health alert"""
    alert_id: str
    alert_type: AlertType
    severity: HealthStatus
    message: str
    timestamp: datetime
    nova_id: str
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None

@dataclass
class SystemHealth:
    """Overall system health summary"""
    overall_status: HealthStatus
    memory_usage_percent: float
    performance_score: float
    consolidation_efficiency: float
    error_rate: float
    active_alerts: int
    timestamp: datetime

class MemoryHealthMonitor:
    """Monitors memory system health metrics"""
    
    def __init__(self, db_pool: NovaDatabasePool, memory_api: UnifiedMemoryAPI):
        self.db_pool = db_pool
        self.memory_api = memory_api
        self.metrics_history: Dict[str, List[HealthMetric]] = {}
        self.active_alerts: List[HealthAlert] = []
        self.alert_history: List[HealthAlert] = []
        
        # Monitoring configuration
        self.monitoring_interval = 30  # seconds
        self.metrics_retention_days = 30
        self.alert_thresholds = self._initialize_thresholds()
        
        # Performance tracking
        self.performance_samples = []
        self.error_counts = {}
        
    def _initialize_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Initialize health monitoring thresholds"""
        return {
            "memory_usage": {"warning": 70.0, "critical": 85.0},
            "consolidation_backlog": {"warning": 1000.0, "critical": 5000.0},
            "error_rate": {"warning": 0.01, "critical": 0.05},
            "response_time": {"warning": 1.0, "critical": 5.0},
            "decay_rate": {"warning": 0.15, "critical": 0.30},
            "storage_utilization": {"warning": 80.0, "critical": 90.0},
            "fragmentation": {"warning": 30.0, "critical": 50.0}
        }
    
    async def collect_health_metrics(self, nova_id: str) -> List[HealthMetric]:
        """Collect comprehensive health metrics"""
        metrics = []
        timestamp = datetime.now()
        
        # Memory usage metrics
        memory_usage = await self._collect_memory_usage_metrics(nova_id, timestamp)
        metrics.extend(memory_usage)
        
        # Performance metrics
        performance = await self._collect_performance_metrics(nova_id, timestamp)
        metrics.extend(performance)
        
        # Storage metrics
        storage = await self._collect_storage_metrics(nova_id, timestamp)
        metrics.extend(storage)
        
        # Consolidation metrics
        consolidation = await self._collect_consolidation_metrics(nova_id, timestamp)
        metrics.extend(consolidation)
        
        # Error metrics
        error_metrics = await self._collect_error_metrics(nova_id, timestamp)
        metrics.extend(error_metrics)
        
        return metrics
    
    async def _collect_memory_usage_metrics(self, nova_id: str, timestamp: datetime) -> List[HealthMetric]:
        """Collect memory usage metrics"""
        metrics = []
        
        # Simulate memory usage data (in production would query actual usage)
        memory_usage_percent = 45.2  # Would calculate from actual memory pools
        
        thresholds = self.alert_thresholds["memory_usage"]
        status = self._determine_status(memory_usage_percent, thresholds)
        
        metrics.append(HealthMetric(
            name="memory_usage",
            value=memory_usage_percent,
            unit="percent",
            status=status,
            timestamp=timestamp,
            threshold_warning=thresholds["warning"],
            threshold_critical=thresholds["critical"],
            description="Percentage of memory pool currently in use"
        ))
        
        # Memory fragmentation
        fragmentation_percent = 12.8
        frag_thresholds = self.alert_thresholds["fragmentation"]
        frag_status = self._determine_status(fragmentation_percent, frag_thresholds)
        
        metrics.append(HealthMetric(
            name="memory_fragmentation",
            value=fragmentation_percent,
            unit="percent",
            status=frag_status,
            timestamp=timestamp,
            threshold_warning=frag_thresholds["warning"],
            threshold_critical=frag_thresholds["critical"],
            description="Memory fragmentation level"
        ))
        
        return metrics
    
    async def _collect_performance_metrics(self, nova_id: str, timestamp: datetime) -> List[HealthMetric]:
        """Collect performance metrics"""
        metrics = []
        
        # Average response time
        response_time = 0.23  # Would measure actual API response times
        resp_thresholds = self.alert_thresholds["response_time"]
        resp_status = self._determine_status(response_time, resp_thresholds)
        
        metrics.append(HealthMetric(
            name="avg_response_time",
            value=response_time,
            unit="seconds",
            status=resp_status,
            timestamp=timestamp,
            threshold_warning=resp_thresholds["warning"],
            threshold_critical=resp_thresholds["critical"],
            description="Average memory API response time"
        ))
        
        # Throughput (operations per second)
        throughput = 1250.0  # Would calculate from actual operation counts
        
        metrics.append(HealthMetric(
            name="throughput",
            value=throughput,
            unit="ops/sec",
            status=HealthStatus.GOOD,
            timestamp=timestamp,
            threshold_warning=500.0,
            threshold_critical=100.0,
            description="Memory operations per second"
        ))
        
        return metrics
    
    async def _collect_storage_metrics(self, nova_id: str, timestamp: datetime) -> List[HealthMetric]:
        """Collect storage-related metrics"""
        metrics = []
        
        # Storage utilization
        storage_util = 68.5  # Would calculate from actual storage usage
        storage_thresholds = self.alert_thresholds["storage_utilization"]
        storage_status = self._determine_status(storage_util, storage_thresholds)
        
        metrics.append(HealthMetric(
            name="storage_utilization",
            value=storage_util,
            unit="percent",
            status=storage_status,
            timestamp=timestamp,
            threshold_warning=storage_thresholds["warning"],
            threshold_critical=storage_thresholds["critical"],
            description="Storage space utilization percentage"
        ))
        
        # Database connection health
        connection_health = 95.0  # Percentage of healthy connections
        
        metrics.append(HealthMetric(
            name="db_connection_health",
            value=connection_health,
            unit="percent",
            status=HealthStatus.EXCELLENT,
            timestamp=timestamp,
            threshold_warning=90.0,
            threshold_critical=70.0,
            description="Database connection pool health"
        ))
        
        return metrics
    
    async def _collect_consolidation_metrics(self, nova_id: str, timestamp: datetime) -> List[HealthMetric]:
        """Collect consolidation and compaction metrics"""
        metrics = []
        
        # Consolidation backlog
        backlog_count = 342  # Would query actual consolidation queue
        backlog_thresholds = self.alert_thresholds["consolidation_backlog"]
        backlog_status = self._determine_status(backlog_count, backlog_thresholds)
        
        metrics.append(HealthMetric(
            name="consolidation_backlog",
            value=backlog_count,
            unit="items",
            status=backlog_status,
            timestamp=timestamp,
            threshold_warning=backlog_thresholds["warning"],
            threshold_critical=backlog_thresholds["critical"],
            description="Number of memories waiting for consolidation"
        ))
        
        # Compression efficiency
        compression_efficiency = 0.73  # Would calculate from actual compression stats
        
        metrics.append(HealthMetric(
            name="compression_efficiency",
            value=compression_efficiency,
            unit="ratio",
            status=HealthStatus.GOOD,
            timestamp=timestamp,
            threshold_warning=0.50,
            threshold_critical=0.30,
            description="Memory compression effectiveness ratio"
        ))
        
        return metrics
    
    async def _collect_error_metrics(self, nova_id: str, timestamp: datetime) -> List[HealthMetric]:
        """Collect error and reliability metrics"""
        metrics = []
        
        # Error rate
        error_rate = 0.003  # 0.3% error rate
        error_thresholds = self.alert_thresholds["error_rate"]
        error_status = self._determine_status(error_rate, error_thresholds)
        
        metrics.append(HealthMetric(
            name="error_rate",
            value=error_rate,
            unit="ratio",
            status=error_status,
            timestamp=timestamp,
            threshold_warning=error_thresholds["warning"],
            threshold_critical=error_thresholds["critical"],
            description="Percentage of operations resulting in errors"
        ))
        
        # Memory decay rate
        decay_rate = 0.08  # 8% decay rate
        decay_thresholds = self.alert_thresholds["decay_rate"]
        decay_status = self._determine_status(decay_rate, decay_thresholds)
        
        metrics.append(HealthMetric(
            name="memory_decay_rate",
            value=decay_rate,
            unit="ratio",
            status=decay_status,
            timestamp=timestamp,
            threshold_warning=decay_thresholds["warning"],
            threshold_critical=decay_thresholds["critical"],
            description="Rate of memory strength degradation"
        ))
        
        return metrics
    
    def _determine_status(self, value: float, thresholds: Dict[str, float]) -> HealthStatus:
        """Determine health status based on value and thresholds"""
        if value >= thresholds["critical"]:
            return HealthStatus.CRITICAL
        elif value >= thresholds["warning"]:
            return HealthStatus.WARNING
        else:
            return HealthStatus.GOOD
    
    async def check_for_alerts(self, metrics: List[HealthMetric], nova_id: str) -> List[HealthAlert]:
        """Check metrics for alert conditions"""
        new_alerts = []
        
        for metric in metrics:
            if metric.status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                alert = await self._create_alert(metric, nova_id)
                if alert:
                    new_alerts.append(alert)
        
        return new_alerts
    
    async def _create_alert(self, metric: HealthMetric, nova_id: str) -> Optional[HealthAlert]:
        """Create alert based on metric"""
        alert_id = f"alert_{int(time.time())}_{metric.name}"
        
        # Check if similar alert already exists
        existing_alert = next((a for a in self.active_alerts 
                              if a.nova_id == nova_id and metric.name in a.message and not a.resolved), None)
        
        if existing_alert:
            return None  # Don't create duplicate alerts
        
        # Determine alert type
        alert_type = self._determine_alert_type(metric.name)
        
        # Create alert message
        message = self._generate_alert_message(metric)
        
        alert = HealthAlert(
            alert_id=alert_id,
            alert_type=alert_type,
            severity=metric.status,
            message=message,
            timestamp=datetime.now(),
            nova_id=nova_id
        )
        
        return alert
    
    def _determine_alert_type(self, metric_name: str) -> AlertType:
        """Determine alert type based on metric name"""
        if "memory" in metric_name or "storage" in metric_name:
            return AlertType.MEMORY_PRESSURE
        elif "response_time" in metric_name or "throughput" in metric_name:
            return AlertType.PERFORMANCE_DEGRADATION
        elif "consolidation" in metric_name:
            return AlertType.CONSOLIDATION_BACKLOG
        elif "error" in metric_name:
            return AlertType.ERROR_RATE
        elif "decay" in metric_name:
            return AlertType.DECAY_ACCELERATION
        else:
            return AlertType.MEMORY_PRESSURE
    
    def _generate_alert_message(self, metric: HealthMetric) -> str:
        """Generate alert message based on metric"""
        severity = "CRITICAL" if metric.status == HealthStatus.CRITICAL else "WARNING"
        
        if metric.name == "memory_usage":
            return f"{severity}: Memory usage at {metric.value:.1f}% (threshold: {metric.threshold_warning:.1f}%)"
        elif metric.name == "consolidation_backlog":
            return f"{severity}: Consolidation backlog at {int(metric.value)} items (threshold: {int(metric.threshold_warning)})"
        elif metric.name == "error_rate":
            return f"{severity}: Error rate at {metric.value:.3f} (threshold: {metric.threshold_warning:.3f})"
        elif metric.name == "avg_response_time":
            return f"{severity}: Average response time {metric.value:.2f}s (threshold: {metric.threshold_warning:.2f}s)"
        else:
            return f"{severity}: {metric.name} at {metric.value:.2f} {metric.unit}"
    
    async def store_metrics(self, metrics: List[HealthMetric], nova_id: str):
        """Store metrics for historical analysis"""
        for metric in metrics:
            key = f"{nova_id}:{metric.name}"
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            
            self.metrics_history[key].append(metric)
            
            # Keep only recent metrics
            cutoff_time = datetime.now() - timedelta(days=self.metrics_retention_days)
            self.metrics_history[key] = [
                m for m in self.metrics_history[key] if m.timestamp > cutoff_time
            ]
    
    async def get_system_health_summary(self, nova_id: str) -> SystemHealth:
        """Get overall system health summary"""
        metrics = await self.collect_health_metrics(nova_id)
        
        # Calculate overall status
        status_counts = {}
        for metric in metrics:
            status = metric.status
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Determine overall status
        if status_counts.get(HealthStatus.CRITICAL, 0) > 0:
            overall_status = HealthStatus.CRITICAL
        elif status_counts.get(HealthStatus.WARNING, 0) > 0:
            overall_status = HealthStatus.WARNING
        else:
            overall_status = HealthStatus.GOOD
        
        # Calculate key metrics
        memory_usage = next((m.value for m in metrics if m.name == "memory_usage"), 0.0)
        response_time = next((m.value for m in metrics if m.name == "avg_response_time"), 0.0)
        throughput = next((m.value for m in metrics if m.name == "throughput"), 0.0)
        compression_eff = next((m.value for m in metrics if m.name == "compression_efficiency"), 0.0)
        error_rate = next((m.value for m in metrics if m.name == "error_rate"), 0.0)
        
        # Calculate performance score (0-100)
        performance_score = max(0, 100 - (response_time * 20) - (error_rate * 1000))
        performance_score = min(100, performance_score)
        
        return SystemHealth(
            overall_status=overall_status,
            memory_usage_percent=memory_usage,
            performance_score=performance_score,
            consolidation_efficiency=compression_eff,
            error_rate=error_rate,
            active_alerts=len([a for a in self.active_alerts if not a.resolved]),
            timestamp=datetime.now()
        )

class MemoryHealthDashboard:
    """Interactive memory health monitoring dashboard"""
    
    def __init__(self, db_pool: NovaDatabasePool):
        self.db_pool = db_pool
        self.memory_api = UnifiedMemoryAPI(db_pool)
        self.health_monitor = MemoryHealthMonitor(db_pool, self.memory_api)
        self.running = False
        self.monitor_task: Optional[asyncio.Task] = None
        
        # Dashboard state
        self.current_metrics: Dict[str, List[HealthMetric]] = {}
        self.health_history: List[SystemHealth] = []
        self.dashboard_config = {
            "refresh_interval": 10,  # seconds
            "alert_sound": True,
            "show_trends": True,
            "compact_view": False
        }
    
    async def start_monitoring(self, nova_ids: List[str] = None):
        """Start continuous health monitoring"""
        if self.running:
            return
        
        self.running = True
        nova_ids = nova_ids or ["bloom"]  # Default to monitoring bloom
        
        self.monitor_task = asyncio.create_task(self._monitoring_loop(nova_ids))
        print("🏥 Memory Health Dashboard started")
    
    async def stop_monitoring(self):
        """Stop health monitoring"""
        self.running = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        print("🛑 Memory Health Dashboard stopped")
    
    async def _monitoring_loop(self, nova_ids: List[str]):
        """Main monitoring loop"""
        while self.running:
            try:
                for nova_id in nova_ids:
                    # Collect metrics
                    metrics = await self.health_monitor.collect_health_metrics(nova_id)
                    
                    # Store metrics
                    await self.health_monitor.store_metrics(metrics, nova_id)
                    self.current_metrics[nova_id] = metrics
                    
                    # Check for alerts
                    new_alerts = await self.health_monitor.check_for_alerts(metrics, nova_id)
                    if new_alerts:
                        self.health_monitor.active_alerts.extend(new_alerts)
                        for alert in new_alerts:
                            await self._handle_new_alert(alert)
                    
                    # Update health history
                    system_health = await self.health_monitor.get_system_health_summary(nova_id)
                    self.health_history.append(system_health)
                    
                    # Keep history manageable
                    if len(self.health_history) > 1440:  # 24 hours at 1-minute intervals
                        self.health_history = self.health_history[-1440:]
                
                # Sleep before next collection
                await asyncio.sleep(self.dashboard_config["refresh_interval"])
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                await asyncio.sleep(30)  # Wait longer after error
    
    async def _handle_new_alert(self, alert: HealthAlert):
        """Handle new alert"""
        print(f"🚨 NEW ALERT: {alert.message}")
        
        # Auto-remediation for certain alerts
        if alert.alert_type == AlertType.CONSOLIDATION_BACKLOG:
            await self._trigger_consolidation(alert.nova_id)
        elif alert.alert_type == AlertType.MEMORY_PRESSURE:
            await self._trigger_compression(alert.nova_id)
    
    async def _trigger_consolidation(self, nova_id: str):
        """Trigger automatic consolidation"""
        print(f"🔄 Auto-triggering consolidation for {nova_id}")
        # Would integrate with compaction scheduler here
    
    async def _trigger_compression(self, nova_id: str):
        """Trigger automatic compression"""
        print(f"🗜️ Auto-triggering compression for {nova_id}")
        # Would integrate with compaction scheduler here
    
    def display_dashboard(self, nova_id: str = "bloom"):
        """Display current dashboard"""
        print(self._generate_dashboard_display(nova_id))
    
    def _generate_dashboard_display(self, nova_id: str) -> str:
        """Generate dashboard display string"""
        output = []
        output.append("=" * 80)
        output.append("🏥 NOVA MEMORY HEALTH DASHBOARD")
        output.append("=" * 80)
        output.append(f"Nova ID: {nova_id}")
        output.append(f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        output.append("")
        
        # System Health Summary
        if self.health_history:
            latest_health = self.health_history[-1]
            output.append("📊 SYSTEM HEALTH SUMMARY")
            output.append("-" * 40)
            output.append(f"Overall Status: {self._status_emoji(latest_health.overall_status)} {latest_health.overall_status.value.upper()}")
            output.append(f"Memory Usage: {latest_health.memory_usage_percent:.1f}%")
            output.append(f"Performance Score: {latest_health.performance_score:.1f}/100")
            output.append(f"Consolidation Efficiency: {latest_health.consolidation_efficiency:.1f}")
            output.append(f"Error Rate: {latest_health.error_rate:.3f}")
            output.append(f"Active Alerts: {latest_health.active_alerts}")
            output.append("")
        
        # Current Metrics
        if nova_id in self.current_metrics:
            metrics = self.current_metrics[nova_id]
            output.append("📈 CURRENT METRICS")
            output.append("-" * 40)
            
            for metric in metrics:
                status_emoji = self._status_emoji(metric.status)
                output.append(f"{status_emoji} {metric.name}: {metric.value:.2f} {metric.unit}")
                
                if metric.status != HealthStatus.GOOD:
                    if metric.status == HealthStatus.WARNING:
                        output.append(f"   ⚠️  Above warning threshold ({metric.threshold_warning:.2f})")
                    elif metric.status == HealthStatus.CRITICAL:
                        output.append(f"   🔴 Above critical threshold ({metric.threshold_critical:.2f})")
            
            output.append("")
        
        # Active Alerts
        active_alerts = [a for a in self.health_monitor.active_alerts if not a.resolved and a.nova_id == nova_id]
        if active_alerts:
            output.append("🚨 ACTIVE ALERTS")
            output.append("-" * 40)
            for alert in active_alerts[-5:]:  # Show last 5 alerts
                age = datetime.now() - alert.timestamp
                age_str = f"{int(age.total_seconds() / 60)}m ago"
                output.append(f"{self._status_emoji(alert.severity)} {alert.message} ({age_str})")
            output.append("")
        
        # Performance Trends
        if len(self.health_history) > 1:
            output.append("📊 PERFORMANCE TRENDS")
            output.append("-" * 40)
            
            recent_scores = [h.performance_score for h in self.health_history[-10:]]
            if len(recent_scores) > 1:
                trend = "📈 Improving" if recent_scores[-1] > recent_scores[0] else "📉 Declining"
                avg_score = statistics.mean(recent_scores)
                output.append(f"Performance Trend: {trend}")
                output.append(f"Average Score (10 samples): {avg_score:.1f}")
            
            recent_memory = [h.memory_usage_percent for h in self.health_history[-10:]]
            if len(recent_memory) > 1:
                trend = "📈 Increasing" if recent_memory[-1] > recent_memory[0] else "📉 Decreasing"
                avg_memory = statistics.mean(recent_memory)
                output.append(f"Memory Usage Trend: {trend}")
                output.append(f"Average Usage (10 samples): {avg_memory:.1f}%")
            
            output.append("")
        
        output.append("=" * 80)
        return "\n".join(output)
    
    def _status_emoji(self, status: HealthStatus) -> str:
        """Get emoji for health status"""
        emoji_map = {
            HealthStatus.EXCELLENT: "🟢",
            HealthStatus.GOOD: "🟢",
            HealthStatus.WARNING: "🟡",
            HealthStatus.CRITICAL: "🔴",
            HealthStatus.EMERGENCY: "🚨"
        }
        return emoji_map.get(status, "⚪")
    
    async def get_metrics_report(self, nova_id: str, hours: int = 24) -> Dict[str, Any]:
        """Get detailed metrics report"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Filter metrics
        recent_health = [h for h in self.health_history if h.timestamp > cutoff_time]
        
        if not recent_health:
            return {"error": "No data available for the specified time period"}
        
        # Calculate statistics
        memory_usage = [h.memory_usage_percent for h in recent_health]
        performance = [h.performance_score for h in recent_health]
        error_rates = [h.error_rate for h in recent_health]
        
        return {
            "nova_id": nova_id,
            "time_period_hours": hours,
            "sample_count": len(recent_health),
            "memory_usage": {
                "current": memory_usage[-1] if memory_usage else 0,
                "average": statistics.mean(memory_usage) if memory_usage else 0,
                "max": max(memory_usage) if memory_usage else 0,
                "min": min(memory_usage) if memory_usage else 0
            },
            "performance": {
                "current": performance[-1] if performance else 0,
                "average": statistics.mean(performance) if performance else 0,
                "max": max(performance) if performance else 0,
                "min": min(performance) if performance else 0
            },
            "error_rates": {
                "current": error_rates[-1] if error_rates else 0,
                "average": statistics.mean(error_rates) if error_rates else 0,
                "max": max(error_rates) if error_rates else 0
            },
            "alerts": {
                "total_active": len([a for a in self.health_monitor.active_alerts if not a.resolved]),
                "critical_count": len([a for a in self.health_monitor.active_alerts 
                                     if a.severity == HealthStatus.CRITICAL and not a.resolved]),
                "warning_count": len([a for a in self.health_monitor.active_alerts 
                                    if a.severity == HealthStatus.WARNING and not a.resolved])
            }
        }
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Manually resolve an alert"""
        for alert in self.health_monitor.active_alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                alert.resolution_timestamp = datetime.now()
                print(f"✅ Resolved alert: {alert.message}")
                return True
        return False
    
    async def set_threshold(self, metric_name: str, warning: float, critical: float):
        """Update alert thresholds"""
        if metric_name in self.health_monitor.alert_thresholds:
            self.health_monitor.alert_thresholds[metric_name] = {
                "warning": warning,
                "critical": critical
            }
            print(f"📊 Updated thresholds for {metric_name}: warning={warning}, critical={critical}")
        else:
            print(f"❌ Unknown metric: {metric_name}")
    
    def configure_dashboard(self, **kwargs):
        """Configure dashboard settings"""
        for key, value in kwargs.items():
            if key in self.dashboard_config:
                self.dashboard_config[key] = value
                print(f"⚙️ Dashboard setting updated: {key} = {value}")


# Mock database pool for demonstration
class MockDatabasePool:
    def get_connection(self, db_name):
        return None

class MockMemoryAPI:
    def __init__(self, db_pool):
        self.db_pool = db_pool

# Demo function
async def demo_health_dashboard():
    """Demonstrate the health monitoring dashboard"""
    print("🏥 Memory Health Dashboard Demonstration")
    print("=" * 60)
    
    # Initialize
    db_pool = MockDatabasePool()
    dashboard = MemoryHealthDashboard(db_pool)
    
    # Start monitoring
    await dashboard.start_monitoring(["bloom", "nova_001"])
    
    # Let it collect some data
    print("📊 Collecting initial health metrics...")
    await asyncio.sleep(3)
    
    # Display dashboard
    print("\n" + "📺 DASHBOARD DISPLAY:")
    dashboard.display_dashboard("bloom")
    
    # Simulate some alerts
    print("\n🚨 Simulating high memory usage alert...")
    high_memory_metric = HealthMetric(
        name="memory_usage",
        value=87.5,  # Above critical threshold
        unit="percent",
        status=HealthStatus.CRITICAL,
        timestamp=datetime.now(),
        threshold_warning=70.0,
        threshold_critical=85.0,
        description="Memory usage critical"
    )
    
    alert = await dashboard.health_monitor._create_alert(high_memory_metric, "bloom")
    if alert:
        dashboard.health_monitor.active_alerts.append(alert)
        await dashboard._handle_new_alert(alert)
    
    # Display updated dashboard
    print("\n📺 UPDATED DASHBOARD (with alert):")
    dashboard.display_dashboard("bloom")
    
    # Get detailed report
    print("\n📋 24-HOUR METRICS REPORT:")
    report = await dashboard.get_metrics_report("bloom", 24)
    print(json.dumps(report, indent=2, default=str))
    
    # Test threshold adjustment
    print("\n⚙️ Adjusting memory usage thresholds...")
    await dashboard.set_threshold("memory_usage", 75.0, 90.0)
    
    # Stop monitoring
    await dashboard.stop_monitoring()
    
    print("\n✅ Health Dashboard demonstration completed!")


if __name__ == "__main__":
    asyncio.run(demo_health_dashboard())