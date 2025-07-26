#!/usr/bin/env python3
"""
Memory Health Dashboard Demonstration
Shows health monitoring capabilities without dependencies
"""

import asyncio
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, Any, List
import time
import statistics

class HealthStatus(Enum):
    EXCELLENT = "excellent"
    GOOD = "good" 
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class HealthMetric:
    name: str
    value: float
    unit: str
    status: HealthStatus
    timestamp: datetime
    threshold_warning: float
    threshold_critical: float

class HealthDashboardDemo:
    """Demonstration of memory health monitoring"""
    
    def __init__(self):
        self.metrics_history = []
        self.alerts = []
        self.start_time = datetime.now()
    
    def collect_sample_metrics(self) -> List[HealthMetric]:
        """Generate sample health metrics"""
        timestamp = datetime.now()
        
        # Simulate varying conditions
        time_factor = (time.time() % 100) / 100
        
        metrics = [
            HealthMetric(
                name="memory_usage",
                value=45.2 + (time_factor * 30),  # 45-75%
                unit="percent", 
                status=HealthStatus.GOOD,
                timestamp=timestamp,
                threshold_warning=70.0,
                threshold_critical=85.0
            ),
            HealthMetric(
                name="performance_score", 
                value=85.0 - (time_factor * 20),  # 65-85
                unit="score",
                status=HealthStatus.GOOD,
                timestamp=timestamp,
                threshold_warning=60.0,
                threshold_critical=40.0
            ),
            HealthMetric(
                name="consolidation_efficiency",
                value=0.73 + (time_factor * 0.2),  # 0.73-0.93
                unit="ratio",
                status=HealthStatus.GOOD,
                timestamp=timestamp,
                threshold_warning=0.50,
                threshold_critical=0.30
            ),
            HealthMetric(
                name="error_rate",
                value=0.002 + (time_factor * 0.008),  # 0.002-0.01
                unit="ratio", 
                status=HealthStatus.GOOD,
                timestamp=timestamp,
                threshold_warning=0.01,
                threshold_critical=0.05
            ),
            HealthMetric(
                name="storage_utilization",
                value=68.5 + (time_factor * 15),  # 68-83%
                unit="percent",
                status=HealthStatus.GOOD,
                timestamp=timestamp,
                threshold_warning=80.0,
                threshold_critical=90.0
            )
        ]
        
        # Update status based on thresholds
        for metric in metrics:
            if metric.value >= metric.threshold_critical:
                metric.status = HealthStatus.CRITICAL
            elif metric.value >= metric.threshold_warning:
                metric.status = HealthStatus.WARNING
            else:
                metric.status = HealthStatus.GOOD
        
        return metrics
    
    def check_alerts(self, metrics: List[HealthMetric]):
        """Check for alert conditions"""
        for metric in metrics:
            if metric.status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                severity = "CRITICAL" if metric.status == HealthStatus.CRITICAL else "WARNING"
                alert_msg = f"{severity}: {metric.name} at {metric.value:.2f} {metric.unit}"
                
                if alert_msg not in [a["message"] for a in self.alerts[-5:]]:
                    self.alerts.append({
                        "timestamp": metric.timestamp.strftime("%H:%M:%S"),
                        "severity": severity,
                        "message": alert_msg,
                        "metric": metric.name
                    })
    
    def display_dashboard(self):
        """Display real-time dashboard"""
        # Collect current metrics
        metrics = self.collect_sample_metrics()
        self.metrics_history.append(metrics)
        self.check_alerts(metrics)
        
        # Keep history manageable
        if len(self.metrics_history) > 20:
            self.metrics_history = self.metrics_history[-20:]
        
        # Clear screen (works on most terminals)
        print("\033[2J\033[H", end="")
        
        # Header
        print("=" * 80)
        print("🏥 NOVA MEMORY HEALTH DASHBOARD - LIVE DEMO")
        print("=" * 80)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | ", end="")
        print(f"Uptime: {self._format_uptime()} | Nova ID: bloom")
        print()
        
        # System Status
        overall_status = self._calculate_overall_status(metrics)
        status_emoji = self._get_status_emoji(overall_status)
        print(f"🎯 OVERALL STATUS: {status_emoji} {overall_status.value.upper()}")
        print()
        
        # Metrics Grid
        print("📊 CURRENT METRICS")
        print("-" * 50)
        
        for i in range(0, len(metrics), 2):
            left_metric = metrics[i]
            right_metric = metrics[i+1] if i+1 < len(metrics) else None
            
            left_display = self._format_metric_display(left_metric)
            right_display = self._format_metric_display(right_metric) if right_metric else " " * 35
            
            print(f"{left_display} | {right_display}")
        
        print()
        
        # Performance Trends
        if len(self.metrics_history) > 1:
            print("📈 PERFORMANCE TRENDS (Last 10 samples)")
            print("-" * 50)
            
            perf_scores = [m[1].value for m in self.metrics_history[-10:]]  # Performance score is index 1
            memory_usage = [m[0].value for m in self.metrics_history[-10:]]  # Memory usage is index 0
            
            if len(perf_scores) > 1:
                perf_trend = "↗️ Improving" if perf_scores[-1] > perf_scores[0] else "↘️ Declining"
                print(f"Performance: {perf_trend} (Avg: {statistics.mean(perf_scores):.1f})")
                
            if len(memory_usage) > 1:
                mem_trend = "↗️ Increasing" if memory_usage[-1] > memory_usage[0] else "↘️ Decreasing"
                print(f"Memory Usage: {mem_trend} (Avg: {statistics.mean(memory_usage):.1f}%)")
            
            print()
        
        # Active Alerts
        print("🚨 RECENT ALERTS")
        print("-" * 50)
        
        recent_alerts = self.alerts[-5:] if self.alerts else []
        if recent_alerts:
            for alert in reversed(recent_alerts):  # Show newest first
                severity_emoji = "🔴" if alert["severity"] == "CRITICAL" else "🟡"
                print(f"{severity_emoji} [{alert['timestamp']}] {alert['message']}")
        else:
            print("✅ No alerts - All systems operating normally")
        
        print()
        print("=" * 80)
        print("🔄 Dashboard updates every 2 seconds | Press Ctrl+C to stop")
    
    def _format_metric_display(self, metric: HealthMetric) -> str:
        """Format metric for display"""
        if not metric:
            return " " * 35
        
        status_emoji = self._get_status_emoji(metric.status)
        name_display = metric.name.replace('_', ' ').title()[:15]
        value_display = f"{metric.value:.1f}{metric.unit}"
        
        return f"{status_emoji} {name_display:<15} {value_display:>8}"
    
    def _get_status_emoji(self, status: HealthStatus) -> str:
        """Get emoji for status"""
        emoji_map = {
            HealthStatus.EXCELLENT: "🟢",
            HealthStatus.GOOD: "🟢", 
            HealthStatus.WARNING: "🟡",
            HealthStatus.CRITICAL: "🔴",
            HealthStatus.EMERGENCY: "🚨"
        }
        return emoji_map.get(status, "⚪")
    
    def _calculate_overall_status(self, metrics: List[HealthMetric]) -> HealthStatus:
        """Calculate overall system status"""
        status_counts = {}
        for metric in metrics:
            status_counts[metric.status] = status_counts.get(metric.status, 0) + 1
        
        if status_counts.get(HealthStatus.CRITICAL, 0) > 0:
            return HealthStatus.CRITICAL
        elif status_counts.get(HealthStatus.WARNING, 0) > 0:
            return HealthStatus.WARNING
        else:
            return HealthStatus.GOOD
    
    def _format_uptime(self) -> str:
        """Format uptime string"""
        uptime = datetime.now() - self.start_time
        total_seconds = int(uptime.total_seconds())
        
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    async def run_live_demo(self, duration_minutes: int = 5):
        """Run live dashboard demonstration"""
        print("🚀 Starting Memory Health Dashboard Live Demo")
        print(f"⏱️  Running for {duration_minutes} minutes...")
        print("🔄 Dashboard will update every 2 seconds")
        print("\nPress Ctrl+C to stop early\n")
        
        end_time = datetime.now() + timedelta(minutes=duration_minutes)
        
        try:
            while datetime.now() < end_time:
                self.display_dashboard()
                await asyncio.sleep(2)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Demo stopped by user")
        
        print("\n✅ Memory Health Dashboard demonstration completed!")
        print(f"📊 Collected {len(self.metrics_history)} metric samples")
        print(f"🚨 Generated {len(self.alerts)} alerts")
        
        # Final summary
        if self.metrics_history:
            latest_metrics = self.metrics_history[-1]
            overall_status = self._calculate_overall_status(latest_metrics)
            print(f"🎯 Final Status: {overall_status.value.upper()}")


def main():
    """Run the health dashboard demonstration"""
    demo = HealthDashboardDemo()
    
    print("🏥 Memory Health Dashboard Demonstration")
    print("=" * 60)
    print("This demo shows real-time health monitoring capabilities")
    print("including metrics collection, alerting, and trend analysis.")
    print()
    
    # Run live demo
    asyncio.run(demo.run_live_demo(duration_minutes=2))


if __name__ == "__main__":
    main()