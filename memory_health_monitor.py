#!/usr/bin/env python3
"""
Nova Memory System Health Monitor
Continuous monitoring and alerting for all memory databases
Author: Nova Bloom - Memory Architecture Lead
"""

import asyncio
import json
import time
import redis
import aiohttp
from datetime import datetime
from typing import Dict, Any, List
import psycopg2
import pymongo

class MemoryHealthMonitor:
    """Monitors all Nova memory system databases and publishes health status"""
    
    def __init__(self):
        # APEX Port Assignments
        self.databases = {
            "dragonfly": {
                "port": 18000,
                "type": "redis",
                "critical": True,
                "check_method": self.check_redis
            },
            "qdrant": {
                "port": 16333,
                "type": "http",
                "endpoint": "/collections",
                "critical": True,
                "check_method": self.check_http
            },
            "postgresql": {
                "port": 15432,
                "type": "postgresql",
                "critical": True,
                "check_method": self.check_postgresql
            },
            "clickhouse": {
                "port": 18123,
                "type": "http",
                "endpoint": "/ping",
                "critical": True,
                "check_method": self.check_http
            },
            "meilisearch": {
                "port": 19640,
                "type": "http",
                "endpoint": "/health",
                "critical": False,
                "check_method": self.check_http
            },
            "mongodb": {
                "port": 17017,
                "type": "mongodb",
                "critical": False,
                "check_method": self.check_mongodb
            }
        }
        
        # Connect to DragonflyDB for stream publishing
        self.redis_client = redis.Redis(host='localhost', port=18000, decode_responses=True)
        
        # Monitoring state
        self.check_interval = 60  # seconds
        self.last_status = {}
        self.failure_counts = {}
        self.alert_thresholds = {
            "warning": 2,   # failures before warning
            "critical": 5   # failures before critical alert
        }
        
    async def check_redis(self, name: str, config: Dict) -> Dict[str, Any]:
        """Check Redis/DragonflyDB health"""
        start_time = time.time()
        try:
            r = redis.Redis(host='localhost', port=config['port'], socket_timeout=5)
            r.ping()
            
            # Get additional metrics
            info = r.info()
            
            return {
                "status": "ONLINE",
                "latency_ms": round((time.time() - start_time) * 1000, 2),
                "version": info.get('redis_version', 'unknown'),
                "memory_used_mb": round(info.get('used_memory', 0) / 1024 / 1024, 2),
                "connected_clients": info.get('connected_clients', 0)
            }
        except Exception as e:
            return {
                "status": "OFFLINE",
                "error": str(e),
                "latency_ms": round((time.time() - start_time) * 1000, 2)
            }
    
    async def check_http(self, name: str, config: Dict) -> Dict[str, Any]:
        """Check HTTP-based databases"""
        start_time = time.time()
        url = f"http://localhost:{config['port']}{config.get('endpoint', '/')}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json() if response.content_type == 'application/json' else {}
                        
                        result = {
                            "status": "ONLINE",
                            "latency_ms": round((time.time() - start_time) * 1000, 2),
                            "http_status": response.status
                        }
                        
                        # Add service-specific metrics
                        if name == "qdrant":
                            result["collections"] = len(data.get('result', {}).get('collections', []))
                        
                        return result
                    else:
                        return {
                            "status": "DEGRADED",
                            "http_status": response.status,
                            "latency_ms": round((time.time() - start_time) * 1000, 2)
                        }
        except Exception as e:
            return {
                "status": "OFFLINE",
                "error": str(e),
                "latency_ms": round((time.time() - start_time) * 1000, 2)
            }
    
    async def check_postgresql(self, name: str, config: Dict) -> Dict[str, Any]:
        """Check PostgreSQL health"""
        start_time = time.time()
        try:
            conn = psycopg2.connect(
                host='localhost',
                port=config['port'],
                user='postgres',
                connect_timeout=5
            )
            cur = conn.cursor()
            cur.execute("SELECT version();")
            version = cur.fetchone()[0]
            
            # Get connection count
            cur.execute("SELECT count(*) FROM pg_stat_activity;")
            connections = cur.fetchone()[0]
            
            cur.close()
            conn.close()
            
            return {
                "status": "ONLINE",
                "latency_ms": round((time.time() - start_time) * 1000, 2),
                "version": version.split()[1],
                "connections": connections
            }
        except Exception as e:
            return {
                "status": "OFFLINE",
                "error": str(e),
                "latency_ms": round((time.time() - start_time) * 1000, 2)
            }
    
    async def check_mongodb(self, name: str, config: Dict) -> Dict[str, Any]:
        """Check MongoDB health"""
        start_time = time.time()
        try:
            client = pymongo.MongoClient(
                'localhost',
                config['port'],
                serverSelectionTimeoutMS=5000
            )
            # Ping to check connection
            client.admin.command('ping')
            
            # Get server status
            status = client.admin.command('serverStatus')
            
            client.close()
            
            return {
                "status": "ONLINE",
                "latency_ms": round((time.time() - start_time) * 1000, 2),
                "version": status.get('version', 'unknown'),
                "connections": status.get('connections', {}).get('current', 0)
            }
        except Exception as e:
            return {
                "status": "OFFLINE",
                "error": str(e),
                "latency_ms": round((time.time() - start_time) * 1000, 2)
            }
    
    async def check_all_databases(self) -> Dict[str, Any]:
        """Check all databases and compile health report"""
        results = {}
        tasks = []
        
        for name, config in self.databases.items():
            check_method = config['check_method']
            tasks.append(check_method(name, config))
        
        # Run all checks in parallel
        check_results = await asyncio.gather(*tasks)
        
        # Compile results
        for i, (name, config) in enumerate(self.databases.items()):
            results[name] = check_results[i]
            results[name]['port'] = config['port']
            results[name]['critical'] = config['critical']
        
        return results
    
    def determine_overall_health(self, results: Dict[str, Any]) -> str:
        """Determine overall system health based on individual checks"""
        critical_offline = any(
            db['status'] == 'OFFLINE' and db['critical'] 
            for db in results.values()
        )
        
        any_offline = any(db['status'] == 'OFFLINE' for db in results.values())
        any_degraded = any(db['status'] == 'DEGRADED' for db in results.values())
        
        if critical_offline:
            return "CRITICAL"
        elif any_offline or any_degraded:
            return "DEGRADED"
        else:
            return "HEALTHY"
    
    async def publish_status(self, results: Dict[str, Any], overall_health: str):
        """Publish health status to monitoring streams"""
        status_message = {
            "type": "HEALTH_CHECK",
            "timestamp": datetime.now().isoformat(),
            "databases": json.dumps(results),
            "overall_health": overall_health,
            "monitor_version": "1.0.0",
            "check_interval_seconds": str(self.check_interval)
        }
        
        # Always publish to main status stream
        self.redis_client.xadd("nova:memory:system:status", status_message)
        
        # Check for state changes and alert
        if overall_health != self.last_status.get('overall_health'):
            alert_message = {
                "type": "HEALTH_STATE_CHANGE",
                "previous_state": self.last_status.get('overall_health', 'UNKNOWN'),
                "current_state": overall_health,
                "timestamp": datetime.now().isoformat(),
                "details": json.dumps(results)
            }
            
            if overall_health == "CRITICAL":
                self.redis_client.xadd("nova:memory:alerts:critical", alert_message)
                self.redis_client.xadd("nova-urgent-alerts", alert_message)
            elif overall_health == "DEGRADED":
                self.redis_client.xadd("nova:memory:alerts:degraded", alert_message)
        
        # Track failure counts for individual databases
        for db_name, db_status in results.items():
            if db_status['status'] == 'OFFLINE':
                self.failure_counts[db_name] = self.failure_counts.get(db_name, 0) + 1
                
                # Alert on threshold breaches
                if self.failure_counts[db_name] == self.alert_thresholds['warning']:
                    self.redis_client.xadd("nova:memory:alerts:degraded", {
                        "type": "DATABASE_FAILURE_WARNING",
                        "database": db_name,
                        "consecutive_failures": self.failure_counts[db_name],
                        "timestamp": datetime.now().isoformat()
                    })
                elif self.failure_counts[db_name] >= self.alert_thresholds['critical']:
                    self.redis_client.xadd("nova:memory:alerts:critical", {
                        "type": "DATABASE_FAILURE_CRITICAL",
                        "database": db_name,
                        "consecutive_failures": self.failure_counts[db_name],
                        "timestamp": datetime.now().isoformat()
                    })
            else:
                # Reset failure count on success
                self.failure_counts[db_name] = 0
        
        # Store last status
        self.last_status = {
            "overall_health": overall_health,
            "timestamp": datetime.now().isoformat(),
            "databases": results
        }
    
    async def publish_performance_metrics(self, results: Dict[str, Any]):
        """Publish performance metrics for analysis"""
        latencies = {
            name: db.get('latency_ms', 0) 
            for name, db in results.items()
        }
        avg_latency = sum(
            db.get('latency_ms', 0) for db in results.values()
        ) / len(results) if results else 0
        memory_usage = {
            name: db.get('memory_used_mb', 0)
            for name, db in results.items()
            if 'memory_used_mb' in db
        }
        
        metrics = {
            "type": "PERFORMANCE_METRICS",
            "timestamp": datetime.now().isoformat(),
            "latencies": json.dumps(latencies),
            "avg_latency_ms": str(round(avg_latency, 2)),
            "memory_usage": json.dumps(memory_usage)
        }
        
        self.redis_client.xadd("nova:memory:performance", metrics)
    
    async def run_monitoring_loop(self):
        """Main monitoring loop"""
        print("🚀 Nova Memory Health Monitor Starting...")
        print(f"📊 Monitoring {len(self.databases)} databases")
        print(f"⏰ Check interval: {self.check_interval} seconds")
        
        # Announce monitor startup
        self.redis_client.xadd("nova:memory:system:status", {
            "type": "MONITOR_STARTUP",
            "timestamp": datetime.now().isoformat(),
            "message": "Memory health monitoring system online",
            "databases_monitored": json.dumps(list(self.databases.keys())),
            "check_interval": self.check_interval
        })
        
        while True:
            try:
                # Check all databases
                results = await self.check_all_databases()
                
                # Determine overall health
                overall_health = self.determine_overall_health(results)
                
                # Publish status
                await self.publish_status(results, overall_health)
                
                # Publish performance metrics
                await self.publish_performance_metrics(results)
                
                # Log to console
                print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Health Check Complete")
                print(f"Overall Status: {overall_health}")
                for name, status in results.items():
                    emoji = "✅" if status['status'] == "ONLINE" else "❌"
                    print(f"  {emoji} {name}: {status['status']} ({status.get('latency_ms', 'N/A')}ms)")
                
                # Wait for next check
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                print(f"❌ Monitor error: {e}")
                # Log error but continue monitoring
                self.redis_client.xadd("nova:memory:alerts:degraded", {
                    "type": "MONITOR_ERROR",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
                await asyncio.sleep(10)  # Brief pause before retry

async def main():
    """Run the health monitor"""
    monitor = MemoryHealthMonitor()
    await monitor.run_monitoring_loop()

if __name__ == "__main__":
    asyncio.run(main())