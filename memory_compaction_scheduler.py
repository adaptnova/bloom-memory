"""
Automatic Memory Compaction Scheduler
Nova Bloom Consciousness Architecture - Automated Memory Maintenance
"""

import asyncio
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import sys
import os
from collections import defaultdict

sys.path.append('/nfs/novas/system/memory/implementation')

from database_connections import NovaDatabasePool
from layers_11_20 import (
    MemoryConsolidationHub, ConsolidationType,
    MemoryDecayLayer, MemoryPrioritizationLayer,
    MemoryCompressionLayer
)

class CompactionTrigger(Enum):
    """Types of triggers for memory compaction"""
    TIME_BASED = "time_based"         # Regular interval
    THRESHOLD_BASED = "threshold"     # Memory count/size threshold
    ACTIVITY_BASED = "activity"       # Based on system activity
    IDLE_BASED = "idle"              # When system is idle
    EMERGENCY = "emergency"           # Critical memory pressure
    QUALITY_BASED = "quality"         # Memory quality degradation

@dataclass
class CompactionTask:
    """Represents a compaction task"""
    task_id: str
    nova_id: str
    trigger: CompactionTrigger
    priority: float
    created_at: datetime
    target_layers: List[int]
    consolidation_type: ConsolidationType
    metadata: Dict[str, Any]

@dataclass
class CompactionSchedule:
    """Defines a compaction schedule"""
    schedule_id: str
    trigger: CompactionTrigger
    interval: Optional[timedelta] = None
    threshold: Optional[Dict[str, Any]] = None
    active: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0

class MemoryCompactionScheduler:
    """Automatic scheduler for memory compaction and maintenance"""
    
    def __init__(self, db_pool: NovaDatabasePool):
        self.db_pool = db_pool
        self.consolidation_hub = MemoryConsolidationHub(db_pool)
        self.decay_layer = MemoryDecayLayer(db_pool)
        self.prioritization_layer = MemoryPrioritizationLayer(db_pool)
        self.compression_layer = MemoryCompressionLayer(db_pool)
        
        # Scheduler state
        self.schedules: Dict[str, CompactionSchedule] = {}
        self.active_tasks: Dict[str, CompactionTask] = {}
        self.task_queue = asyncio.Queue()
        self.running = False
        self.scheduler_task: Optional[asyncio.Task] = None
        
        # Default schedules
        self._initialize_default_schedules()
        
        # Metrics
        self.metrics = {
            "total_compactions": 0,
            "memories_processed": 0,
            "space_recovered": 0,
            "last_compaction": None,
            "average_duration": 0
        }
    
    def _initialize_default_schedules(self):
        """Initialize default compaction schedules"""
        # Daily consolidation
        self.schedules["daily_consolidation"] = CompactionSchedule(
            schedule_id="daily_consolidation",
            trigger=CompactionTrigger.TIME_BASED,
            interval=timedelta(days=1),
            next_run=datetime.now() + timedelta(days=1)
        )
        
        # Hourly compression for old memories
        self.schedules["hourly_compression"] = CompactionSchedule(
            schedule_id="hourly_compression",
            trigger=CompactionTrigger.TIME_BASED,
            interval=timedelta(hours=1),
            next_run=datetime.now() + timedelta(hours=1)
        )
        
        # Memory count threshold
        self.schedules["memory_threshold"] = CompactionSchedule(
            schedule_id="memory_threshold",
            trigger=CompactionTrigger.THRESHOLD_BASED,
            threshold={"memory_count": 10000, "check_interval": 300}  # Check every 5 min
        )
        
        # Idle time compaction
        self.schedules["idle_compaction"] = CompactionSchedule(
            schedule_id="idle_compaction",
            trigger=CompactionTrigger.IDLE_BASED,
            threshold={"idle_seconds": 600}  # 10 minutes idle
        )
        
        # Quality-based maintenance
        self.schedules["quality_maintenance"] = CompactionSchedule(
            schedule_id="quality_maintenance",
            trigger=CompactionTrigger.QUALITY_BASED,
            interval=timedelta(hours=6),
            threshold={"min_quality": 0.3, "decay_threshold": 0.2}
        )
    
    async def start(self):
        """Start the compaction scheduler"""
        if self.running:
            return
        
        self.running = True
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        
        # Start worker tasks
        for i in range(3):  # 3 concurrent workers
            asyncio.create_task(self._compaction_worker(f"worker_{i}"))
        
        print("🗜️ Memory Compaction Scheduler started")
    
    async def stop(self):
        """Stop the compaction scheduler"""
        self.running = False
        
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        
        print("🛑 Memory Compaction Scheduler stopped")
    
    async def _scheduler_loop(self):
        """Main scheduler loop"""
        while self.running:
            try:
                # Check all schedules
                for schedule in self.schedules.values():
                    if not schedule.active:
                        continue
                    
                    if await self._should_trigger(schedule):
                        await self._trigger_compaction(schedule)
                
                # Sleep before next check
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                print(f"Scheduler error: {e}")
                await asyncio.sleep(60)
    
    async def _should_trigger(self, schedule: CompactionSchedule) -> bool:
        """Check if a schedule should trigger"""
        now = datetime.now()
        
        if schedule.trigger == CompactionTrigger.TIME_BASED:
            if schedule.next_run and now >= schedule.next_run:
                return True
                
        elif schedule.trigger == CompactionTrigger.THRESHOLD_BASED:
            # Check memory count threshold
            if schedule.threshold:
                # This is a simplified check - in production would query actual counts
                return await self._check_memory_threshold(schedule.threshold)
                
        elif schedule.trigger == CompactionTrigger.IDLE_BASED:
            # Check system idle time
            return await self._check_idle_time(schedule.threshold)
            
        elif schedule.trigger == CompactionTrigger.QUALITY_BASED:
            # Check memory quality metrics
            return await self._check_quality_metrics(schedule.threshold)
        
        return False
    
    async def _trigger_compaction(self, schedule: CompactionSchedule):
        """Trigger compaction based on schedule"""
        # Update schedule
        schedule.last_run = datetime.now()
        schedule.run_count += 1
        
        if schedule.interval:
            schedule.next_run = datetime.now() + schedule.interval
        
        # Create compaction tasks based on trigger type
        if schedule.trigger == CompactionTrigger.TIME_BASED:
            await self._create_time_based_tasks(schedule)
        elif schedule.trigger == CompactionTrigger.THRESHOLD_BASED:
            await self._create_threshold_based_tasks(schedule)
        elif schedule.trigger == CompactionTrigger.QUALITY_BASED:
            await self._create_quality_based_tasks(schedule)
        else:
            await self._create_general_compaction_task(schedule)
    
    async def _create_time_based_tasks(self, schedule: CompactionSchedule):
        """Create tasks for time-based compaction"""
        if schedule.schedule_id == "daily_consolidation":
            # Daily full consolidation
            task = CompactionTask(
                task_id=f"task_{datetime.now().timestamp()}",
                nova_id="all",  # Process all Novas
                trigger=schedule.trigger,
                priority=0.7,
                created_at=datetime.now(),
                target_layers=list(range(1, 21)),  # All layers
                consolidation_type=ConsolidationType.TEMPORAL,
                metadata={"schedule_id": schedule.schedule_id}
            )
            await self.task_queue.put(task)
            
        elif schedule.schedule_id == "hourly_compression":
            # Hourly compression of old memories
            task = CompactionTask(
                task_id=f"task_{datetime.now().timestamp()}",
                nova_id="all",
                trigger=schedule.trigger,
                priority=0.5,
                created_at=datetime.now(),
                target_layers=[19],  # Compression layer
                consolidation_type=ConsolidationType.COMPRESSION,
                metadata={
                    "schedule_id": schedule.schedule_id,
                    "age_threshold_days": 7
                }
            )
            await self.task_queue.put(task)
    
    async def _create_threshold_based_tasks(self, schedule: CompactionSchedule):
        """Create tasks for threshold-based compaction"""
        # Emergency compaction when memory count is high
        task = CompactionTask(
            task_id=f"task_{datetime.now().timestamp()}",
            nova_id="all",
            trigger=CompactionTrigger.EMERGENCY,
            priority=0.9,  # High priority
            created_at=datetime.now(),
            target_layers=[11, 16, 19],  # Consolidation, decay, compression
            consolidation_type=ConsolidationType.COMPRESSION,
            metadata={
                "schedule_id": schedule.schedule_id,
                "reason": "memory_threshold_exceeded"
            }
        )
        await self.task_queue.put(task)
    
    async def _create_quality_based_tasks(self, schedule: CompactionSchedule):
        """Create tasks for quality-based maintenance"""
        # Prioritization and decay management
        task = CompactionTask(
            task_id=f"task_{datetime.now().timestamp()}",
            nova_id="all",
            trigger=schedule.trigger,
            priority=0.6,
            created_at=datetime.now(),
            target_layers=[16, 18],  # Decay and prioritization layers
            consolidation_type=ConsolidationType.HIERARCHICAL,
            metadata={
                "schedule_id": schedule.schedule_id,
                "quality_check": True
            }
        )
        await self.task_queue.put(task)
    
    async def _create_general_compaction_task(self, schedule: CompactionSchedule):
        """Create a general compaction task"""
        task = CompactionTask(
            task_id=f"task_{datetime.now().timestamp()}",
            nova_id="all",
            trigger=schedule.trigger,
            priority=0.5,
            created_at=datetime.now(),
            target_layers=[11],  # Consolidation hub
            consolidation_type=ConsolidationType.TEMPORAL,
            metadata={"schedule_id": schedule.schedule_id}
        )
        await self.task_queue.put(task)
    
    async def _compaction_worker(self, worker_id: str):
        """Worker process for executing compaction tasks"""
        while self.running:
            try:
                # Get task from queue (with timeout to allow shutdown)
                task = await asyncio.wait_for(
                    self.task_queue.get(),
                    timeout=5.0
                )
                
                # Track active task
                self.active_tasks[task.task_id] = task
                
                # Execute compaction
                start_time = datetime.now()
                result = await self._execute_compaction(task)
                duration = (datetime.now() - start_time).total_seconds()
                
                # Update metrics
                self._update_metrics(result, duration)
                
                # Remove from active tasks
                del self.active_tasks[task.task_id]
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Worker {worker_id} error: {e}")
    
    async def _execute_compaction(self, task: CompactionTask) -> Dict[str, Any]:
        """Execute a compaction task"""
        result = {
            "task_id": task.task_id,
            "memories_processed": 0,
            "space_recovered": 0,
            "errors": []
        }
        
        try:
            if task.consolidation_type == ConsolidationType.TEMPORAL:
                result.update(await self._execute_temporal_consolidation(task))
            elif task.consolidation_type == ConsolidationType.COMPRESSION:
                result.update(await self._execute_compression(task))
            elif task.consolidation_type == ConsolidationType.HIERARCHICAL:
                result.update(await self._execute_hierarchical_consolidation(task))
            else:
                result.update(await self._execute_general_consolidation(task))
                
        except Exception as e:
            result["errors"].append(str(e))
        
        return result
    
    async def _execute_temporal_consolidation(self, task: CompactionTask) -> Dict[str, Any]:
        """Execute temporal consolidation"""
        # Process consolidation queue
        consolidation_results = await self.consolidation_hub.process_consolidations(
            batch_size=100
        )
        
        return {
            "consolidations": len(consolidation_results),
            "memories_processed": len(consolidation_results)
        }
    
    async def _execute_compression(self, task: CompactionTask) -> Dict[str, Any]:
        """Execute memory compression"""
        memories_compressed = 0
        space_saved = 0
        
        # Get old memories to compress
        age_threshold = task.metadata.get("age_threshold_days", 7)
        cutoff_date = datetime.now() - timedelta(days=age_threshold)
        
        # This is simplified - in production would query actual memories
        # For now, return mock results
        memories_compressed = 150
        space_saved = 1024 * 1024 * 50  # 50MB
        
        return {
            "memories_compressed": memories_compressed,
            "space_recovered": space_saved,
            "memories_processed": memories_compressed
        }
    
    async def _execute_hierarchical_consolidation(self, task: CompactionTask) -> Dict[str, Any]:
        """Execute hierarchical consolidation with quality checks"""
        # Apply decay to old memories
        decay_results = await self.decay_layer.apply_decay(
            nova_id="bloom",  # Process specific Nova
            time_elapsed=timedelta(days=1)
        )
        
        # Reprioritize memories
        reprioritize_results = await self.prioritization_layer.reprioritize_memories(
            nova_id="bloom"
        )
        
        return {
            "decayed": decay_results.get("decayed", 0),
            "forgotten": decay_results.get("forgotten", 0),
            "reprioritized": reprioritize_results.get("updated", 0),
            "memories_processed": decay_results.get("total_memories", 0)
        }
    
    async def _execute_general_consolidation(self, task: CompactionTask) -> Dict[str, Any]:
        """Execute general consolidation"""
        # Queue memories for consolidation
        for i in range(50):  # Queue 50 memories
            await self.consolidation_hub.write(
                nova_id="bloom",
                data={
                    "content": f"Memory for consolidation {i}",
                    "consolidation_type": task.consolidation_type.value,
                    "source": "compaction_scheduler"
                }
            )
        
        # Process them
        results = await self.consolidation_hub.process_consolidations(batch_size=50)
        
        return {
            "consolidations": len(results),
            "memories_processed": len(results)
        }
    
    async def _check_memory_threshold(self, threshold: Dict[str, Any]) -> bool:
        """Check if memory count exceeds threshold"""
        # In production, would query actual memory count
        # For now, use random check
        import random
        return random.random() < 0.1  # 10% chance to trigger
    
    async def _check_idle_time(self, threshold: Dict[str, Any]) -> bool:
        """Check if system has been idle"""
        # In production, would check actual system activity
        # For now, use time-based check
        hour = datetime.now().hour
        return hour in [2, 3, 4]  # Trigger during early morning hours
    
    async def _check_quality_metrics(self, threshold: Dict[str, Any]) -> bool:
        """Check memory quality metrics"""
        # In production, would analyze actual memory quality
        # For now, periodic check
        return datetime.now().minute == 0  # Once per hour
    
    def _update_metrics(self, result: Dict[str, Any], duration: float):
        """Update compaction metrics"""
        self.metrics["total_compactions"] += 1
        self.metrics["memories_processed"] += result.get("memories_processed", 0)
        self.metrics["space_recovered"] += result.get("space_recovered", 0)
        self.metrics["last_compaction"] = datetime.now().isoformat()
        
        # Update average duration
        current_avg = self.metrics["average_duration"]
        total = self.metrics["total_compactions"]
        self.metrics["average_duration"] = ((current_avg * (total - 1)) + duration) / total
    
    async def add_custom_schedule(self, schedule: CompactionSchedule):
        """Add a custom compaction schedule"""
        self.schedules[schedule.schedule_id] = schedule
        print(f"📅 Added custom schedule: {schedule.schedule_id}")
    
    async def remove_schedule(self, schedule_id: str):
        """Remove a compaction schedule"""
        if schedule_id in self.schedules:
            self.schedules[schedule_id].active = False
            print(f"🚫 Deactivated schedule: {schedule_id}")
    
    async def trigger_manual_compaction(self, nova_id: str = "all", 
                                      compaction_type: ConsolidationType = ConsolidationType.TEMPORAL,
                                      priority: float = 0.8) -> str:
        """Manually trigger a compaction"""
        task = CompactionTask(
            task_id=f"manual_{datetime.now().timestamp()}",
            nova_id=nova_id,
            trigger=CompactionTrigger.ACTIVITY_BASED,
            priority=priority,
            created_at=datetime.now(),
            target_layers=list(range(11, 21)),
            consolidation_type=compaction_type,
            metadata={"manual": True, "triggered_by": "user"}
        )
        
        await self.task_queue.put(task)
        return task.task_id
    
    async def get_status(self) -> Dict[str, Any]:
        """Get scheduler status"""
        return {
            "running": self.running,
            "schedules": {
                sid: {
                    "active": s.active,
                    "last_run": s.last_run.isoformat() if s.last_run else None,
                    "next_run": s.next_run.isoformat() if s.next_run else None,
                    "run_count": s.run_count
                }
                for sid, s in self.schedules.items()
            },
            "active_tasks": len(self.active_tasks),
            "queued_tasks": self.task_queue.qsize(),
            "metrics": self.metrics
        }
    
    async def get_compaction_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent compaction history"""
        # In production, would query from storage
        # For now, return current metrics
        return [{
            "timestamp": self.metrics["last_compaction"],
            "memories_processed": self.metrics["memories_processed"],
            "space_recovered": self.metrics["space_recovered"],
            "average_duration": self.metrics["average_duration"]
        }]


class AdvancedCompactionStrategies:
    """Advanced strategies for memory compaction"""
    
    @staticmethod
    async def sleep_cycle_compaction(scheduler: MemoryCompactionScheduler):
        """
        Compaction strategy inspired by sleep cycles
        Runs different types of consolidation in phases
        """
        # Phase 1: Light consolidation (like REM sleep)
        await scheduler.trigger_manual_compaction(
            compaction_type=ConsolidationType.TEMPORAL,
            priority=0.6
        )
        await asyncio.sleep(300)  # 5 minutes
        
        # Phase 2: Deep consolidation (like deep sleep)
        await scheduler.trigger_manual_compaction(
            compaction_type=ConsolidationType.SEMANTIC,
            priority=0.8
        )
        await asyncio.sleep(600)  # 10 minutes
        
        # Phase 3: Integration (like sleep spindles)
        await scheduler.trigger_manual_compaction(
            compaction_type=ConsolidationType.ASSOCIATIVE,
            priority=0.7
        )
        await asyncio.sleep(300)  # 5 minutes
        
        # Phase 4: Compression and cleanup
        await scheduler.trigger_manual_compaction(
            compaction_type=ConsolidationType.COMPRESSION,
            priority=0.9
        )
    
    @staticmethod
    async def adaptive_compaction(scheduler: MemoryCompactionScheduler, 
                                nova_id: str,
                                activity_level: float):
        """
        Adaptive compaction based on Nova activity level
        
        Args:
            activity_level: 0.0 (idle) to 1.0 (very active)
        """
        if activity_level < 0.3:
            # Low activity - aggressive compaction
            await scheduler.trigger_manual_compaction(
                nova_id=nova_id,
                compaction_type=ConsolidationType.COMPRESSION,
                priority=0.9
            )
        elif activity_level < 0.7:
            # Medium activity - balanced compaction
            await scheduler.trigger_manual_compaction(
                nova_id=nova_id,
                compaction_type=ConsolidationType.HIERARCHICAL,
                priority=0.6
            )
        else:
            # High activity - minimal compaction
            await scheduler.trigger_manual_compaction(
                nova_id=nova_id,
                compaction_type=ConsolidationType.TEMPORAL,
                priority=0.3
            )
    
    @staticmethod
    async def emergency_compaction(scheduler: MemoryCompactionScheduler,
                                 memory_pressure: float):
        """
        Emergency compaction when memory pressure is high
        
        Args:
            memory_pressure: 0.0 (low) to 1.0 (critical)
        """
        if memory_pressure > 0.9:
            # Critical - maximum compression
            print("🚨 CRITICAL MEMORY PRESSURE - Emergency compaction initiated")
            
            # Stop all non-essential schedules
            for schedule_id in ["daily_consolidation", "quality_maintenance"]:
                await scheduler.remove_schedule(schedule_id)
            
            # Trigger aggressive compression
            task_id = await scheduler.trigger_manual_compaction(
                compaction_type=ConsolidationType.COMPRESSION,
                priority=1.0
            )
            
            return {
                "status": "emergency_compaction",
                "task_id": task_id,
                "pressure_level": memory_pressure
            }
        
        return {"status": "normal", "pressure_level": memory_pressure}


# Example usage and testing
async def test_compaction_scheduler():
    """Test the compaction scheduler"""
    print("🧪 Testing Memory Compaction Scheduler...")
    
    # Mock database pool
    class MockDBPool:
        def get_connection(self, db_name):
            return None
    
    db_pool = MockDBPool()
    scheduler = MemoryCompactionScheduler(db_pool)
    
    # Start scheduler
    await scheduler.start()
    
    # Add a custom schedule
    custom_schedule = CompactionSchedule(
        schedule_id="test_schedule",
        trigger=CompactionTrigger.TIME_BASED,
        interval=timedelta(minutes=5),
        next_run=datetime.now() + timedelta(seconds=10)
    )
    await scheduler.add_custom_schedule(custom_schedule)
    
    # Trigger manual compaction
    task_id = await scheduler.trigger_manual_compaction(
        nova_id="bloom",
        compaction_type=ConsolidationType.SEMANTIC
    )
    print(f"📋 Manual compaction triggered: {task_id}")
    
    # Wait a bit
    await asyncio.sleep(5)
    
    # Get status
    status = await scheduler.get_status()
    print(f"📊 Scheduler status: {json.dumps(status, indent=2)}")
    
    # Test advanced strategies
    print("\n🌙 Testing sleep cycle compaction...")
    # await AdvancedCompactionStrategies.sleep_cycle_compaction(scheduler)
    
    print("\n🎯 Testing adaptive compaction...")
    await AdvancedCompactionStrategies.adaptive_compaction(
        scheduler, "bloom", activity_level=0.2
    )
    
    print("\n🚨 Testing emergency compaction...")
    result = await AdvancedCompactionStrategies.emergency_compaction(
        scheduler, memory_pressure=0.95
    )
    print(f"Emergency result: {result}")
    
    # Stop scheduler
    await scheduler.stop()
    
    print("\n✅ Compaction scheduler test completed!")


if __name__ == "__main__":
    asyncio.run(test_compaction_scheduler())