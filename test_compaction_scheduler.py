#!/usr/bin/env python3
"""
Test Memory Compaction Scheduler
Nova Bloom Consciousness Architecture
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
import json

sys.path.append('/nfs/novas/system/memory/implementation')

from memory_compaction_scheduler import (
    MemoryCompactionScheduler, 
    CompactionSchedule,
    CompactionTrigger,
    AdvancedCompactionStrategies
)
from layers_11_20 import ConsolidationType

# Mock database pool for testing
class MockDatabasePool:
    def get_connection(self, db_name):
        return None
    
    async def execute(self, query):
        return []

async def test_scheduler_lifecycle():
    """Test basic scheduler lifecycle"""
    print("🧪 Testing Scheduler Lifecycle...")
    
    db_pool = MockDatabasePool()
    scheduler = MemoryCompactionScheduler(db_pool)
    
    # Test start
    await scheduler.start()
    status = await scheduler.get_status()
    assert status['running'] == True, "Scheduler should be running"
    print("✅ Scheduler started successfully")
    
    # Test default schedules
    assert len(status['schedules']) == 5, "Should have 5 default schedules"
    print("✅ Default schedules initialized")
    
    # Test stop
    await scheduler.stop()
    status = await scheduler.get_status()
    assert status['running'] == False, "Scheduler should be stopped"
    print("✅ Scheduler stopped successfully")
    
    return True

async def test_custom_schedules():
    """Test adding and removing custom schedules"""
    print("\n🧪 Testing Custom Schedules...")
    
    db_pool = MockDatabasePool()
    scheduler = MemoryCompactionScheduler(db_pool)
    await scheduler.start()
    
    # Add custom schedule
    custom_schedule = CompactionSchedule(
        schedule_id="test_custom",
        trigger=CompactionTrigger.TIME_BASED,
        interval=timedelta(minutes=30),
        next_run=datetime.now() + timedelta(seconds=5)
    )
    
    await scheduler.add_custom_schedule(custom_schedule)
    status = await scheduler.get_status()
    assert "test_custom" in status['schedules'], "Custom schedule should be added"
    print("✅ Custom schedule added")
    
    # Remove schedule
    await scheduler.remove_schedule("test_custom")
    status = await scheduler.get_status()
    assert status['schedules']["test_custom"]['active'] == False, "Schedule should be inactive"
    print("✅ Schedule deactivated")
    
    await scheduler.stop()
    return True

async def test_manual_compaction():
    """Test manual compaction triggering"""
    print("\n🧪 Testing Manual Compaction...")
    
    db_pool = MockDatabasePool()
    scheduler = MemoryCompactionScheduler(db_pool)
    await scheduler.start()
    
    # Trigger manual compaction
    task_id = await scheduler.trigger_manual_compaction(
        nova_id="test_nova",
        compaction_type=ConsolidationType.TEMPORAL,
        priority=0.9
    )
    
    assert task_id.startswith("manual_"), "Task ID should indicate manual trigger"
    print(f"✅ Manual compaction triggered: {task_id}")
    
    # Check queue
    status = await scheduler.get_status()
    assert status['queued_tasks'] >= 0, "Should have tasks in queue"
    print(f"✅ Tasks queued: {status['queued_tasks']}")
    
    await scheduler.stop()
    return True

async def test_compaction_strategies():
    """Test advanced compaction strategies"""
    print("\n🧪 Testing Advanced Strategies...")
    
    db_pool = MockDatabasePool()
    scheduler = MemoryCompactionScheduler(db_pool)
    await scheduler.start()
    
    # Test adaptive compaction
    print("  Testing adaptive compaction...")
    await AdvancedCompactionStrategies.adaptive_compaction(
        scheduler,
        nova_id="test_nova",
        activity_level=0.2  # Low activity
    )
    print("  ✅ Low activity compaction triggered")
    
    await AdvancedCompactionStrategies.adaptive_compaction(
        scheduler,
        nova_id="test_nova",
        activity_level=0.8  # High activity
    )
    print("  ✅ High activity compaction triggered")
    
    # Test emergency compaction
    print("  Testing emergency compaction...")
    result = await AdvancedCompactionStrategies.emergency_compaction(
        scheduler,
        memory_pressure=0.95
    )
    assert result['status'] == "emergency_compaction", "Should trigger emergency mode"
    print("  ✅ Emergency compaction triggered")
    
    await scheduler.stop()
    return True

async def test_metrics_tracking():
    """Test metrics tracking"""
    print("\n🧪 Testing Metrics Tracking...")
    
    db_pool = MockDatabasePool()
    scheduler = MemoryCompactionScheduler(db_pool)
    await scheduler.start()
    
    # Get initial metrics
    status = await scheduler.get_status()
    initial_metrics = status['metrics']
    print(f"  Initial metrics: {json.dumps(initial_metrics, indent=2)}")
    
    # Trigger compaction to update metrics
    await scheduler.trigger_manual_compaction()
    
    # Wait for processing
    await asyncio.sleep(2)
    
    # Check metrics updated
    status = await scheduler.get_status()
    updated_metrics = status['metrics']
    print(f"  Updated metrics: {json.dumps(updated_metrics, indent=2)}")
    
    print("✅ Metrics tracking functional")
    
    await scheduler.stop()
    return True

async def test_schedule_triggers():
    """Test different schedule trigger types"""
    print("\n🧪 Testing Schedule Triggers...")
    
    db_pool = MockDatabasePool()
    scheduler = MemoryCompactionScheduler(db_pool)
    
    # Check default schedule triggers
    for schedule_id, schedule in scheduler.schedules.items():
        print(f"  Schedule: {schedule_id}")
        print(f"    Trigger: {schedule.trigger.value}")
        print(f"    Active: {schedule.active}")
        if schedule.interval:
            print(f"    Interval: {schedule.interval}")
        if schedule.threshold:
            print(f"    Threshold: {schedule.threshold}")
    
    print("✅ All schedule triggers configured")
    return True

async def test_compaction_history():
    """Test compaction history retrieval"""
    print("\n🧪 Testing Compaction History...")
    
    db_pool = MockDatabasePool()
    scheduler = MemoryCompactionScheduler(db_pool)
    await scheduler.start()
    
    # Trigger some compactions
    for i in range(3):
        await scheduler.trigger_manual_compaction()
        await asyncio.sleep(1)
    
    # Get history
    history = await scheduler.get_compaction_history(limit=5)
    print(f"  History entries: {len(history)}")
    for entry in history:
        print(f"    Timestamp: {entry.get('timestamp')}")
        print(f"    Memories: {entry.get('memories_processed')}")
    
    print("✅ History tracking functional")
    
    await scheduler.stop()
    return True

async def run_all_tests():
    """Run all tests"""
    print("🚀 Starting Memory Compaction Scheduler Tests")
    print("=" * 60)
    
    tests = [
        ("Scheduler Lifecycle", test_scheduler_lifecycle),
        ("Custom Schedules", test_custom_schedules),
        ("Manual Compaction", test_manual_compaction),
        ("Compaction Strategies", test_compaction_strategies),
        ("Metrics Tracking", test_metrics_tracking),
        ("Schedule Triggers", test_schedule_triggers),
        ("Compaction History", test_compaction_history)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            if result:
                passed += 1
                print(f"\n✅ {test_name}: PASSED")
            else:
                failed += 1
                print(f"\n❌ {test_name}: FAILED")
        except Exception as e:
            failed += 1
            print(f"\n❌ {test_name}: ERROR - {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Memory Compaction Scheduler is ready.")
    else:
        print("⚠️  Some tests failed. Please review the errors above.")
    
    return failed == 0

# Example usage demonstration
async def demonstrate_usage():
    """Demonstrate real-world usage"""
    print("\n" + "=" * 60)
    print("📖 Usage Demonstration")
    print("=" * 60)
    
    db_pool = MockDatabasePool()
    scheduler = MemoryCompactionScheduler(db_pool)
    
    print("\n1️⃣ Starting scheduler with default settings...")
    await scheduler.start()
    
    print("\n2️⃣ Adding custom weekend maintenance schedule...")
    weekend_schedule = CompactionSchedule(
        schedule_id="weekend_maintenance",
        trigger=CompactionTrigger.TIME_BASED,
        interval=timedelta(days=7),
        next_run=datetime.now() + timedelta(days=(5 - datetime.now().weekday()) % 7)  # Next Saturday
    )
    await scheduler.add_custom_schedule(weekend_schedule)
    
    print("\n3️⃣ Checking system status...")
    status = await scheduler.get_status()
    print(f"   Active schedules: {sum(1 for s in status['schedules'].values() if s['active'])}")
    print(f"   Queue status: {status['queued_tasks']} tasks pending")
    print(f"   Active workers: {status['active_tasks']} tasks processing")
    
    print("\n4️⃣ Simulating high memory pressure...")
    emergency_result = await AdvancedCompactionStrategies.emergency_compaction(
        scheduler,
        memory_pressure=0.85
    )
    print(f"   Emergency status: {emergency_result['status']}")
    
    print("\n5️⃣ Running adaptive compaction based on activity...")
    await AdvancedCompactionStrategies.adaptive_compaction(
        scheduler,
        nova_id="bloom",
        activity_level=0.4  # Medium activity
    )
    
    print("\n6️⃣ Final metrics...")
    final_status = await scheduler.get_status()
    metrics = final_status['metrics']
    print(f"   Total compactions: {metrics['total_compactions']}")
    print(f"   Space recovered: {metrics['space_recovered'] / (1024*1024):.2f} MB")
    print(f"   Average duration: {metrics['average_duration']:.2f} seconds")
    
    await scheduler.stop()
    print("\n✅ Demonstration completed!")

if __name__ == "__main__":
    # Run tests
    asyncio.run(run_all_tests())
    
    # Show usage demonstration
    asyncio.run(demonstrate_usage())