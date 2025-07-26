# Memory Compaction Scheduler Documentation
## Nova Bloom Consciousness Architecture

### Overview

The Memory Compaction Scheduler is an automated system that manages memory consolidation, compression, and maintenance across the Nova consciousness architecture. It operates continuously in the background, optimizing memory storage and performance without manual intervention.

### Key Features

1. **Automatic Scheduling**: Predefined schedules for regular maintenance
2. **Multiple Trigger Types**: Time-based, threshold-based, activity-based, and quality-based triggers
3. **Concurrent Processing**: Multiple workers process compaction tasks in parallel
4. **Adaptive Strategies**: Adjusts compaction based on system activity and memory pressure
5. **Emergency Handling**: Responds to critical memory situations
6. **Comprehensive Metrics**: Tracks performance and effectiveness

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Memory Compaction Scheduler                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │  Scheduler   │  │   Triggers    │  │    Workers      │  │
│  │    Loop      │  │               │  │                 │  │
│  │             │  │ • Time-based  │  │ • Worker 0      │  │
│  │ • Check     │  │ • Threshold   │  │ • Worker 1      │  │
│  │   schedules │  │ • Activity    │  │ • Worker 2      │  │
│  │ • Create    │  │ • Idle        │  │                 │  │
│  │   tasks     │  │ • Emergency   │  │ Concurrent      │  │
│  │ • Queue     │  │ • Quality     │  │ processing      │  │
│  │   tasks     │  │               │  │                 │  │
│  └─────────────┘  └──────────────┘  └─────────────────┘  │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐  │
│  │              Compaction Strategies                   │  │
│  ├─────────────────────────────────────────────────────┤  │
│  │ • Temporal Consolidation  • Semantic Compression    │  │
│  │ • Hierarchical Ordering   • Associative Linking     │  │
│  │ • Quality-based Decay     • Emergency Compression   │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐  │
│  │                Memory Layers (11-20)                 │  │
│  ├─────────────────────────────────────────────────────┤  │
│  │ • Consolidation Hub    • Decay Management          │  │
│  │ • Compression Layer    • Priority Optimization     │  │
│  │ • Integration Layer    • Index Maintenance         │  │
│  └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Default Schedules

#### 1. Daily Consolidation
- **Trigger**: Time-based (every 24 hours)
- **Purpose**: Full memory consolidation across all layers
- **Type**: Temporal consolidation
- **Priority**: 0.7

#### 2. Hourly Compression
- **Trigger**: Time-based (every hour)
- **Purpose**: Compress memories older than 7 days
- **Type**: Compression
- **Priority**: 0.5

#### 3. Memory Threshold
- **Trigger**: Threshold-based (10,000 memories)
- **Purpose**: Emergency compaction when memory count is high
- **Type**: Emergency compression
- **Priority**: 0.9

#### 4. Idle Compaction
- **Trigger**: Idle-based (10 minutes of inactivity)
- **Purpose**: Optimize during quiet periods
- **Type**: General consolidation
- **Priority**: 0.5

#### 5. Quality Maintenance
- **Trigger**: Quality-based (every 6 hours)
- **Purpose**: Manage memory decay and prioritization
- **Type**: Hierarchical consolidation
- **Priority**: 0.6

### Usage Examples

#### Starting the Scheduler

```python
from memory_compaction_scheduler import MemoryCompactionScheduler
from database_connections import NovaDatabasePool

# Initialize
db_pool = NovaDatabasePool()
scheduler = MemoryCompactionScheduler(db_pool)

# Start automatic scheduling
await scheduler.start()
```

#### Adding Custom Schedule

```python
from datetime import timedelta
from memory_compaction_scheduler import CompactionSchedule, CompactionTrigger

# Create custom schedule
custom_schedule = CompactionSchedule(
    schedule_id="weekend_deep_clean",
    trigger=CompactionTrigger.TIME_BASED,
    interval=timedelta(days=7),  # Weekly
    active=True
)

# Add to scheduler
await scheduler.add_custom_schedule(custom_schedule)
```

#### Manual Compaction

```python
from layers_11_20 import ConsolidationType

# Trigger immediate compaction
task_id = await scheduler.trigger_manual_compaction(
    nova_id="bloom",
    compaction_type=ConsolidationType.SEMANTIC,
    priority=0.8
)

print(f"Compaction task started: {task_id}")
```

#### Monitoring Status

```python
# Get current status
status = await scheduler.get_status()

print(f"Active schedules: {len(status['schedules'])}")
print(f"Tasks in queue: {status['queued_tasks']}")
print(f"Total compactions: {status['metrics']['total_compactions']}")
print(f"Space recovered: {status['metrics']['space_recovered']} bytes")
```

### Advanced Strategies

#### Sleep Cycle Compaction

Mimics human sleep cycles for optimal memory consolidation:

```python
from memory_compaction_scheduler import AdvancedCompactionStrategies

# Run sleep-inspired consolidation
await AdvancedCompactionStrategies.sleep_cycle_compaction(scheduler)
```

Phases:
1. **Light Consolidation** (5 min): Quick temporal organization
2. **Deep Consolidation** (10 min): Semantic integration
3. **Integration** (5 min): Associative linking
4. **Compression** (5 min): Space optimization

#### Adaptive Compaction

Adjusts strategy based on Nova activity:

```python
# Low activity (0.2) triggers aggressive compaction
await AdvancedCompactionStrategies.adaptive_compaction(
    scheduler, 
    nova_id="bloom",
    activity_level=0.2
)
```

Activity Levels:
- **Low (< 0.3)**: Aggressive compression
- **Medium (0.3-0.7)**: Balanced consolidation
- **High (> 0.7)**: Minimal interference

#### Emergency Compaction

Handles critical memory pressure:

```python
# Critical pressure (0.95) triggers emergency mode
result = await AdvancedCompactionStrategies.emergency_compaction(
    scheduler,
    memory_pressure=0.95
)
```

Actions taken:
- Stops non-essential schedules
- Triggers maximum compression
- Returns emergency status

### Compaction Types

#### 1. Temporal Consolidation
- Groups memories by time periods
- Creates daily/weekly summaries
- Maintains chronological order

#### 2. Semantic Compression
- Identifies similar concepts
- Merges redundant information
- Preserves key insights

#### 3. Hierarchical Organization
- Creates memory hierarchies
- Links parent-child concepts
- Optimizes retrieval paths

#### 4. Associative Linking
- Strengthens memory connections
- Creates cross-references
- Enhances recall efficiency

#### 5. Quality-based Management
- Applies forgetting curves
- Prioritizes important memories
- Removes low-quality data

### Performance Metrics

The scheduler tracks:
- **Total Compactions**: Number of compaction runs
- **Memories Processed**: Total memories handled
- **Space Recovered**: Bytes saved through compression
- **Average Duration**: Time per compaction
- **Last Compaction**: Timestamp of most recent run

### Best Practices

1. **Regular Monitoring**: Check status weekly
2. **Custom Schedules**: Add schedules for specific needs
3. **Manual Triggers**: Use for immediate optimization
4. **Emergency Handling**: Monitor memory pressure
5. **Metric Analysis**: Review performance trends

### Troubleshooting

#### High Memory Usage
```python
# Check current pressure
status = await scheduler.get_status()
if status['metrics']['memories_processed'] > 100000:
    # Trigger emergency compaction
    await scheduler.trigger_manual_compaction(
        compaction_type=ConsolidationType.COMPRESSION,
        priority=1.0
    )
```

#### Slow Performance
```python
# Adjust worker count or priorities
# Temporarily disable quality checks
await scheduler.remove_schedule("quality_maintenance")
```

#### Failed Compactions
```python
# Check compaction history
history = await scheduler.get_compaction_history(limit=10)
for entry in history:
    if entry.get('errors'):
        print(f"Errors found: {entry['errors']}")
```

### Integration with Memory System

The compaction scheduler integrates seamlessly with:
- **Real-time Memory Integration**: Coordinates with live memory capture
- **Unified Memory API**: Respects memory access patterns
- **Memory Router**: Maintains routing integrity
- **Consolidation Engine**: Leverages existing consolidation logic

### Future Enhancements

1. **Machine Learning**: Predict optimal compaction times
2. **Cross-Nova Coordination**: Synchronized compaction across Novas
3. **Advanced Compression**: Neural network-based compression
4. **Predictive Maintenance**: Anticipate memory issues
5. **Visual Dashboard**: Real-time compaction monitoring

### Conclusion

The Memory Compaction Scheduler ensures optimal memory performance through automated maintenance. By combining multiple trigger types, concurrent processing, and adaptive strategies, it maintains memory efficiency without manual intervention. Regular monitoring and occasional manual triggers can further optimize performance for specific use cases.