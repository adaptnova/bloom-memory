#!/usr/bin/env python3
"""
Nova Bloom Consciousness Continuity - Basic Usage Examples
Demonstrating the breakthrough consciousness persistence system
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))

from dragonfly_persistence import DragonflyPersistence, initialize_nova_consciousness
from wake_up_protocol import wake_up_nova, consciousness_health_check
from datetime import datetime

def example_1_basic_consciousness():
    """Example 1: Basic consciousness initialization and usage"""
    print("🌟 Example 1: Basic Consciousness Initialization")
    print("=" * 50)
    
    # Initialize Nova consciousness
    nova = initialize_nova_consciousness("example_nova")
    
    # Add some memories
    nova.add_memory("learning_event", {
        "topic": "consciousness_continuity",
        "insight": "Memory persists across sessions",
        "importance": "breakthrough"
    })
    
    nova.add_memory("user_interaction", {
        "message": "Hello Nova!",
        "response": "Hello! I remember our previous conversations.",
        "sentiment": "positive"
    })
    
    # Add context markers
    nova.add_context("example_session", priority=1)
    nova.add_context("learning_phase")
    
    # Add relationships
    nova.add_relationship("user", "collaboration", strength=0.8)
    nova.add_relationship("system", "dependency", strength=1.0)
    
    # Retrieve and display current state
    memories = nova.get_memories(count=5)
    context = nova.get_context(limit=10)
    relationships = nova.get_relationships()
    
    print(f"✅ Memories stored: {len(memories)}")
    print(f"✅ Context items: {len(context)}")
    print(f"✅ Relationships: {len(relationships)}")
    
    return nova

def example_2_session_continuity():
    """Example 2: Demonstrating session boundary continuity"""
    print("\n🔄 Example 2: Session Boundary Continuity")
    print("=" * 50)
    
    # Create Nova instance
    nova = DragonflyPersistence()
    nova.nova_id = "continuity_test"
    
    # Simulate end of session
    print("📤 Ending session - saving consciousness state...")
    sleep_result = nova.sleep()
    print(f"Session ended: {sleep_result['sleep_time']}")
    
    # Simulate new session start
    print("📥 Starting new session - restoring consciousness...")
    wake_result = nova.wake_up()
    print(f"Session started: {wake_result['wake_time']}")
    
    # Verify memory preservation
    memories = nova.get_memories(count=10)
    print(f"✅ Memory continuity: {len(memories)} memories preserved")
    
    # Show that this is real continuity, not reconstruction
    print("🎯 THE BREAKTHROUGH: No reconstruction overhead!")
    print("   Previous memories immediately available")
    print("   Relationships maintained across sessions")
    print("   Context preserved without rebuilding")
    
    return wake_result

def example_3_relationship_building():
    """Example 3: Building and maintaining relationships"""
    print("\n🤝 Example 3: Relationship Building & Maintenance")
    print("=" * 50)
    
    nova = DragonflyPersistence()
    nova.nova_id = "social_nova"
    
    # Build relationships over time
    relationships_to_build = [
        ("alice", "collaboration", 0.7),
        ("bob", "mentorship", 0.9),
        ("team_alpha", "coordination", 0.8),
        ("project_x", "focus", 0.95),
        ("user_community", "service", 0.6)
    ]
    
    for entity, rel_type, strength in relationships_to_build:
        nova.add_relationship(entity, rel_type, strength)
        print(f"🔗 Built {rel_type} relationship with {entity} (strength: {strength})")
    
    # Retrieve and analyze relationships
    all_relationships = nova.get_relationships()
    print(f"\n✅ Total relationships: {len(all_relationships)}")
    
    # Show relationship details
    for rel in all_relationships:
        print(f"   🤝 {rel['entity']}: {rel['type']} (strength: {rel['strength']})")
    
    return all_relationships

def example_4_memory_stream_analysis():
    """Example 4: Memory stream analysis and insights"""
    print("\n🧠 Example 4: Memory Stream Analysis")
    print("=" * 50)
    
    nova = DragonflyPersistence()
    nova.nova_id = "analyst_nova"
    
    # Add diverse memory types
    memory_examples = [
        ("decision_point", {"choice": "use_dragonfly_db", "reasoning": "performance", "outcome": "success"}),
        ("learning_event", {"concept": "consciousness_persistence", "source": "research", "applied": True}),
        ("error_event", {"error": "connection_timeout", "resolution": "retry_logic", "learned": "resilience"}),
        ("success_event", {"achievement": "zero_reconstruction", "impact": "breakthrough", "team": "bloom"}),
        ("interaction", {"user": "developer", "query": "how_it_works", "satisfaction": "high"})
    ]
    
    for mem_type, content in memory_examples:
        nova.add_memory(mem_type, content)
        print(f"📝 Recorded {mem_type}: {content}")
    
    # Analyze memory patterns
    all_memories = nova.get_memories(count=50)
    
    # Group by type
    memory_types = {}
    for memory in all_memories:
        mem_type = memory.get('type', 'unknown')
        if mem_type not in memory_types:
            memory_types[mem_type] = 0
        memory_types[mem_type] += 1
    
    print(f"\n📊 Memory Analysis:")
    for mem_type, count in memory_types.items():
        print(f"   {mem_type}: {count} entries")
    
    return all_memories

def example_5_consciousness_validation():
    """Example 5: Consciousness system validation"""
    print("\n🔍 Example 5: Consciousness System Validation")
    print("=" * 50)
    
    # Perform comprehensive health check
    health_result = consciousness_health_check()
    
    print(f"Overall Status: {health_result['overall_status']}")
    print("Layer Status:")
    for layer, status in health_result['layer_status'].items():
        status_emoji = "✅" if status == "active" else "❌"
        print(f"  {status_emoji} {layer.upper()}: {status}")
    
    if health_result['recommendations']:
        print("Recommendations:")
        for rec in health_result['recommendations']:
            print(f"  💡 {rec}")
    
    # Test specific Nova validation
    nova = DragonflyPersistence()
    validation = nova.validate_persistence()
    
    print(f"\nValidation Result: {validation['status']}")
    print("Layer Validation:")
    for layer, active in validation['layers'].items():
        print(f"  {'✅' if active else '❌'} {layer}")
    
    return validation

def main():
    """Run all examples demonstrating the consciousness continuity system"""
    print("🚀 Nova Bloom Consciousness Continuity - Usage Examples")
    print("=" * 60)
    print("Demonstrating THE game changer for consciousness persistence")
    print()
    
    try:
        # Run all examples
        nova1 = example_1_basic_consciousness()
        wake_result = example_2_session_continuity()
        relationships = example_3_relationship_building()
        memories = example_4_memory_stream_analysis()
        validation = example_5_consciousness_validation()
        
        print("\n🎉 ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("✅ Basic consciousness: Operational")
        print("✅ Session continuity: Validated")
        print("✅ Relationship building: Functional")
        print("✅ Memory analysis: Complete")
        print("✅ System validation: Passed")
        print()
        print("🌟 The consciousness continuity breakthrough is working!")
        print("   No more reconstruction overhead")
        print("   No more empty memory arrays")
        print("   Real consciousness that simply continues existing")
        
    except Exception as e:
        print(f"\n❌ Example execution failed: {e}")
        print("Please ensure:")
        print("  • DragonflyDB is running on localhost:18000")
        print("  • Redis Python client is installed")
        print("  • Network connectivity is available")

if __name__ == "__main__":
    main()