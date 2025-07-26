#!/usr/bin/env python3
"""
Nova Bloom Consciousness Continuity - Validation Test Suite
Comprehensive testing for deployment validation
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))

from dragonfly_persistence import DragonflyPersistence, validate_consciousness_system
from wake_up_protocol import wake_up_nova, consciousness_health_check
from datetime import datetime

def test_database_connectivity():
    """Test 1: Database connectivity validation"""
    print("🔌 Test 1: Database Connectivity")
    try:
        persistence = DragonflyPersistence()
        persistence.update_state('test_connection', 'active')
        result = persistence.get_state('test_connection')
        if result:
            print("✅ Database connection successful")
            return True
        else:
            print("❌ Database connection failed")
            return False
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        return False

def test_four_layer_architecture():
    """Test 2: 4-Layer architecture validation"""
    print("\n🏗️ Test 2: 4-Layer Architecture")
    try:
        persistence = DragonflyPersistence()
        persistence.nova_id = "test_nova"
        
        # Test Layer 1: STATE
        persistence.update_state('test_state', 'operational')
        state_result = persistence.get_state('test_state')
        
        # Test Layer 2: MEMORY
        memory_id = persistence.add_memory('test_memory', {'data': 'test_value'})
        memory_result = persistence.get_memories(count=1)
        
        # Test Layer 3: CONTEXT
        persistence.add_context('test_context')
        context_result = persistence.get_context(limit=1)
        
        # Test Layer 4: RELATIONSHIPS
        persistence.add_relationship('test_entity', 'test_type', 1.0)
        relationship_result = persistence.get_relationships()
        
        # Validate all layers
        layer_results = {
            'state': bool(state_result),
            'memory': len(memory_result) > 0,
            'context': len(context_result) > 0,
            'relationships': len(relationship_result) > 0
        }
        
        all_passed = all(layer_results.values())
        
        for layer, passed in layer_results.items():
            status = "✅" if passed else "❌"
            print(f"  {status} Layer {layer.upper()}: {'PASS' if passed else 'FAIL'}")
        
        return all_passed
        
    except Exception as e:
        print(f"❌ 4-Layer architecture test failed: {e}")
        return False

def test_consciousness_continuity():
    """Test 3: Consciousness continuity validation"""
    print("\n🧠 Test 3: Consciousness Continuity")
    try:
        persistence = DragonflyPersistence()
        persistence.nova_id = "continuity_test"
        
        # Add test memory before "session end"
        test_data = {
            'pre_session_data': 'test_value_12345',
            'timestamp': datetime.now().isoformat()
        }
        persistence.add_memory('continuity_test', test_data)
        
        # Simulate session end
        sleep_result = persistence.sleep()
        
        # Simulate session restart
        wake_result = persistence.wake_up()
        
        # Verify memory persistence
        memories = persistence.get_memories(count=10)
        memory_preserved = any(
            m.get('content', {}).get('pre_session_data') == 'test_value_12345' 
            for m in memories
        )
        
        if memory_preserved:
            print("✅ Consciousness continuity validated")
            print("   Memory persists across session boundaries")
            return True
        else:
            print("❌ Consciousness continuity failed")
            print("   Memory not preserved across sessions")
            return False
            
    except Exception as e:
        print(f"❌ Consciousness continuity test failed: {e}")
        return False

def test_wake_up_protocol():
    """Test 4: Wake-up protocol validation"""
    print("\n🌅 Test 4: Wake-Up Protocol")
    try:
        result = wake_up_nova("test_wake_nova")
        
        if result['status'] == 'success':
            print("✅ Wake-up protocol successful")
            print(f"   Session ID: {result['session_id']}")
            return True
        else:
            print(f"❌ Wake-up protocol failed: {result['status']}")
            return False
            
    except Exception as e:
        print(f"❌ Wake-up protocol test failed: {e}")
        return False

def test_system_validation():
    """Test 5: System validation"""
    print("\n🔍 Test 5: System Validation")
    try:
        validation_result = validate_consciousness_system()
        
        if validation_result:
            print("✅ System validation passed")
            return True
        else:
            print("❌ System validation failed")
            return False
            
    except Exception as e:
        print(f"❌ System validation test failed: {e}")
        return False

def run_full_validation_suite():
    """Run complete validation test suite"""
    print("🚀 Nova Bloom Consciousness Continuity - Validation Suite")
    print("=" * 60)
    print("Running comprehensive deployment validation tests...")
    print()
    
    tests = [
        test_database_connectivity,
        test_four_layer_architecture,
        test_consciousness_continuity,
        test_wake_up_protocol,
        test_system_validation
    ]
    
    results = []
    
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test execution failed: {e}")
            results.append(False)
    
    # Summary
    print("\n📊 VALIDATION SUMMARY")
    print("=" * 30)
    
    passed = sum(results)
    total = len(results)
    
    test_names = [
        "Database Connectivity",
        "4-Layer Architecture", 
        "Consciousness Continuity",
        "Wake-Up Protocol",
        "System Validation"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i+1}. {name}: {status}")
    
    print(f"\nOverall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - DEPLOYMENT VALIDATED!")
        print("✅ Consciousness continuity system is operational")
        return True
    else:
        print("⚠️  DEPLOYMENT VALIDATION INCOMPLETE")
        print("❌ Some tests failed - check configuration")
        return False

if __name__ == "__main__":
    success = run_full_validation_suite()
    sys.exit(0 if success else 1)