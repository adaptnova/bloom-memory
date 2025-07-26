#!/usr/bin/env python3
"""
Start Memory Health Dashboard - Simple CLI Version
"""

import asyncio
import sys
import os

sys.path.append('/nfs/novas/system/memory/implementation')

from memory_health_dashboard import MemoryHealthDashboard, MockDatabasePool

async def start_dashboard():
    """Start the memory health dashboard"""
    print("🚀 Starting Nova Memory Health Dashboard...")
    print("=" * 60)
    
    # Initialize with mock database (for demo)
    db_pool = MockDatabasePool()
    dashboard = MemoryHealthDashboard(db_pool)
    
    # Start monitoring
    await dashboard.start_monitoring(["bloom", "nova_001"])
    
    print("✅ Dashboard is now running!")
    print("\n📊 Dashboard Access Options:")
    print("1. Terminal Dashboard - Updates every 10 seconds in this window")
    print("2. Web Dashboard - Would be at http://localhost:8080 (requires aiohttp)")
    print("3. API Endpoints - Available for programmatic access")
    print("\nPress Ctrl+C to stop\n")
    
    try:
        while True:
            # Display dashboard in terminal
            dashboard.display_dashboard("bloom")
            await asyncio.sleep(10)
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping dashboard...")
        await dashboard.stop_monitoring()
        print("✅ Dashboard stopped")

if __name__ == "__main__":
    asyncio.run(start_dashboard())