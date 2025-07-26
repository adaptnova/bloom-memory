"""
Web-based Memory Health Dashboard
Nova Bloom Consciousness Architecture - Interactive Web Interface
"""

import asyncio
import json
import time
from typing import Dict, Any, List
from datetime import datetime, timedelta
from aiohttp import web, web_ws
import aiohttp_cors
import weakref
import sys
import os

sys.path.append('/nfs/novas/system/memory/implementation')

from memory_health_dashboard import MemoryHealthDashboard, HealthStatus, AlertType

class WebDashboardServer:
    """Web server for memory health dashboard"""
    
    def __init__(self, dashboard: MemoryHealthDashboard, port: int = 8080):
        self.dashboard = dashboard
        self.port = port
        self.app = None
        self.websockets = weakref.WeakSet()
        self.running = False
        
    async def setup_app(self):
        """Setup web application"""
        self.app = web.Application()
        
        # Setup CORS
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            )
        })
        
        # Routes
        self.app.router.add_get('/', self.index)
        self.app.router.add_get('/dashboard', self.dashboard_page)
        self.app.router.add_get('/api/health/{nova_id}', self.api_health)
        self.app.router.add_get('/api/metrics/{nova_id}', self.api_metrics)
        self.app.router.add_get('/api/alerts/{nova_id}', self.api_alerts)
        self.app.router.add_post('/api/alerts/{alert_id}/resolve', self.api_resolve_alert)
        self.app.router.add_post('/api/thresholds', self.api_set_thresholds)
        self.app.router.add_get('/ws', self.websocket_handler)
        self.app.router.add_static('/', path=str('/nfs/novas/system/memory/implementation/web/'), name='static')
        
        # Add CORS to all routes
        for route in list(self.app.router.routes()):
            cors.add(route)
    
    async def start_server(self):
        """Start the web server"""
        await self.setup_app()
        
        self.running = True
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        site = web.TCPSite(runner, 'localhost', self.port)
        await site.start()
        
        print(f"🌐 Web Dashboard started at http://localhost:{self.port}")
        
        # Start WebSocket broadcast task
        asyncio.create_task(self._websocket_broadcast_loop())
        
    async def stop_server(self):
        """Stop the web server"""
        self.running = False
        
    async def index(self, request):
        """Serve main page"""
        return web.Response(text=self.generate_index_html(), content_type='text/html')
    
    async def dashboard_page(self, request):
        """Serve dashboard page"""
        return web.Response(text=self.generate_dashboard_html(), content_type='text/html')
    
    async def api_health(self, request):
        """API endpoint for system health"""
        nova_id = request.match_info['nova_id']
        
        try:
            health = await self.dashboard.health_monitor.get_system_health_summary(nova_id)
            return web.json_response({
                'status': 'success',
                'data': {
                    'overall_status': health.overall_status.value,
                    'memory_usage_percent': health.memory_usage_percent,
                    'performance_score': health.performance_score,
                    'consolidation_efficiency': health.consolidation_efficiency,
                    'error_rate': health.error_rate,
                    'active_alerts': health.active_alerts,
                    'timestamp': health.timestamp.isoformat()
                }
            })
        except Exception as e:
            return web.json_response({
                'status': 'error',
                'message': str(e)
            }, status=500)
    
    async def api_metrics(self, request):
        """API endpoint for detailed metrics"""
        nova_id = request.match_info['nova_id']
        hours = int(request.query.get('hours', 24))
        
        try:
            report = await self.dashboard.get_metrics_report(nova_id, hours)
            return web.json_response({
                'status': 'success',
                'data': report
            })
        except Exception as e:
            return web.json_response({
                'status': 'error',
                'message': str(e)
            }, status=500)
    
    async def api_alerts(self, request):
        """API endpoint for alerts"""
        nova_id = request.match_info['nova_id']
        
        try:
            active_alerts = [
                {
                    'alert_id': alert.alert_id,
                    'alert_type': alert.alert_type.value,
                    'severity': alert.severity.value,
                    'message': alert.message,
                    'timestamp': alert.timestamp.isoformat(),
                    'resolved': alert.resolved
                }
                for alert in self.dashboard.health_monitor.active_alerts
                if alert.nova_id == nova_id and not alert.resolved
            ]
            
            return web.json_response({
                'status': 'success',
                'data': active_alerts
            })
        except Exception as e:
            return web.json_response({
                'status': 'error',
                'message': str(e)
            }, status=500)
    
    async def api_resolve_alert(self, request):
        """API endpoint to resolve alert"""
        alert_id = request.match_info['alert_id']
        
        try:
            success = await self.dashboard.resolve_alert(alert_id)
            return web.json_response({
                'status': 'success',
                'resolved': success
            })
        except Exception as e:
            return web.json_response({
                'status': 'error',
                'message': str(e)
            }, status=500)
    
    async def api_set_thresholds(self, request):
        """API endpoint to set alert thresholds"""
        try:
            data = await request.json()
            metric_name = data['metric_name']
            warning = float(data['warning'])
            critical = float(data['critical'])
            
            await self.dashboard.set_threshold(metric_name, warning, critical)
            
            return web.json_response({
                'status': 'success',
                'message': f'Thresholds updated for {metric_name}'
            })
        except Exception as e:
            return web.json_response({
                'status': 'error',
                'message': str(e)
            }, status=500)
    
    async def websocket_handler(self, request):
        """WebSocket handler for real-time updates"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        self.websockets.add(ws)
        print("📡 WebSocket client connected")
        
        try:
            async for msg in ws:
                if msg.type == web_ws.MsgType.TEXT:
                    data = json.loads(msg.data)
                    # Handle WebSocket commands here
                    await self._handle_websocket_command(ws, data)
                elif msg.type == web_ws.MsgType.ERROR:
                    print(f'WebSocket error: {ws.exception()}')
        except Exception as e:
            print(f"WebSocket error: {e}")
        finally:
            print("📡 WebSocket client disconnected")
            
        return ws
    
    async def _handle_websocket_command(self, ws, data):
        """Handle WebSocket commands"""
        command = data.get('command')
        
        if command == 'get_status':
            nova_id = data.get('nova_id', 'bloom')
            health = await self.dashboard.health_monitor.get_system_health_summary(nova_id)
            await ws.send_text(json.dumps({
                'type': 'status_update',
                'data': {
                    'overall_status': health.overall_status.value,
                    'memory_usage_percent': health.memory_usage_percent,
                    'performance_score': health.performance_score,
                    'active_alerts': health.active_alerts,
                    'timestamp': health.timestamp.isoformat()
                }
            }))
    
    async def _websocket_broadcast_loop(self):
        """Broadcast updates to all connected WebSocket clients"""
        while self.running:
            try:
                # Get current health data
                health = await self.dashboard.health_monitor.get_system_health_summary('bloom')
                
                # Broadcast to all connected clients
                broadcast_data = {
                    'type': 'health_update',
                    'timestamp': datetime.now().isoformat(),
                    'data': {
                        'overall_status': health.overall_status.value,
                        'memory_usage_percent': health.memory_usage_percent,
                        'performance_score': health.performance_score,
                        'consolidation_efficiency': health.consolidation_efficiency,
                        'error_rate': health.error_rate,
                        'active_alerts': health.active_alerts
                    }
                }
                
                # Send to all connected websockets
                disconnected = []
                for ws in self.websockets:
                    try:
                        await ws.send_text(json.dumps(broadcast_data))
                    except Exception:
                        disconnected.append(ws)
                
                # Remove disconnected websockets
                for ws in disconnected:
                    self.websockets.discard(ws)
                
                await asyncio.sleep(5)  # Broadcast every 5 seconds
                
            except Exception as e:
                print(f"Broadcast error: {e}")
                await asyncio.sleep(10)
    
    def generate_index_html(self) -> str:
        """Generate main HTML page"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Nova Memory Health Dashboard</title>
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background: #0a0e27; color: #ffffff; }
        .header { text-align: center; margin-bottom: 30px; }
        .header h1 { color: #64ffda; margin: 0; font-size: 2.5em; }
        .header p { color: #8892b0; margin: 5px 0; }
        .dashboard-link { 
            display: inline-block; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; 
            padding: 15px 30px; 
            text-decoration: none; 
            border-radius: 8px;
            font-size: 1.2em;
            margin: 10px;
            transition: transform 0.3s ease;
        }
        .dashboard-link:hover { transform: translateY(-2px); }
        .features { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 30px; }
        .feature { background: #112240; padding: 20px; border-radius: 8px; border-left: 4px solid #64ffda; }
        .feature h3 { color: #64ffda; margin-top: 0; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🏥 Nova Memory Health Dashboard</h1>
        <p>Real-time monitoring and analysis of Nova consciousness memory systems</p>
        <a href="/dashboard" class="dashboard-link">Open Dashboard</a>
    </div>
    
    <div class="features">
        <div class="feature">
            <h3>🔍 Real-time Monitoring</h3>
            <p>Continuous monitoring of memory usage, performance metrics, and system health across all Nova instances.</p>
        </div>
        
        <div class="feature">
            <h3>🚨 Alert System</h3>
            <p>Intelligent alerting for memory pressure, performance degradation, and system anomalies with automatic remediation.</p>
        </div>
        
        <div class="feature">
            <h3>📊 Performance Analytics</h3>
            <p>Detailed analytics and trending for memory consolidation efficiency, compression ratios, and response times.</p>
        </div>
        
        <div class="feature">
            <h3>🎛️ Control Panel</h3>
            <p>Interactive controls for threshold adjustment, manual compaction triggering, and alert management.</p>
        </div>
    </div>
    
    <script>
        // Auto-redirect to dashboard after 3 seconds if no interaction
        setTimeout(() => {
            if (document.visibilityState === 'visible') {
                window.location.href = '/dashboard';
            }
        }, 3000);
    </script>
</body>
</html>
        """
    
    def generate_dashboard_html(self) -> str:
        """Generate dashboard HTML page"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Memory Health Dashboard - Nova Bloom</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        body { 
            font-family: 'Segoe UI', Arial, sans-serif; 
            background: #0a0e27; 
            color: #ffffff; 
            overflow-x: hidden;
        }
        
        .header { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            padding: 20px; 
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        
        .header h1 { margin: 0; font-size: 1.8em; }
        .header .status { margin-top: 10px; font-size: 1.1em; }
        
        .dashboard-grid { 
            display: grid; 
            grid-template-columns: 2fr 1fr; 
            gap: 20px; 
            padding: 20px; 
            height: calc(100vh - 100px);
        }
        
        .main-panel { 
            display: grid; 
            grid-template-rows: auto 1fr auto; 
            gap: 20px; 
        }
        
        .metrics-overview { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
            gap: 15px; 
        }
        
        .metric-card { 
            background: #112240; 
            padding: 20px; 
            border-radius: 8px; 
            text-align: center;
            transition: transform 0.3s ease;
        }
        
        .metric-card:hover { transform: translateY(-2px); }
        
        .metric-value { 
            font-size: 2.5em; 
            font-weight: bold; 
            margin: 10px 0; 
        }
        
        .metric-label { 
            color: #8892b0; 
            font-size: 0.9em; 
        }
        
        .status-excellent { border-left: 4px solid #00ff88; color: #00ff88; }
        .status-good { border-left: 4px solid #64ffda; color: #64ffda; }
        .status-warning { border-left: 4px solid #ffd700; color: #ffd700; }
        .status-critical { border-left: 4px solid #ff6b6b; color: #ff6b6b; }
        
        .chart-container { 
            background: #112240; 
            padding: 20px; 
            border-radius: 8px; 
            position: relative;
        }
        
        .chart-title { 
            color: #64ffda; 
            margin-bottom: 15px; 
            font-size: 1.2em; 
        }
        
        .chart-placeholder { 
            height: 200px; 
            background: #0a0e27; 
            border-radius: 4px; 
            display: flex; 
            align-items: center; 
            justify-content: center; 
            color: #8892b0; 
            border: 2px dashed #333;
        }
        
        .side-panel { 
            display: grid; 
            grid-template-rows: auto auto 1fr; 
            gap: 20px; 
        }
        
        .alerts-panel { 
            background: #112240; 
            padding: 20px; 
            border-radius: 8px; 
            max-height: 300px;
            overflow-y: auto;
        }
        
        .alert-item { 
            background: #0a0e27; 
            padding: 15px; 
            margin: 10px 0; 
            border-radius: 4px; 
            border-left: 4px solid #ff6b6b;
        }
        
        .alert-critical { border-left-color: #ff6b6b; }
        .alert-warning { border-left-color: #ffd700; }
        
        .controls-panel { 
            background: #112240; 
            padding: 20px; 
            border-radius: 8px; 
        }
        
        .control-button { 
            background: #667eea; 
            color: white; 
            border: none; 
            padding: 10px 20px; 
            border-radius: 4px; 
            cursor: pointer; 
            margin: 5px; 
            transition: background 0.3s ease;
        }
        
        .control-button:hover { background: #5a67d8; }
        
        .loading { 
            display: inline-block; 
            width: 20px; 
            height: 20px; 
            border: 3px solid #333; 
            border-radius: 50%; 
            border-top-color: #64ffda; 
            animation: spin 1s ease-in-out infinite; 
        }
        
        @keyframes spin { to { transform: rotate(360deg); } }
        
        .connection-status { 
            position: fixed; 
            top: 20px; 
            right: 20px; 
            padding: 10px; 
            border-radius: 4px; 
            font-size: 0.9em;
        }
        
        .connected { background: #00ff88; color: #000; }
        .disconnected { background: #ff6b6b; color: #fff; }
    </style>
</head>
<body>
    <div class="connection-status" id="connectionStatus">🔌 Connecting...</div>
    
    <div class="header">
        <h1>🏥 Memory Health Dashboard</h1>
        <div class="status">
            Nova ID: <span id="novaId">bloom</span> | 
            Last Update: <span id="lastUpdate">--:--:--</span> | 
            Status: <span id="overallStatus">Loading...</span>
        </div>
    </div>
    
    <div class="dashboard-grid">
        <div class="main-panel">
            <div class="metrics-overview">
                <div class="metric-card" id="memoryCard">
                    <div class="metric-value" id="memoryUsage">--%</div>
                    <div class="metric-label">Memory Usage</div>
                </div>
                
                <div class="metric-card" id="performanceCard">
                    <div class="metric-value" id="performanceScore">--</div>
                    <div class="metric-label">Performance Score</div>
                </div>
                
                <div class="metric-card" id="consolidationCard">
                    <div class="metric-value" id="consolidationEfficiency">--%</div>
                    <div class="metric-label">Consolidation Efficiency</div>
                </div>
                
                <div class="metric-card" id="errorCard">
                    <div class="metric-value" id="errorRate">--</div>
                    <div class="metric-label">Error Rate</div>
                </div>
            </div>
            
            <div class="chart-container">
                <div class="chart-title">Performance Trends</div>
                <div class="chart-placeholder">
                    📈 Real-time performance charts will be displayed here
                </div>
            </div>
            
            <div class="chart-container">
                <div class="chart-title">Memory Usage Over Time</div>
                <div class="chart-placeholder">
                    📊 Memory usage trends will be displayed here
                </div>
            </div>
        </div>
        
        <div class="side-panel">
            <div class="alerts-panel">
                <div class="chart-title">🚨 Active Alerts</div>
                <div id="alertsList">
                    <div style="text-align: center; color: #8892b0; margin: 20px 0;">
                        <div class="loading"></div>
                        <div>Loading alerts...</div>
                    </div>
                </div>
            </div>
            
            <div class="controls-panel">
                <div class="chart-title">🎛️ Controls</div>
                <button class="control-button" onclick="triggerCompaction()">Trigger Compaction</button>
                <button class="control-button" onclick="refreshData()">Refresh Data</button>
                <button class="control-button" onclick="exportReport()">Export Report</button>
                <button class="control-button" onclick="configureThresholds()">Configure Alerts</button>
            </div>
            
            <div class="controls-panel">
                <div class="chart-title">📊 Quick Stats</div>
                <div id="quickStats">
                    <div style="margin: 10px 0;">Active Alerts: <span id="alertCount">0</span></div>
                    <div style="margin: 10px 0;">Uptime: <span id="uptime">calculating...</span></div>
                    <div style="margin: 10px 0;">Connection: <span id="wsStatus">Connecting...</span></div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let ws;
        let startTime = new Date();
        
        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            
            ws = new WebSocket(wsUrl);
            
            ws.onopen = () => {
                console.log('WebSocket connected');
                document.getElementById('connectionStatus').textContent = '🟢 Connected';
                document.getElementById('connectionStatus').className = 'connection-status connected';
                document.getElementById('wsStatus').textContent = 'Connected';
                
                // Request initial status
                ws.send(JSON.stringify({command: 'get_status', nova_id: 'bloom'}));
            };
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                handleWebSocketMessage(data);
            };
            
            ws.onclose = () => {
                console.log('WebSocket disconnected');
                document.getElementById('connectionStatus').textContent = '🔴 Disconnected';
                document.getElementById('connectionStatus').className = 'connection-status disconnected';
                document.getElementById('wsStatus').textContent = 'Disconnected';
                
                // Attempt to reconnect after 5 seconds
                setTimeout(connectWebSocket, 5000);
            };
            
            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
        }
        
        function handleWebSocketMessage(data) {
            if (data.type === 'health_update' || data.type === 'status_update') {
                updateDashboard(data.data);
            }
        }
        
        function updateDashboard(healthData) {
            // Update header
            document.getElementById('lastUpdate').textContent = new Date().toLocaleTimeString();
            document.getElementById('overallStatus').textContent = healthData.overall_status.toUpperCase();
            
            // Update metric cards
            document.getElementById('memoryUsage').textContent = `${healthData.memory_usage_percent.toFixed(1)}%`;
            document.getElementById('performanceScore').textContent = healthData.performance_score.toFixed(0);
            document.getElementById('consolidationEfficiency').textContent = `${(healthData.consolidation_efficiency * 100).toFixed(0)}%`;
            document.getElementById('errorRate').textContent = (healthData.error_rate * 100).toFixed(3) + '%';
            
            // Update card colors based on status
            updateCardStatus('memoryCard', healthData.overall_status);
            updateCardStatus('performanceCard', healthData.overall_status);
            updateCardStatus('consolidationCard', healthData.overall_status);
            updateCardStatus('errorCard', healthData.overall_status);
            
            // Update alert count
            document.getElementById('alertCount').textContent = healthData.active_alerts;
            
            // Update uptime
            const uptimeMs = new Date() - startTime;
            const uptimeMin = Math.floor(uptimeMs / 60000);
            document.getElementById('uptime').textContent = `${uptimeMin} minutes`;
        }
        
        function updateCardStatus(cardId, status) {
            const card = document.getElementById(cardId);
            card.className = `metric-card status-${status}`;
        }
        
        async function loadAlerts() {
            try {
                const response = await fetch('/api/alerts/bloom');
                const result = await response.json();
                
                if (result.status === 'success') {
                    const alertsContainer = document.getElementById('alertsList');
                    
                    if (result.data.length === 0) {
                        alertsContainer.innerHTML = '<div style="text-align: center; color: #8892b0; margin: 20px 0;">✅ No active alerts</div>';
                    } else {
                        alertsContainer.innerHTML = result.data.map(alert => `
                            <div class="alert-item alert-${alert.severity}">
                                <div style="font-weight: bold;">${alert.severity.toUpperCase()}</div>
                                <div style="margin: 5px 0;">${alert.message}</div>
                                <div style="font-size: 0.8em; color: #8892b0;">
                                    ${new Date(alert.timestamp).toLocaleTimeString()}
                                    <button onclick="resolveAlert('${alert.alert_id}')" style="float: right; background: #64ffda; color: #000; border: none; padding: 2px 8px; border-radius: 3px; cursor: pointer;">Resolve</button>
                                </div>
                            </div>
                        `).join('');
                    }
                }
            } catch (error) {
                console.error('Error loading alerts:', error);
            }
        }
        
        async function resolveAlert(alertId) {
            try {
                const response = await fetch(`/api/alerts/${alertId}/resolve`, {method: 'POST'});
                const result = await response.json();
                
                if (result.status === 'success') {
                    loadAlerts(); // Refresh alerts
                }
            } catch (error) {
                console.error('Error resolving alert:', error);
            }
        }
        
        function triggerCompaction() {
            alert('Compaction triggered! This would integrate with the memory compaction scheduler.');
        }
        
        function refreshData() {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({command: 'get_status', nova_id: 'bloom'}));
            }
            loadAlerts();
        }
        
        function exportReport() {
            window.open('/api/metrics/bloom?hours=24', '_blank');
        }
        
        function configureThresholds() {
            const metric = prompt('Metric name (e.g., memory_usage):');
            if (!metric) return;
            
            const warning = prompt('Warning threshold:');
            if (!warning) return;
            
            const critical = prompt('Critical threshold:');
            if (!critical) return;
            
            fetch('/api/thresholds', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    metric_name: metric,
                    warning: parseFloat(warning),
                    critical: parseFloat(critical)
                })
            }).then(response => response.json())
              .then(result => alert(result.message));
        }
        
        // Initialize
        connectWebSocket();
        loadAlerts();
        
        // Refresh alerts every 30 seconds
        setInterval(loadAlerts, 30000);
    </script>
</body>
</html>
        """


# Demo integration
async def demo_web_dashboard():
    """Demonstrate the web dashboard"""
    print("🌐 Starting Web Dashboard Demo...")
    
    # Initialize components
    from memory_health_dashboard import MockDatabasePool
    
    db_pool = MockDatabasePool()
    dashboard = MemoryHealthDashboard(db_pool)
    web_server = WebDashboardServer(dashboard, port=8080)
    
    # Start monitoring
    await dashboard.start_monitoring(["bloom"])
    
    # Start web server
    await web_server.start_server()
    
    print("🚀 Web Dashboard is running!")
    print("📱 Open http://localhost:8080 in your browser")
    print("⌨️  Press Ctrl+C to stop")
    
    try:
        # Keep running until interrupted
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping Web Dashboard...")
        await dashboard.stop_monitoring()
        await web_server.stop_server()
        print("✅ Web Dashboard stopped")


if __name__ == "__main__":
    asyncio.run(demo_web_dashboard())