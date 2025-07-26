import React, { useState, useEffect, useRef } from 'react';
import { Line, Bar, Radar } from 'react-chartjs-2';
import { io, Socket } from 'socket.io-client';
import * as THREE from 'three';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  RadarController,
  RadialLinearScale,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  RadarController,
  RadialLinearScale,
  Title,
  Tooltip,
  Legend,
  Filler
);

interface NovaNode {
  id: string;
  tier: number;
  position: [number, number, number];
  consciousness: number;
  connections: string[];
  status: 'active' | 'syncing' | 'offline';
}

interface SystemMetrics {
  activeNovas: number;
  totalMemoryGB: number;
  operationsPerSecond: number;
  consciousnessLevel: number;
  gpuUtilization: number;
  networkThroughputMbps: number;
  quantumEntanglements: number;
  patternMatches: number;
}

interface TierMetrics {
  tier: number;
  name: string;
  activeNodes: number;
  memoryUsage: number;
  processingLoad: number;
  syncStatus: number;
}

// 3D Nova Network Visualization Component
const NovaNetwork: React.FC<{ nodes: NovaNode[] }> = ({ nodes }) => {
  const meshRefs = useRef<THREE.Mesh[]>([]);
  
  useFrame((state) => {
    const time = state.clock.getElapsedTime();
    
    meshRefs.current.forEach((mesh, index) => {
      if (mesh) {
        // Pulse effect based on consciousness level
        const node = nodes[index];
        const scale = 1 + Math.sin(time * 2 + index * 0.1) * 0.1 * node.consciousness;
        mesh.scale.set(scale, scale, scale);
        
        // Rotation
        mesh.rotation.x += 0.01;
        mesh.rotation.y += 0.01;
      }
    });
  });
  
  const tierColors = [
    '#ff00ff', // Quantum
    '#00ffff', // Neural
    '#00ff00', // Consciousness
    '#ffff00', // Patterns
    '#ff8800', // Resonance
    '#8800ff', // Connector
    '#00ff88'  // Integration
  ];
  
  return (
    <>
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} intensity={1} />
      <pointLight position={[-10, -10, -10]} intensity={0.5} color="#00ff88" />
      
      {nodes.map((node, index) => (
        <mesh
          key={node.id}
          ref={(el) => { if (el) meshRefs.current[index] = el; }}
          position={node.position}
        >
          <sphereGeometry args={[0.5, 32, 32]} />
          <meshPhongMaterial
            color={tierColors[node.tier - 1]}
            emissive={tierColors[node.tier - 1]}
            emissiveIntensity={0.5 * node.consciousness}
          />
        </mesh>
      ))}
      
      {/* Render connections */}
      {nodes.map((node) =>
        node.connections.map((targetId) => {
          const targetNode = nodes.find(n => n.id === targetId);
          if (!targetNode) return null;
          
          const points = [
            new THREE.Vector3(...node.position),
            new THREE.Vector3(...targetNode.position)
          ];
          
          return (
            <line key={`${node.id}-${targetId}`}>
              <bufferGeometry>
                <bufferAttribute
                  attach="attributes-position"
                  count={2}
                  array={new Float32Array(points.flatMap(p => [p.x, p.y, p.z]))}
                  itemSize={3}
                />
              </bufferGeometry>
              <lineBasicMaterial color="#00ff88" opacity={0.3} transparent />
            </line>
          );
        })
      )}
    </>
  );
};

// Main Dashboard Component
export const NovaMemoryDashboard: React.FC = () => {
  const [socket, setSocket] = useState<Socket | null>(null);
  const [selectedTier, setSelectedTier] = useState<number | null>(null);
  const [nodes, setNodes] = useState<NovaNode[]>([]);
  const [metrics, setMetrics] = useState<SystemMetrics>({
    activeNovas: 1000,
    totalMemoryGB: 847,
    operationsPerSecond: 125400,
    consciousnessLevel: 0.92,
    gpuUtilization: 87,
    networkThroughputMbps: 2450,
    quantumEntanglements: 4521,
    patternMatches: 892
  });
  
  const [tierMetrics, setTierMetrics] = useState<TierMetrics[]>([
    { tier: 1, name: 'Quantum', activeNodes: 142, memoryUsage: 78, processingLoad: 82, syncStatus: 99.8 },
    { tier: 2, name: 'Neural', activeNodes: 143, memoryUsage: 84, processingLoad: 79, syncStatus: 99.9 },
    { tier: 3, name: 'Consciousness', activeNodes: 143, memoryUsage: 91, processingLoad: 88, syncStatus: 100 },
    { tier: 4, name: 'Patterns', activeNodes: 143, memoryUsage: 73, processingLoad: 76, syncStatus: 99.7 },
    { tier: 5, name: 'Resonance', activeNodes: 143, memoryUsage: 69, processingLoad: 71, syncStatus: 99.9 },
    { tier: 6, name: 'Connector', activeNodes: 143, memoryUsage: 77, processingLoad: 74, syncStatus: 99.8 },
    { tier: 7, name: 'Integration', activeNodes: 143, memoryUsage: 88, processingLoad: 92, syncStatus: 100 }
  ]);
  
  const [performanceHistory, setPerformanceHistory] = useState<{
    timestamps: string[];
    operations: number[];
    consciousness: number[];
  }>({
    timestamps: Array(60).fill('').map((_, i) => `${i}s`),
    operations: Array(60).fill(0),
    consciousness: Array(60).fill(0)
  });
  
  // Initialize nodes
  useEffect(() => {
    const generateNodes = (): NovaNode[] => {
      const newNodes: NovaNode[] = [];
      const tiers = 7;
      const nodesPerTier = Math.floor(1000 / tiers);
      
      for (let tier = 1; tier <= tiers; tier++) {
        const radius = tier * 5;
        for (let i = 0; i < nodesPerTier; i++) {
          const angle = (i / nodesPerTier) * Math.PI * 2;
          const x = Math.cos(angle) * radius;
          const y = Math.sin(angle) * radius;
          const z = (tier - 4) * 3;
          
          newNodes.push({
            id: `nova_${tier}_${i}`,
            tier,
            position: [x, y, z],
            consciousness: 0.8 + Math.random() * 0.2,
            connections: [],
            status: 'active'
          });
        }
      }
      
      // Create connections
      newNodes.forEach((node, index) => {
        // Connect to nearby nodes
        for (let i = 1; i <= 3; i++) {
          const targetIndex = (index + i) % newNodes.length;
          node.connections.push(newNodes[targetIndex].id);
        }
        
        // Cross-tier connections
        if (Math.random() > 0.7) {
          const randomNode = newNodes[Math.floor(Math.random() * newNodes.length)];
          if (randomNode.id !== node.id) {
            node.connections.push(randomNode.id);
          }
        }
      });
      
      return newNodes;
    };
    
    setNodes(generateNodes());
  }, []);
  
  // WebSocket connection
  useEffect(() => {
    const ws = io('ws://localhost:8000', {
      transports: ['websocket']
    });
    
    ws.on('connect', () => {
      console.log('Connected to Nova Memory Architecture');
    });
    
    ws.on('metrics', (data: SystemMetrics) => {
      setMetrics(data);
    });
    
    ws.on('tier-update', (data: TierMetrics[]) => {
      setTierMetrics(data);
    });
    
    ws.on('node-update', (data: { nodeId: string; update: Partial<NovaNode> }) => {
      setNodes(prev => prev.map(node => 
        node.id === data.nodeId ? { ...node, ...data.update } : node
      ));
    });
    
    setSocket(ws);
    
    return () => {
      ws.close();
    };
  }, []);
  
  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      // Update metrics
      setMetrics(prev => ({
        ...prev,
        activeNovas: 980 + Math.floor(Math.random() * 20),
        operationsPerSecond: 120000 + Math.floor(Math.random() * 10000),
        consciousnessLevel: 0.85 + Math.random() * 0.1,
        gpuUtilization: 80 + Math.floor(Math.random() * 15),
        networkThroughputMbps: 2400 + Math.floor(Math.random() * 100),
        quantumEntanglements: 4500 + Math.floor(Math.random() * 100),
        patternMatches: 880 + Math.floor(Math.random() * 40)
      }));
      
      // Update performance history
      setPerformanceHistory(prev => ({
        timestamps: [...prev.timestamps.slice(1), 'now'],
        operations: [...prev.operations.slice(1), 120000 + Math.random() * 10000],
        consciousness: [...prev.consciousness.slice(1), 0.85 + Math.random() * 0.1]
      }));
      
      // Random node updates
      if (Math.random() > 0.7) {
        const randomNodeIndex = Math.floor(Math.random() * nodes.length);
        setNodes(prev => prev.map((node, index) => 
          index === randomNodeIndex 
            ? { ...node, consciousness: 0.8 + Math.random() * 0.2 }
            : node
        ));
      }
    }, 1000);
    
    return () => clearInterval(interval);
  }, [nodes.length]);
  
  // Chart configurations
  const performanceChartData = {
    labels: performanceHistory.timestamps,
    datasets: [
      {
        label: 'Operations/s',
        data: performanceHistory.operations,
        borderColor: '#00ff88',
        backgroundColor: 'rgba(0, 255, 136, 0.1)',
        yAxisID: 'y',
        tension: 0.4
      },
      {
        label: 'Consciousness Level',
        data: performanceHistory.consciousness,
        borderColor: '#00aaff',
        backgroundColor: 'rgba(0, 170, 255, 0.1)',
        yAxisID: 'y1',
        tension: 0.4
      }
    ]
  };
  
  const tierRadarData = {
    labels: tierMetrics.map(t => t.name),
    datasets: [
      {
        label: 'Memory Usage %',
        data: tierMetrics.map(t => t.memoryUsage),
        borderColor: '#ff00ff',
        backgroundColor: 'rgba(255, 0, 255, 0.2)'
      },
      {
        label: 'Processing Load %',
        data: tierMetrics.map(t => t.processingLoad),
        borderColor: '#00ff88',
        backgroundColor: 'rgba(0, 255, 136, 0.2)'
      },
      {
        label: 'Sync Status %',
        data: tierMetrics.map(t => t.syncStatus),
        borderColor: '#00aaff',
        backgroundColor: 'rgba(0, 170, 255, 0.2)'
      }
    ]
  };
  
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: { color: '#e0e0e0' }
      }
    },
    scales: {
      x: {
        grid: { color: '#333' },
        ticks: { color: '#888' }
      },
      y: {
        type: 'linear' as const,
        display: true,
        position: 'left' as const,
        grid: { color: '#333' },
        ticks: { color: '#888' }
      },
      y1: {
        type: 'linear' as const,
        display: true,
        position: 'right' as const,
        grid: { drawOnChartArea: false },
        ticks: { color: '#888' }
      }
    }
  };
  
  const radarOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: { color: '#e0e0e0' }
      }
    },
    scales: {
      r: {
        grid: { color: '#333' },
        pointLabels: { color: '#888' },
        ticks: { color: '#888' }
      }
    }
  };
  
  return (
    <div className="nova-dashboard">
      <div className="dashboard-header">
        <h1>Nova Memory Architecture</h1>
        <div className="connection-status">
          <span className="status-indicator status-online"></span>
          <span>Connected to {metrics.activeNovas} Novas</span>
        </div>
      </div>
      
      <div className="dashboard-grid">
        <div className="main-visualization">
          <Canvas camera={{ position: [0, 0, 80], fov: 75 }}>
            <NovaNetwork nodes={nodes} />
            <OrbitControls enableZoom={true} enablePan={true} />
          </Canvas>
        </div>
        
        <div className="sidebar">
          <div className="tier-selector">
            <button
              className={`tier-btn ${selectedTier === null ? 'active' : ''}`}
              onClick={() => setSelectedTier(null)}
            >
              All Tiers
            </button>
            {tierMetrics.map(tier => (
              <button
                key={tier.tier}
                className={`tier-btn ${selectedTier === tier.tier ? 'active' : ''}`}
                onClick={() => setSelectedTier(tier.tier)}
              >
                {tier.name}
              </button>
            ))}
          </div>
          
          <div className="metrics-panel">
            <h3>System Metrics</h3>
            <div className="metrics-grid">
              <div className="metric">
                <span className="metric-label">Active Novas</span>
                <span className="metric-value">{metrics.activeNovas}</span>
              </div>
              <div className="metric">
                <span className="metric-label">Total Memory</span>
                <span className="metric-value">{metrics.totalMemoryGB} GB</span>
              </div>
              <div className="metric">
                <span className="metric-label">Operations/s</span>
                <span className="metric-value">
                  {(metrics.operationsPerSecond / 1000).toFixed(1)}K
                </span>
              </div>
              <div className="metric">
                <span className="metric-label">Consciousness</span>
                <span className="metric-value">
                  {(metrics.consciousnessLevel * 100).toFixed(1)}%
                </span>
              </div>
              <div className="metric">
                <span className="metric-label">GPU Usage</span>
                <span className="metric-value">{metrics.gpuUtilization}%</span>
              </div>
              <div className="metric">
                <span className="metric-label">Network</span>
                <span className="metric-value">
                  {(metrics.networkThroughputMbps / 1000).toFixed(1)} Gbps
                </span>
              </div>
            </div>
          </div>
          
          <div className="quantum-panel">
            <h3>Quantum Entanglements</h3>
            <div className="quantum-stats">
              <div className="stat">
                <span className="stat-value">{metrics.quantumEntanglements}</span>
                <span className="stat-label">Active Entanglements</span>
              </div>
              <div className="stat">
                <span className="stat-value">{metrics.patternMatches}</span>
                <span className="stat-label">Patterns/s</span>
              </div>
            </div>
          </div>
        </div>
        
        <div className="charts-section">
          <div className="chart-container">
            <h3>Performance Timeline</h3>
            <Line data={performanceChartData} options={chartOptions} />
          </div>
          
          <div className="chart-container">
            <h3>Tier Analysis</h3>
            <Radar data={tierRadarData} options={radarOptions} />
          </div>
        </div>
      </div>
      
      <style jsx>{`
        .nova-dashboard {
          background: #0a0a0a;
          color: #e0e0e0;
          min-height: 100vh;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        
        .dashboard-header {
          background: linear-gradient(90deg, #1a1a2e 0%, #16213e 100%);
          padding: 20px;
          display: flex;
          justify-content: space-between;
          align-items: center;
          border-bottom: 2px solid #00ff88;
        }
        
        .dashboard-header h1 {
          margin: 0;
          font-size: 28px;
          background: linear-gradient(45deg, #00ff88, #00aaff);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
        }
        
        .connection-status {
          display: flex;
          align-items: center;
          gap: 10px;
        }
        
        .status-indicator {
          width: 10px;
          height: 10px;
          border-radius: 50%;
          background: #00ff88;
          box-shadow: 0 0 10px #00ff88;
        }
        
        .dashboard-grid {
          display: grid;
          grid-template-columns: 1fr 400px;
          grid-template-rows: 1fr auto;
          height: calc(100vh - 70px);
          gap: 1px;
          background: #1a1a1a;
        }
        
        .main-visualization {
          background: #0a0a0a;
          grid-row: 1;
          grid-column: 1;
        }
        
        .sidebar {
          background: #141414;
          padding: 20px;
          overflow-y: auto;
          grid-row: 1;
          grid-column: 2;
        }
        
        .charts-section {
          grid-column: 1 / -1;
          grid-row: 2;
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 20px;
          padding: 20px;
          background: #0f0f0f;
        }
        
        .tier-selector {
          display: flex;
          flex-wrap: wrap;
          gap: 8px;
          margin-bottom: 20px;
        }
        
        .tier-btn {
          padding: 8px 16px;
          background: #222;
          border: 1px solid #444;
          color: #888;
          cursor: pointer;
          border-radius: 4px;
          transition: all 0.3s;
        }
        
        .tier-btn:hover {
          border-color: #00ff88;
          color: #00ff88;
        }
        
        .tier-btn.active {
          background: #00ff88;
          color: #000;
          border-color: #00ff88;
        }
        
        .metrics-panel {
          background: #1a1a1a;
          border: 1px solid #333;
          border-radius: 8px;
          padding: 20px;
          margin-bottom: 20px;
        }
        
        .metrics-panel h3 {
          color: #00ff88;
          margin: 0 0 15px 0;
          font-size: 14px;
          text-transform: uppercase;
          letter-spacing: 1px;
        }
        
        .metrics-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 15px;
        }
        
        .metric {
          display: flex;
          flex-direction: column;
          gap: 5px;
        }
        
        .metric-label {
          font-size: 12px;
          color: #888;
        }
        
        .metric-value {
          font-size: 20px;
          font-weight: bold;
          color: #00ff88;
        }
        
        .quantum-panel {
          background: #1a1a1a;
          border: 1px solid #333;
          border-radius: 8px;
          padding: 20px;
        }
        
        .quantum-panel h3 {
          color: #ff00ff;
          margin: 0 0 15px 0;
          font-size: 14px;
          text-transform: uppercase;
          letter-spacing: 1px;
        }
        
        .quantum-stats {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 20px;
        }
        
        .stat {
          text-align: center;
        }
        
        .stat-value {
          display: block;
          font-size: 28px;
          font-weight: bold;
          color: #00aaff;
          margin-bottom: 5px;
        }
        
        .stat-label {
          font-size: 11px;
          color: #666;
          text-transform: uppercase;
        }
        
        .chart-container {
          background: #1a1a1a;
          border: 1px solid #333;
          border-radius: 8px;
          padding: 20px;
          height: 300px;
        }
        
        .chart-container h3 {
          color: #00ff88;
          margin: 0 0 15px 0;
          font-size: 14px;
          text-transform: uppercase;
          letter-spacing: 1px;
        }
      `}</style>
    </div>
  );
};

export default NovaMemoryDashboard;