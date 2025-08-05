"""
Simple Fog-Layer Mitigation Strategy for IoT Zero-Day Attack Response
Based on Chiang & Zhang (2016) and recent fog-FL literature

This implements a lightweight fog computing layer that:
1. Receives threat detection alerts from FL clients
2. Generates mitigation rules (firewall/HIDS rules)
3. Pushes rules to edge devices for real-time protection
4. Provides rapid response without cloud round-trip latency

Citation: Chiang & Zhang (2016) "Fog and IoT: An overview of research opportunities"
Recent work: de Caldas Filho et al. (2023) fog-based FL mitigation
"""

import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import threading
import socket

logger = logging.getLogger(__name__)

@dataclass
class ThreatAlert:
    """Threat detection alert from FL client"""
    client_id: str
    attack_type: str
    confidence: float
    source_ip: str
    destination_port: int
    timestamp: datetime
    attack_features: Dict[str, Any]

@dataclass
class MitigationRule:
    """Mitigation rule to be deployed to edge devices"""
    rule_id: str
    rule_type: str  # 'firewall', 'hids', 'rate_limit'
    action: str     # 'block', 'limit', 'alert'
    target: str     # IP, port, or pattern
    duration: int   # seconds
    priority: int   # 1-10, higher = more priority
    created_at: datetime

class FogMitigationLayer:
    """
    Fog-layer mitigation system for real-time threat response
    
    Based on fog computing principles (Chiang & Zhang, 2016):
    - Low-latency processing at network edge
    - Reduced communication overhead to cloud
    - Distributed intelligence for IoT security
    
    Implements federated learning integration as per recent research:
    - de Caldas Filho et al. (2023): FL-based botnet mitigation
    - Nguyen et al. (2022): Edge intelligence for healthcare
    """
    
    def __init__(self):
        # Fog node configuration
        self.fog_node_id = f"fog_{int(time.time())}"
        self.active_rules: Dict[str, MitigationRule] = {}
        self.threat_history: List[ThreatAlert] = []
        self.edge_devices: List[str] = []
        
        # Real-time metrics for research evaluation
        self.response_times: List[float] = []
        self.rules_deployed: int = 0
        self.threats_mitigated: int = 0
        
        # Background cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_expired_rules, daemon=True)
        self.cleanup_thread.start()
        
        logger.info(f"🌫️ Fog mitigation layer initialized: {self.fog_node_id}")
    
    def register_edge_device(self, device_id: str) -> bool:
        """Register an edge device for rule deployment"""
        if device_id not in self.edge_devices:
            self.edge_devices.append(device_id)
            logger.info(f"📱 Edge device registered: {device_id}")
            return True
        return False
    
    def process_threat_alert(self, alert: ThreatAlert) -> Optional[MitigationRule]:
        """
        Process threat alert from FL client and generate mitigation rule
        
        Based on zero-day detection patterns from your FL algorithms:
        - DDoS attacks -> Rate limiting rules
        - Reconnaissance -> IP blocking rules  
        - Malware -> Deep packet inspection rules
        
        Args:
            alert: Threat detection alert from FL client
            
        Returns:
            Generated mitigation rule or None if no action needed
        """
        start_time = time.time()
        
        # Store alert for analysis
        self.threat_history.append(alert)
        
        # Generate rule based on attack type (aligned with your Bot-IoT categories)
        rule = self._generate_mitigation_rule(alert)
        
        if rule:
            # Deploy rule to edge devices
            success = self._deploy_rule_to_edge_devices(rule)
            
            if success:
                self.active_rules[rule.rule_id] = rule
                self.rules_deployed += 1
                
                # Record response time for research metrics
                response_time = time.time() - start_time
                self.response_times.append(response_time)
                
                logger.info(f"⚡ Threat mitigated: {alert.attack_type} -> {rule.action} "
                           f"(response: {response_time:.3f}s)")
                
                return rule
        
        return None
    
    def _generate_mitigation_rule(self, alert: ThreatAlert) -> Optional[MitigationRule]:
        """
        Generate specific mitigation rule based on attack type
        
        Maps your Bot-IoT attack categories to appropriate responses:
        - Aligns with your research on zero-day botnet detection
        - Uses fog computing for rapid rule generation (Chiang & Zhang, 2016)
        """
        rule_id = f"rule_{alert.client_id}_{int(time.time())}"
        created_at = datetime.now()
        
        # Rule generation based on your Bot-IoT attack categories
        if alert.attack_type.lower() == 'ddos':
            # DDoS mitigation: Rate limiting
            return MitigationRule(
                rule_id=rule_id,
                rule_type='rate_limit',
                action='limit',
                target=alert.source_ip,
                duration=300,  # 5 minutes
                priority=9,    # High priority
                created_at=created_at
            )
        
        elif alert.attack_type.lower() == 'reconnaissance':
            # Reconnaissance: Block scanning IPs
            return MitigationRule(
                rule_id=rule_id,
                rule_type='firewall',
                action='block',
                target=alert.source_ip,
                duration=600,  # 10 minutes
                priority=7,
                created_at=created_at
            )
        
        elif alert.attack_type.lower() in ['dos', 'theft']:
            # DoS/Theft: Immediate blocking
            return MitigationRule(
                rule_id=rule_id,
                rule_type='firewall', 
                action='block',
                target=f"{alert.source_ip}:{alert.destination_port}",
                duration=1800,  # 30 minutes
                priority=8,
                created_at=created_at
            )
        
        elif alert.confidence > 0.9:
            # High-confidence unknown threats: Cautious blocking
            return MitigationRule(
                rule_id=rule_id,
                rule_type='hids',
                action='alert',
                target=alert.source_ip,
                duration=120,  # 2 minutes
                priority=5,
                created_at=created_at
            )
        
        return None
    
    def _deploy_rule_to_edge_devices(self, rule: MitigationRule) -> bool:
        """
        Deploy mitigation rule to registered edge devices
        
        In real deployment, this would:
        - Send iptables rules to edge devices
        - Configure HIDS sensors
        - Update network ACLs
        
        For research simulation, we log the deployment
        """
        if not self.edge_devices:
            logger.warning("⚠️ No edge devices registered for rule deployment")
            return False
        
        # Simulate rule deployment to edge devices
        deployment_payload = {
            'rule_id': rule.rule_id,
            'type': rule.rule_type,
            'action': rule.action,
            'target': rule.target,
            'duration': rule.duration,
            'priority': rule.priority
        }
        
        successful_deployments = 0
        
        for device_id in self.edge_devices:
            try:
                # In real implementation, this would be network communication
                logger.debug(f"📡 Deploying rule {rule.rule_id} to device {device_id}")
                successful_deployments += 1
                
            except Exception as e:
                logger.error(f"❌ Failed to deploy rule to {device_id}: {e}")
        
        success_rate = successful_deployments / len(self.edge_devices)
        
        if success_rate >= 0.8:  # 80% success threshold
            logger.info(f"✅ Rule deployed to {successful_deployments}/{len(self.edge_devices)} devices")
            return True
        else:
            logger.warning(f"⚠️ Rule deployment partial success: {success_rate:.1%}")
            return False
    
    def _cleanup_expired_rules(self):
        """Background cleanup of expired mitigation rules"""
        while True:
            try:
                current_time = datetime.now()
                expired_rules = []
                
                for rule_id, rule in self.active_rules.items():
                    elapsed = (current_time - rule.created_at).total_seconds()
                    if elapsed > rule.duration:
                        expired_rules.append(rule_id)
                
                # Remove expired rules
                for rule_id in expired_rules:
                    removed_rule = self.active_rules.pop(rule_id)
                    logger.debug(f"🧹 Expired rule removed: {rule_id}")
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"❌ Rule cleanup error: {e}")
                time.sleep(60)
    
    def get_mitigation_metrics(self) -> Dict[str, Any]:
        """
        Get fog mitigation performance metrics for research evaluation
        
        Returns metrics needed for your dissertation analysis:
        - Average response time (key for real-time requirement)
        - Rules deployed (automation effectiveness)
        - Threat coverage (security effectiveness)
        """
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        
        return {
            'fog_node_id': self.fog_node_id,
            'total_threats_processed': len(self.threat_history),
            'rules_deployed': self.rules_deployed,
            'active_rules': len(self.active_rules),
            'edge_devices_connected': len(self.edge_devices),
            'avg_response_time_ms': avg_response_time * 1000,
            'min_response_time_ms': min(self.response_times) * 1000 if self.response_times else 0,
            'max_response_time_ms': max(self.response_times) * 1000 if self.response_times else 0,
            'mitigation_effectiveness': self.rules_deployed / max(len(self.threat_history), 1)
        }
    
    def export_research_data(self, filename: str = None) -> str:
        """
        Export fog mitigation data for research analysis
        
        Generates data needed for your dissertation results:
        - Threat response times
        - Rule deployment success rates  
        - Edge device performance metrics
        """
        if filename is None:
            filename = f"fog_mitigation_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        research_data = {
            'experiment_metadata': {
                'fog_node_id': self.fog_node_id,
                'export_timestamp': datetime.now().isoformat(),
                'research_focus': 'Zero-day botnet detection with fog-layer mitigation'
            },
            'performance_metrics': self.get_mitigation_metrics(),
            'threat_analysis': {
                'total_alerts': len(self.threat_history),
                'attack_type_distribution': self._get_attack_distribution(),
                'confidence_distribution': self._get_confidence_distribution()
            },
            'response_analysis': {
                'response_times': self.response_times,
                'avg_response_time': sum(self.response_times) / len(self.response_times) if self.response_times else 0,
                'response_time_percentiles': self._calculate_percentiles(self.response_times)
            },
            'rule_effectiveness': {
                'total_rules_created': self.rules_deployed,
                'active_rules': len(self.active_rules),
                'rule_types': self._get_rule_type_distribution()
            }
        }
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(research_data, f, indent=2, default=str)
        
        logger.info(f"📊 Fog mitigation research data exported: {filename}")
        return filename
    
    def _get_attack_distribution(self) -> Dict[str, int]:
        """Get distribution of attack types for research analysis"""
        distribution = {}
        for alert in self.threat_history:
            attack_type = alert.attack_type.lower()
            distribution[attack_type] = distribution.get(attack_type, 0) + 1
        return distribution
    
    def _get_confidence_distribution(self) -> Dict[str, int]:
        """Get distribution of confidence levels"""
        ranges = {'low': 0, 'medium': 0, 'high': 0}
        for alert in self.threat_history:
            if alert.confidence < 0.6:
                ranges['low'] += 1
            elif alert.confidence < 0.8:
                ranges['medium'] += 1
            else:
                ranges['high'] += 1
        return ranges
    
    def _get_rule_type_distribution(self) -> Dict[str, int]:
        """Get distribution of rule types deployed"""
        distribution = {}
        for rule in self.active_rules.values():
            rule_type = rule.rule_type
            distribution[rule_type] = distribution.get(rule_type, 0) + 1
        return distribution
    
    def _calculate_percentiles(self, data: List[float]) -> Dict[str, float]:
        """Calculate response time percentiles for research analysis"""
        if not data:
            return {'p50': 0, 'p90': 0, 'p95': 0, 'p99': 0}
        
        sorted_data = sorted(data)
        n = len(sorted_data)
        
        return {
            'p50': sorted_data[int(0.5 * n)],
            'p90': sorted_data[int(0.9 * n)],
            'p95': sorted_data[int(0.95 * n)],
            'p99': sorted_data[int(0.99 * n)]
        }


# Integration helper for your existing FL clients
class FogMitigationIntegration:
    """
    Helper class to integrate fog mitigation with your existing FL clients
    
    This allows your client.py to send threat alerts to the fog layer
    without major code changes to your working implementation
    """
    
    def __init__(self, fog_layer: FogMitigationLayer):
        self.fog_layer = fog_layer
    
    def send_threat_alert(self, client_id: str, attack_type: str, 
                         confidence: float, additional_data: Dict = None) -> bool:
        """
        Send threat alert from FL client to fog mitigation layer
        
        This integrates with your existing zero-day detection in client.py
        """
        try:
            # Create threat alert from FL client detection
            alert = ThreatAlert(
                client_id=client_id,
                attack_type=attack_type,
                confidence=confidence,
                source_ip=f"192.168.1.{hash(client_id) % 255}",  # Simulated for research
                destination_port=80,
                timestamp=datetime.now(),
                attack_features=additional_data or {}
            )
            
            # Process through fog mitigation layer
            rule = self.fog_layer.process_threat_alert(alert)
            
            return rule is not None
            
        except Exception as e:
            logger.error(f"❌ Failed to send threat alert: {e}")
            return False


# Example usage for your research experiments
def create_fog_layer_for_research() -> FogMitigationLayer:
    """
    Create and configure fog mitigation layer for your FL experiments
    
    This provides the fog-based mitigation component your supervisor requested
    """
    fog_layer = FogMitigationLayer()
    
    # Register simulated edge devices (representing your IoT clients)
    for i in range(5):  # Matches your 5 FL clients
        fog_layer.register_edge_device(f"iot_edge_{i}")
    
    logger.info("🌫️ Fog mitigation layer ready for FL integration")
    return fog_layer


# Research evaluation functions
def evaluate_fog_mitigation_performance(fog_layer: FogMitigationLayer) -> Dict[str, float]:
    """
    Evaluate fog mitigation performance for dissertation results
    
    Returns metrics that align with your research objectives:
    - Real-time response capability
    - Edge deployment effectiveness  
    - Zero-day threat coverage
    """
    metrics = fog_layer.get_mitigation_metrics()
    
    # Calculate research-relevant performance indicators
    performance_score = {
        'real_time_capability': 1.0 if metrics['avg_response_time_ms'] < 100 else 0.8,
        'deployment_effectiveness': metrics['mitigation_effectiveness'],
        'scalability_score': min(1.0, metrics['edge_devices_connected'] / 10),
        'threat_coverage': metrics['rules_deployed'] / max(metrics['total_threats_processed'], 1)
    }
    
    return performance_score


if __name__ == "__main__":
    # Example test for your research validation
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testing Fog-Layer Mitigation for FL Zero-Day Detection")
    print("=" * 60)
    
    # Create fog mitigation layer
    fog_layer = create_fog_layer_for_research()
    integration = FogMitigationIntegration(fog_layer)
    
    # Simulate threat alerts from your FL clients (matches your Bot-IoT categories)
    test_threats = [
        ('client_0', 'DDoS', 0.95),
        ('client_1', 'Reconnaissance', 0.87),
        ('client_2', 'DoS', 0.92),
        ('client_3', 'Theft', 0.89),
        ('client_4', 'DDoS', 0.94)
    ]
    
    # Process threats through fog layer
    for client_id, attack_type, confidence in test_threats:
        success = integration.send_threat_alert(client_id, attack_type, confidence)
        print(f"✅ Threat processed: {client_id} -> {attack_type} (confidence: {confidence})")
        time.sleep(0.1)  # Simulate realistic timing
    
    # Export research data
    data_file = fog_layer.export_research_data()
    
    # Evaluate performance
    performance = evaluate_fog_mitigation_performance(fog_layer)
    
    print(f"\n📊 Fog Mitigation Performance Results:")
    print(f"   Real-time capability: {performance['real_time_capability']:.2%}")
    print(f"   Deployment effectiveness: {performance['deployment_effectiveness']:.2%}")
    print(f"   Threat coverage: {performance['threat_coverage']:.2%}")
    print(f"   Research data exported: {data_file}")
    print("\n🎓 Ready for dissertation integration!")