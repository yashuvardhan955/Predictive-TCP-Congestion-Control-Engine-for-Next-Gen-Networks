from telemetry import NetworkTelemetryIngestor
import numpy as np

def run_unit_tests():
    print("[TESTING RUNTIME] Running structural logic pipeline validations...")
    
    # Validation 1: Size allocations
    ingestor = NetworkTelemetryIngestor()
    df = ingestor.generate_synthetic_pcap_logs(sample_size=15)
    assert len(df) == 15, "Array length alignment error."
    print(" -> Validation Step 1: Telemetry volume allocations correct.")
    
    # Validation 2: Feature containment bounds
    assert 'optimal_cwnd' in df.columns, "Target assignment failure."
    print(" -> Validation Step 2: Supervised schema labels mapped.")
    print("[SUCCESS] All system validation layers completed without structural errors.")

if __name__ == "__main__":
    run_unit_tests()
