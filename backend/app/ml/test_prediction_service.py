"""
YELENA v2 — Prediction Service Test Suite
Run after deploying the prediction service to validate all endpoints.

Usage:
    python test_prediction_service.py [--host localhost] [--port 8001]
"""

import requests
import json
import time
import sys
import argparse
import numpy as np

# Default feature counts per TF (fallback if /health doesn't have metadata)
DEFAULT_TF_FEATURES = {"1min": 163, "5min": 156, "15min": 149, "1hr": 139}

def _get_tf_feature_count(base_url: str, timeframe: str) -> int:
    """Get the expected feature count for a timeframe from the service."""
    try:
        r = requests.get(f"{base_url}/models", timeout=5)
        data = r.json()
        if "tf_metadata" in data and timeframe in data["tf_metadata"]:
            n = data["tf_metadata"][timeframe].get("n_features")
            if n:
                return n
    except Exception:
        pass
    return DEFAULT_TF_FEATURES.get(timeframe, 156)


def test_health(base_url: str) -> bool:
    """Test /health endpoint."""
    print("\n" + "=" * 50)
    print("TEST: /health")
    print("=" * 50)

    try:
        r = requests.get(f"{base_url}/health", timeout=10)
        data = r.json()
        print(f"  Status: {data['status']}")
        print(f"  Models loaded: {data['models_loaded']}")
        print(f"  Timeframes: {data['loaded_timeframes']}")
        print(f"  Uptime: {data['uptime_seconds']}s")
        print(f"  Version: {data['version']}")

        assert data['status'] == 'healthy', f"Expected healthy, got {data['status']}"
        assert data['models_loaded'] > 0, "No models loaded"
        print("  ✅ PASSED")
        return True
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        return False


def test_models_info(base_url: str) -> dict:
    """Test /models endpoint."""
    print("\n" + "=" * 50)
    print("TEST: /models")
    print("=" * 50)

    try:
        r = requests.get(f"{base_url}/models", timeout=10)
        data = r.json()
        print(f"  Total models: {data['total_models']}")
        print(f"  Threshold: {data['threshold']}")
        for tf, models in data['models_per_tf'].items():
            print(f"  {tf}: {len(models)} models — {models}")

        assert data['total_models'] > 0
        print("  ✅ PASSED")
        return data
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        return {}


def test_predict_single(base_url: str, timeframe: str = "15min") -> bool:
    """Test /predict endpoint with synthetic data."""
    print("\n" + "=" * 50)
    print(f"TEST: /predict ({timeframe})")
    print("=" * 50)

    # Get per-TF feature count from health endpoint
    n_features = _get_tf_feature_count(base_url, timeframe)
    print(f"  Using {n_features} features for {timeframe}")

    # Generate synthetic features
    np.random.seed(42)
    features = np.random.randn(n_features).tolist()

    # Sequence for Transformer/CNN (30 bars × n_features)
    sequence = np.random.randn(30, n_features).tolist()

    payload = {
        "symbol": "SPY",
        "timeframe": timeframe,
        "features": features,
        "sequence": sequence
    }

    try:
        start = time.time()
        r = requests.post(f"{base_url}/predict", json=payload, timeout=30)
        client_latency = (time.time() - start) * 1000

        if r.status_code != 200:
            print(f"  ❌ HTTP {r.status_code}: {r.text}")
            return False

        data = r.json()
        print(f"  Direction: {data['direction']}")
        print(f"  Probability: {data['probability']}")
        print(f"  Confidence: {data['confidence']}%")
        print(f"  Grade: {data['grade']}")
        print(f"  Models agreeing: {data['models_agreeing']}")
        print(f"  Unanimous: {data['unanimous']}")
        print(f"  Server latency: {data['latency_ms']}ms")
        print(f"  Client latency: {client_latency:.1f}ms")

        print(f"\n  Individual predictions:")
        for name, pred in data['individual'].items():
            print(f"    {name}: {pred['direction']} {pred['probability']:.4f} → {pred['signal']}")

        # Validate response structure
        assert data['direction'] in ('CALL', 'PUT', 'HOLD')
        assert 0 <= data['probability'] <= 1
        assert 0 <= data['confidence'] <= 100
        assert data['grade'] in ('A+', 'A', 'B+', 'B', 'C')
        assert data['latency_ms'] < 500, f"Latency too high: {data['latency_ms']}ms"

        print("  ✅ PASSED")
        return True
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        return False


def test_predict_xgboost_only(base_url: str, timeframe: str = "15min") -> bool:
    """Test /predict with features only (no sequence = XGBoost + RL only)."""
    print("\n" + "=" * 50)
    print(f"TEST: /predict no-sequence / XGBoost+RL only ({timeframe})")
    print("=" * 50)

    n_features = _get_tf_feature_count(base_url, timeframe)
    np.random.seed(42)
    features = np.random.randn(n_features).tolist()

    payload = {
        "symbol": "SPY",
        "timeframe": timeframe,
        "features": features
        # No sequence — only XGBoost runs
    }

    try:
        r = requests.post(f"{base_url}/predict", json=payload, timeout=30)
        data = r.json()

        if r.status_code != 200:
            print(f"  ❌ HTTP {r.status_code}: {r.text}")
            return False

        print(f"  Direction: {data['direction']}")
        print(f"  Models in response: {list(data['individual'].keys())}")
        print(f"  Latency: {data['latency_ms']}ms")

        # Should have XGBoost and RL predictions (no transformer/cnn without sequence)
        has_xgb = any('xgboost' in k for k in data['individual'])
        has_rl = any('rl' in k for k in data['individual'])
        has_transformer = any('transformer' in k for k in data['individual'])

        assert has_xgb, "Missing XGBoost predictions"
        assert has_rl, "Missing RL predictions"
        assert not has_transformer, "Transformer should not be present without sequence"
        print("  ✅ PASSED")
        return True
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        return False


def test_predict_cross_tf(base_url: str) -> bool:
    """Test /predict/cross-tf endpoint."""
    print("\n" + "=" * 50)
    print("TEST: /predict/cross-tf")
    print("=" * 50)

    np.random.seed(42)

    # Each TF has different feature counts
    tfs = {"5min": _get_tf_feature_count(base_url, "5min"),
           "15min": _get_tf_feature_count(base_url, "15min"),
           "1hr": _get_tf_feature_count(base_url, "1hr")}

    payload = {
        "symbol": "SPY",
        "primary_tf": "15min",
        "timeframes": {
            tf: np.random.randn(n).tolist() for tf, n in tfs.items()
        },
        "sequences": {
            tf: np.random.randn(30, n).tolist() for tf, n in tfs.items()
        }
    }

    try:
        start = time.time()
        r = requests.post(f"{base_url}/predict/cross-tf", json=payload, timeout=60)
        client_latency = (time.time() - start) * 1000

        if r.status_code != 200:
            print(f"  ❌ HTTP {r.status_code}: {r.text}")
            return False

        data = r.json()
        print(f"  Primary direction: {data['primary_direction']}")
        print(f"  Primary confidence: {data['primary_confidence']}%")
        print(f"  Primary grade: {data['primary_grade']}")
        print(f"  TF agreement: {data['tf_agreement']}")
        print(f"  Agreement: {data['agreement_count']}/{data['total_tfs']}")
        print(f"  Confidence bonus: +{data['confidence_bonus']}pts")
        print(f"  Adjusted confidence: {data['adjusted_confidence']}%")
        print(f"  Server latency: {data['latency_ms']}ms")
        print(f"  Client latency: {client_latency:.1f}ms")

        assert data['total_tfs'] == 3
        assert 0 <= data['adjusted_confidence'] <= 100
        print("  ✅ PASSED")
        return True
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        return False


def test_reload(base_url: str) -> bool:
    """Test /models/reload endpoint."""
    print("\n" + "=" * 50)
    print("TEST: /models/reload (15min only)")
    print("=" * 50)

    try:
        r = requests.post(
            f"{base_url}/models/reload",
            json={"timeframe": "15min"},
            timeout=30
        )
        data = r.json()
        print(f"  Result: {data}")
        assert data['status'] == 'success'
        print("  ✅ PASSED")
        return True
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        return False


def test_error_handling(base_url: str) -> bool:
    """Test error cases."""
    print("\n" + "=" * 50)
    print("TEST: Error handling")
    print("=" * 50)

    passed = True

    # Wrong feature count
    r = requests.post(f"{base_url}/predict", json={
        "symbol": "SPY", "timeframe": "15min", "features": [0.1] * 10
    })
    if r.status_code == 400:
        print("  ✅ Wrong feature count → 400")
    else:
        print(f"  ❌ Wrong feature count → {r.status_code} (expected 400)")
        passed = False

    # Invalid timeframe
    r = requests.post(f"{base_url}/predict", json={
        "symbol": "SPY", "timeframe": "2min", "features": [0.1] * 156
    })
    if r.status_code == 400:
        print("  ✅ Invalid timeframe → 400")
    else:
        print(f"  ❌ Invalid timeframe → {r.status_code} (expected 400)")
        passed = False

    if passed:
        print("  ✅ PASSED")
    return passed


def test_latency_benchmark(base_url: str, n_iterations: int = 10) -> bool:
    """Benchmark prediction latency."""
    print("\n" + "=" * 50)
    print(f"TEST: Latency benchmark ({n_iterations} iterations)")
    print("=" * 50)

    np.random.seed(42)
    n_features = _get_tf_feature_count(base_url, "15min")
    features = np.random.randn(n_features).tolist()
    sequence = np.random.randn(30, n_features).tolist()

    payload = {
        "symbol": "SPY",
        "timeframe": "15min",
        "features": features,
        "sequence": sequence
    }

    latencies = []
    server_latencies = []

    for i in range(n_iterations):
        start = time.time()
        r = requests.post(f"{base_url}/predict", json=payload, timeout=30)
        client_lat = (time.time() - start) * 1000
        latencies.append(client_lat)

        if r.status_code == 200:
            server_latencies.append(r.json()['latency_ms'])

    avg_client = np.mean(latencies)
    p95_client = np.percentile(latencies, 95)
    avg_server = np.mean(server_latencies) if server_latencies else 0

    print(f"  Client latency (avg): {avg_client:.1f}ms")
    print(f"  Client latency (p95): {p95_client:.1f}ms")
    print(f"  Server latency (avg): {avg_server:.1f}ms")
    print(f"  Network overhead: {avg_client - avg_server:.1f}ms")

    target = 100  # Target: <100ms server latency
    if avg_server < target:
        print(f"  ✅ PASSED (server avg {avg_server:.1f}ms < {target}ms target)")
        return True
    else:
        print(f"  ⚠️  SLOW (server avg {avg_server:.1f}ms > {target}ms target)")
        return False


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Test YELENA v2 Prediction Service")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--benchmark", action="store_true", help="Run latency benchmark")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    print(f"Testing YELENA v2 Prediction Service at {base_url}")

    results = {}

    # Core tests
    results['health'] = test_health(base_url)
    if not results['health']:
        print("\n❌ Service not reachable. Is it running?")
        sys.exit(1)

    results['models'] = bool(test_models_info(base_url))
    results['predict_full'] = test_predict_single(base_url, "15min")
    results['predict_xgb_only'] = test_predict_xgboost_only(base_url, "15min")
    results['cross_tf'] = test_predict_cross_tf(base_url)
    results['reload'] = test_reload(base_url)
    results['errors'] = test_error_handling(base_url)

    if args.benchmark:
        results['benchmark'] = test_latency_benchmark(base_url)

    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, result in results.items():
        icon = "✅" if result else "❌"
        print(f"  {icon} {name}")
    print(f"\n  {passed}/{total} tests passed")

    if passed == total:
        print("\n🚀 ALL TESTS PASSED — Prediction service is ready!")
    else:
        print("\n⚠️  Some tests failed — review output above")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
