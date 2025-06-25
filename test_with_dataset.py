"""
Comprehensive test runner using the generated dataset
Tests the College ID Validator with realistic synthetic data
"""

import json
import requests
import time
import os
from typing import Dict, List, Any, Optional
from dataset_generator import IDCardDatasetGenerator
import pandas as pd
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import statistics

class DatasetTester:
    def __init__(self, api_base_url="http://localhost:8000", dataset_dir="test_dataset"):
        self.api_base_url = api_base_url
        self.dataset_dir = dataset_dir
        self.results = []
        self.session = requests.Session()  # Reuse connections
        
        # Enhanced metrics tracking
        self.performance_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_time': 0,
            'response_times': [],
            'errors': []
        }
        
        # Confusion matrix data
        self.confusion_matrix = defaultdict(lambda: defaultdict(int))
        
    def load_dataset(self) -> Dict[str, List[Dict]]:
        """Load the generated dataset"""
        dataset_file = os.path.join(self.dataset_dir, "complete_dataset.json")
        
        if not os.path.exists(dataset_file):
            print("Dataset not found. Generating new dataset...")
            generator = IDCardDatasetGenerator(self.dataset_dir)
            return generator.generate_dataset(num_each_category=25)
        
        with open(dataset_file, "r") as f:
            return json.load(f)
    
    def test_api_health(self) -> bool:
        """Test if the API is running"""
        try:
            response = self.session.get(f"{self.api_base_url}/health", timeout=10)
            return response.status_code == 200 and response.json().get("status") == "ok"
        except Exception as e:
            print(f"Health check failed: {e}")
            return False
    
    def validate_single_id(self, user_id: str, image_base64: str, retry_count: int = 2) -> Dict[str, Any]:
        """Validate a single ID card with retry logic"""
        payload = {
            "user_id": user_id,
            "image_base64": image_base64
        }
        
        for attempt in range(retry_count + 1):
            try:
                start_time = time.time()
                response = self.session.post(
                    f"{self.api_base_url}/validate-id", 
                    json=payload, 
                    timeout=30,
                    headers={'Content-Type': 'application/json'}
                )
                end_time = time.time()
                response_time = end_time - start_time
                
                self.performance_metrics['total_requests'] += 1
                self.performance_metrics['total_time'] += response_time
                self.performance_metrics['response_times'].append(response_time)
                
                if response.status_code == 200:
                    result = response.json()
                    result["response_time"] = response_time
                    result["api_status"] = "success"
                    result["attempt"] = attempt + 1
                    self.performance_metrics['successful_requests'] += 1
                    return result
                else:
                    error_result = {
                        "api_status": "error",
                        "error_code": response.status_code,
                        "error_message": response.text[:500],  # Limit error message length
                        "response_time": response_time,
                        "attempt": attempt + 1
                    }
                    
                    if attempt == retry_count:  # Last attempt
                        self.performance_metrics['failed_requests'] += 1
                        self.performance_metrics['errors'].append(error_result)
                        return error_result
                    else:
                        time.sleep(1)  # Wait before retry
                        
            except Exception as e:
                response_time = time.time() - start_time if 'start_time' in locals() else 0
                error_result = {
                    "api_status": "exception",
                    "error_message": str(e)[:500],
                    "response_time": response_time,
                    "attempt": attempt + 1
                }
                
                if attempt == retry_count:  # Last attempt
                    self.performance_metrics['failed_requests'] += 1
                    self.performance_metrics['errors'].append(error_result)
                    return error_result
                else:
                    time.sleep(1)  # Wait before retry
        
        # This should never be reached, but just in case
        return {"api_status": "unknown_error", "response_time": 0}
    
    def validate_batch_concurrent(self, samples: List[Dict], category: str, max_workers: int = 5) -> List[Dict[str, Any]]:
        """Validate multiple samples concurrently"""
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_sample = {}
            for i, sample in enumerate(samples):
                user_id = f"{category}_test_{i:03d}"
                future = executor.submit(self.validate_single_id, user_id, sample["base64"])
                future_to_sample[future] = (i, sample)
            
            # Collect results as they complete
            for future in as_completed(future_to_sample):
                i, sample = future_to_sample[future]
                try:
                    result = future.result()
                    
                    # Add metadata for analysis
                    result["original_category"] = category
                    result["expected_label"] = sample["metadata"]["expected_label"]
                    result["expected_status"] = sample["metadata"]["expected_status"]
                    result["filename"] = sample["filename"]
                    result["test_metadata"] = sample["metadata"]
                    result["sample_index"] = i
                    
                    results.append(result)
                    
                    # Progress update
                    if len(results) % 5 == 0:
                        print(f"   Progress: {len(results)}/{len(samples)} completed")
                        
                except Exception as e:
                    print(f"   Error processing sample {i}: {e}")
                    error_result = {
                        "api_status": "processing_error",
                        "error_message": str(e),
                        "original_category": category,
                        "filename": sample["filename"],
                        "sample_index": i,
                        "response_time": 0
                    }
                    results.append(error_result)
        
        return sorted(results, key=lambda x: x.get('sample_index', 0))
    
    def run_comprehensive_test(self, concurrent: bool = True, max_workers: int = 5) -> Dict[str, Any]:
        """Run comprehensive tests on the entire dataset"""
        print("🧪 Starting Comprehensive Dataset Testing")
        print("=" * 60)
        
        start_total_time = time.time()
        
        # Check API health
        print("🔍 Checking API health...")
        if not self.test_api_health():
            print("❌ API is not running. Please start the server first:")
            print("   python main.py")
            return {"error": "API not available"}
        
        print("✅ API is running and healthy")
        
        # Load dataset
        print("📊 Loading dataset...")
        dataset = self.load_dataset()
        total_samples = sum(len(category_data) for category_data in dataset.values())
        
        print(f"📊 Testing {total_samples} samples across {len(dataset)} categories")
        for category, samples in dataset.items():
            print(f"   - {category.capitalize()}: {len(samples)} samples")
        print()
        
        if concurrent:
            print(f"🚀 Using concurrent testing with {max_workers} workers")
        else:
            print("🐌 Using sequential testing")
        print()
        
        # Test each category
        all_results = []
        category_stats = {}
        
        for category, samples in dataset.items():
            print(f"🔍 Testing {category.upper()} category ({len(samples)} samples)")
            category_start_time = time.time()
            
            if concurrent:
                category_results = self.validate_batch_concurrent(samples, category, max_workers)
            else:
                category_results = []
                for i, sample in enumerate(samples):
                    user_id = f"{category}_test_{i:03d}"
                    result = self.validate_single_id(user_id, sample["base64"])
                    
                    # Add metadata
                    result["original_category"] = category
                    result["expected_label"] = sample["metadata"]["expected_label"]
                    result["expected_status"] = sample["metadata"]["expected_status"]
                    result["filename"] = sample["filename"]
                    result["test_metadata"] = sample["metadata"]
                    
                    category_results.append(result)
                    
                    if (i + 1) % 5 == 0 or (i + 1) == len(samples):
                        print(f"   Progress: {i + 1}/{len(samples)} completed")
            
            category_end_time = time.time()
            category_duration = category_end_time - category_start_time
            
            all_results.extend(category_results)
            
            # Calculate category statistics
            category_stats[category] = self.calculate_category_stats(category_results)
            category_stats[category]['processing_time'] = category_duration
            
            print(f"   ✅ {category.capitalize()} category completed in {category_duration:.2f}s")
            print(f"   📈 Accuracy: {category_stats[category]['accuracy']:.2%}")
            print()
        
        total_end_time = time.time()
        total_duration = total_end_time - start_total_time
        
        # Update confusion matrix
        self.update_confusion_matrix(all_results)
        
        # Calculate overall statistics
        overall_stats = self.calculate_overall_stats(all_results)
        overall_stats['total_processing_time'] = total_duration
        
        # Generate detailed report
        report = {
            "test_summary": {
                "timestamp": datetime.now().isoformat(),
                "total_samples": total_samples,
                "categories_tested": list(dataset.keys()),
                "api_base_url": self.api_base_url,
                "concurrent_testing": concurrent,
                "max_workers": max_workers if concurrent else 1,
                "total_duration": total_duration
            },
            "performance_metrics": self.performance_metrics,
            "overall_stats": overall_stats,
            "category_stats": category_stats,
            "confusion_matrix": dict(self.confusion_matrix),
            "detailed_results": all_results
        }
        
        # Save results
        self.save_test_results(report)
        
        # Generate visualizations
        self.generate_visualizations(report)
        
        # Print summary
        self.print_test_summary(report)
        
        return report
    
    def update_confusion_matrix(self, results: List[Dict[str, Any]]):
        """Update confusion matrix with prediction results"""
        for result in results:
            if result.get("api_status") == "success":
                expected = result.get("expected_label", "unknown")
                predicted = result.get("label", "unknown")
                self.confusion_matrix[expected][predicted] += 1
    
    def calculate_category_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate enhanced statistics for a category"""
        if not results:
            return {}
        
        total_samples = len(results)
        successful_requests = [r for r in results if r.get("api_status") == "success"]
        failed_requests = [r for r in results if r.get("api_status") != "success"]
        
        if not successful_requests:
            return {
                "error": "No successful API calls",
                "total_samples": total_samples,
                "failed_requests": len(failed_requests),
                "failure_rate": len(failed_requests) / total_samples
            }
        
        # Label distribution
        labels = [r.get("label", "unknown") for r in successful_requests]
        label_counts = {label: labels.count(label) for label in set(labels)}
        
        # Status distribution  
        statuses = [r.get("status", "unknown") for r in successful_requests]
        status_counts = {status: statuses.count(status) for status in set(statuses)}
        
        # Score statistics
        scores = [r.get("validation_score", 0) for r in successful_requests if r.get("validation_score") is not None]
        
        # Response time statistics
        response_times = [r.get("response_time", 0) for r in results if r.get("response_time", 0) > 0]
        
        # Accuracy calculation
        correct_predictions = 0
        total_predictions = 0
        misclassifications = []
        
        for result in successful_requests:
            if "expected_label" in result:
                total_predictions += 1
                predicted_label = result.get("label")
                expected_label = result.get("expected_label")
                
                if self.is_prediction_correct(predicted_label, expected_label):
                    correct_predictions += 1
                else:
                    misclassifications.append({
                        "filename": result.get("filename"),
                        "expected": expected_label,
                        "predicted": predicted_label,
                        "score": result.get("validation_score")
                    })
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        # Calculate precision, recall, F1 for each label
        label_metrics = self.calculate_label_metrics(successful_requests)
        
        return {
            "total_samples": total_samples,
            "successful_api_calls": len(successful_requests),
            "failed_api_calls": len(failed_requests),
            "success_rate": len(successful_requests) / total_samples,
            "label_distribution": label_counts,
            "status_distribution": status_counts,
            "score_statistics": {
                "count": len(scores),
                "min": min(scores) if scores else 0,
                "max": max(scores) if scores else 0,
                "mean": statistics.mean(scores) if scores else 0,
                "median": statistics.median(scores) if scores else 0,
                "std_dev": statistics.stdev(scores) if len(scores) > 1 else 0
            },
            "response_time_stats": {
                "count": len(response_times),
                "min": min(response_times) if response_times else 0,
                "max": max(response_times) if response_times else 0,
                "mean": statistics.mean(response_times) if response_times else 0,
                "median": statistics.median(response_times) if response_times else 0,
                "p95": self.percentile(response_times, 95) if response_times else 0,
                "p99": self.percentile(response_times, 99) if response_times else 0
            },
            "accuracy": accuracy,
            "correct_predictions": correct_predictions,
            "total_predictions": total_predictions,
            "misclassifications": misclassifications,
            "label_metrics": label_metrics
        }
    
    def calculate_label_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Calculate precision, recall, F1 for each label"""
        label_metrics = {}
        
        # Get all unique labels
        all_labels = set()
        for result in results:
            if "expected_label" in result:
                all_labels.add(result["expected_label"])
                all_labels.add(result.get("label", "unknown"))
        
        for label in all_labels:
            if label == "unknown":
                continue
                
            tp = sum(1 for r in results if r.get("expected_label") == label and r.get("label") == label)
            fp = sum(1 for r in results if r.get("expected_label") != label and r.get("label") == label)
            fn = sum(1 for r in results if r.get("expected_label") == label and r.get("label") != label)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            label_metrics[label] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn
            }
        
        return label_metrics
    
    def calculate_overall_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall statistics across all categories"""
        overall_stats = self.calculate_category_stats(results)
        
        # Add performance metrics
        overall_stats["performance"] = {
            "total_requests": self.performance_metrics['total_requests'],
            "successful_requests": self.performance_metrics['successful_requests'],
            "failed_requests": self.performance_metrics['failed_requests'],
            "success_rate": self.performance_metrics['successful_requests'] / max(1, self.performance_metrics['total_requests']),
            "average_response_time": self.performance_metrics['total_time'] / max(1, self.performance_metrics['total_requests']),
            "requests_per_second": self.performance_metrics['total_requests'] / max(0.1, self.performance_metrics['total_time'])
        }
        
        return overall_stats
    
    def percentile(self, data: List[float], p: float) -> float:
        """Calculate percentile of data"""
        if not data:
            return 0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * p / 100
        f = int(k)
        c = k - f
        if f == len(sorted_data) - 1:
            return sorted_data[f]
        return sorted_data[f] * (1 - c) + sorted_data[f + 1] * c
    
    def is_prediction_correct(self, predicted: str, expected: str) -> bool:
        """Check if prediction is correct (with some flexibility)"""
        if predicted == expected:
            return True
        
        # Allow some flexibility for edge cases
        flexible_matches = {
            ("suspicious", "fake"): True,  # Suspicious can be acceptable for fake
            ("fake", "suspicious"): True,  # Fake can be acceptable for suspicious
        }
        
        return flexible_matches.get((predicted, expected), False)
    
    def save_test_results(self, report: Dict[str, Any]):
        """Save test results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure reports directory exists
        reports_dir = os.path.join(self.dataset_dir, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        # Save full report
        report_file = os.path.join(reports_dir, f"test_report_{timestamp}.json")
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        
        # Save CSV for analysis
        csv_data = []
        for result in report["detailed_results"]:
            csv_row = {
                "filename": result.get("filename", ""),
                "original_category": result.get("original_category", ""),
                "predicted_label": result.get("label", ""),
                "predicted_status": result.get("status", ""),
                "validation_score": result.get("validation_score", 0),
                "expected_label": result.get("expected_label", ""),
                "expected_status": result.get("expected_status", ""),
                "response_time": result.get("response_time", 0),
                "reason": result.get("reason", ""),
                "api_status": result.get("api_status", ""),
                "correct_prediction": self.is_prediction_correct(
                    result.get("label", ""), 
                    result.get("expected_label", "")
                )
            }
            csv_data.append(csv_row)
        
        csv_file = os.path.join(reports_dir, f"test_results_{timestamp}.csv")
        pd.DataFrame(csv_data).to_csv(csv_file, index=False)
        
        # Save confusion matrix as CSV
        confusion_df = pd.DataFrame(report["confusion_matrix"]).fillna(0)
        confusion_file = os.path.join(reports_dir, f"confusion_matrix_{timestamp}.csv")
        confusion_df.to_csv(confusion_file)
        
        # Save performance summary
        summary_data = {
            "timestamp": report["test_summary"]["timestamp"],
            "total_samples": report["test_summary"]["total_samples"],
            "overall_accuracy": report["overall_stats"]["accuracy"],
            "processing_time": report["test_summary"]["total_duration"],
            "success_rate": report["overall_stats"]["success_rate"],
            "avg_response_time": report["overall_stats"]["response_time_stats"]["mean"]
        }
        
        summary_file = os.path.join(reports_dir, f"summary_{timestamp}.json")
        with open(summary_file, "w") as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"📄 Test results saved to:")
        print(f"   - Full report: {report_file}")
        print(f"   - CSV data: {csv_file}")
        print(f"   - Confusion matrix: {confusion_file}")
        print(f"   - Summary: {summary_file}")
    
    def generate_visualizations(self, report: Dict[str, Any]):
        """Generate visualization charts"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set style
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            
            reports_dir = os.path.join(self.dataset_dir, "reports")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 1. Confusion Matrix Heatmap
            plt.figure(figsize=(10, 8))
            confusion_df = pd.DataFrame(report["confusion_matrix"]).fillna(0)
            sns.heatmap(confusion_df, annot=True, fmt='g', cmap='Blues')
            plt.title('Confusion Matrix - Model Predictions vs Expected')
            plt.ylabel('Expected Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(os.path.join(reports_dir, f"confusion_matrix_{timestamp}.png"), dpi=300)
            plt.close()
            
            # 2. Accuracy by Category
            plt.figure(figsize=(12, 6))
            categories = []
            accuracies = []
            for category, stats in report["category_stats"].items():
                categories.append(category.capitalize())
                accuracies.append(stats.get("accuracy", 0) * 100)
            
            bars = plt.bar(categories, accuracies, color=['#2E8B57', '#FF6347', '#4682B4'])
            plt.title('Accuracy by Category')
            plt.ylabel('Accuracy (%)')
            plt.ylim(0, 100)
            
            # Add value labels on bars
            for bar, acc in zip(bars, accuracies):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(reports_dir, f"accuracy_by_category_{timestamp}.png"), dpi=300)
            plt.close()
            
            # 3. Response Time Distribution
            plt.figure(figsize=(12, 6))
            response_times = [r.get("response_time", 0) for r in report["detailed_results"] 
                            if r.get("response_time", 0) > 0]
            
            plt.hist(response_times, bins=30, edgecolor='black', alpha=0.7)
            plt.title('Response Time Distribution')
            plt.xlabel('Response Time (seconds)')
            plt.ylabel('Frequency')
            plt.axvline(statistics.mean(response_times), color='red', linestyle='--', 
                       label=f'Mean: {statistics.mean(response_times):.3f}s')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(reports_dir, f"response_time_dist_{timestamp}.png"), dpi=300)
            plt.close()
            
            # 4. Score Distribution by Category
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            for i, (category, stats) in enumerate(report["category_stats"].items()):
                category_results = [r for r in report["detailed_results"] 
                                  if r.get("original_category") == category and r.get("api_status") == "success"]
                scores = [r.get("validation_score", 0) for r in category_results]
                
                if scores:
                    axes[i].hist(scores, bins=20, alpha=0.7, edgecolor='black')
                    axes[i].set_title(f'{category.capitalize()} - Score Distribution')
                    axes[i].set_xlabel('Validation Score')
                    axes[i].set_ylabel('Frequency')
                    axes[i].axvline(statistics.mean(scores), color='red', linestyle='--',
                                   label=f'Mean: {statistics.mean(scores):.3f}')
                    axes[i].legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(reports_dir, f"score_distribution_{timestamp}.png"), dpi=300)
            plt.close()
            
            print(f"📊 Visualizations saved to {reports_dir}/")
            
        except ImportError:
            print("⚠️  Matplotlib/Seaborn not available. Skipping visualizations.")
            print("   Install with: pip install matplotlib seaborn")
        except Exception as e:
            print(f"⚠️  Error generating visualizations: {e}")
    
    def print_test_summary(self, report: Dict[str, Any]):
        """Print a comprehensive test summary"""
        print("\n" + "="*80)
        print("🎯 COMPREHENSIVE TEST SUMMARY REPORT")
        print("="*80)
        
        # Overall Performance
        overall = report["overall_stats"]
        test_summary = report["test_summary"]
        
        print(f"📊 OVERALL PERFORMANCE:")
        print(f"   Total Samples: {overall['total_samples']}")
        print(f"   Successful API Calls: {overall['successful_api_calls']}")
        print(f"   Failed API Calls: {overall['failed_api_calls']}")
        print(f"   Success Rate: {overall['success_rate']:.2%}")
        print(f"   Overall Accuracy: {overall['accuracy']:.2%}")
        print(f"   Total Processing Time: {test_summary['total_duration']:.2f}s")
        print(f"   Average Response Time: {overall['response_time_stats']['mean']:.3f}s")
        print(f"   95th Percentile Response Time: {overall['response_time_stats']['p95']:.3f}s")
        print()
        
        # Performance Metrics
        perf = overall.get("performance", {})
        if perf:
            print(f"🚀 PERFORMANCE METRICS:")
            print(f"   Requests per Second: {perf.get('requests_per_second', 0):.2f}")
            print(f"   Average Response Time: {perf.get('average_response_time', 0):.3f}s")
            print()
        
        # Score Statistics
        print(f"📈 SCORE DISTRIBUTION:")
        scores = overall['score_statistics']
        print(f"   Min Score: {scores['min']:.3f}")
        print(f"   Max Score: {scores['max']:.3f}")
        print(f"   Mean Score: {scores['mean']:.3f}")
        print(f"   Median Score: {scores['median']:.3f}")
        print(f"   Standard Deviation: {scores['std_dev']:.3f}")
        print()
        
        # Category Breakdown
        print(f"📂 CATEGORY PERFORMANCE:")
        for category, stats in report["category_stats"].items():
            if isinstance(stats, dict) and "accuracy" in stats:
                print(f"   {category.upper()}:")
                print(f"     Samples: {stats['total_samples']}")
                print(f"     Accuracy: {stats['accuracy']:.2%}")
                print(f"     Success Rate: {stats['success_rate']:.2%}")
                print(f"     Avg Response Time: {stats['response_time_stats']['mean']:.3f}s")
                
                # Show label distribution
                if 'label_distribution' in stats:
                    label_dist = stats['label_distribution']
                    print(f"     Predictions: {', '.join([f'{k}:{v}' for k, v in label_dist.items()])}")
                print()
        
        # Label-wise Metrics (F1, Precision, Recall)
        print(f"🎯 LABEL-WISE METRICS (Overall):")
        if "label_metrics" in report["overall_stats"]:
            for label, metrics in report["overall_stats"]["label_metrics"].items():
                print(f"   {label.upper()}:")
                print(f"     - Precision: {metrics.get('precision', 0):.2%}")
                print(f"     - Recall:    {metrics.get('recall', 0):.2%}")
                print(f"     - F1-Score:  {metrics.get('f1_score', 0):.2%}")
            print()
        else:
            print("   (No label metrics calculated)")
            print()

        # Confusion Matrix
        print("🔄 CONFUSION MATRIX (Expected vs. Predicted):")
        if 'confusion_matrix' in report:
            try:
                confusion_df = pd.DataFrame(report["confusion_matrix"]).fillna(0).astype(int)
                # Ensure all labels are present for a square matrix
                all_labels = sorted(list(set(confusion_df.index) | set(confusion_df.columns)))
                confusion_df = confusion_df.reindex(index=all_labels, columns=all_labels, fill_value=0)
                print(confusion_df.to_string())
            except ImportError:
                print("  (Install pandas for a better view: pip install pandas)")
                print(report['confusion_matrix'])
        print()

        # Top misclassifications
        if "misclassifications" in report["overall_stats"]:
             misclassifications = report["overall_stats"].get("misclassifications", [])
             if misclassifications:
                print(f"⚠️ TOP 5 MISCLASSIFICATIONS:")
                for i, item in enumerate(misclassifications[:5]):
                    print(f"   {i+1}. File: {item.get('filename', 'N/A')}")
                    print(f"      - Expected: '{item.get('expected')}', Predicted: '{item.get('predicted')}', Score: {item.get('score', -1):.2f}")
                print()

        print("=" * 80)
        print("✅ Test run complete. Check the 'reports' directory for detailed logs and charts.")
        print("=" * 80)

def main():
    """Main function to run the test suite"""
    # Create an argument parser to allow for CLI options
    import argparse
    parser = argparse.ArgumentParser(description="Comprehensive Test Runner for College ID Validator")
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the running API"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="test_dataset",
        help="Directory to store or load the dataset from"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of concurrent workers for testing"
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run tests sequentially instead of concurrently"
    )
    
    args = parser.parse_args()
    
    # Check for visual dependencies
    try:
        import matplotlib
        import seaborn
    except ImportError:
        print("⚠️  Matplotlib and Seaborn are not installed. Visualizations will be skipped.")
        print("   Install them using: pip install matplotlib seaborn pandas")


    # Instantiate the tester
    tester = DatasetTester(api_base_url=args.api_url, dataset_dir=args.dataset_dir)
    
    # Run the tests
    tester.run_comprehensive_test(
        concurrent=not args.sequential,
        max_workers=args.workers
    )

if __name__ == "__main__":
    main() 