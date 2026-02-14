"""
Logging and Observability Utilities
====================================

Structured logging with cost tracking and performance monitoring.
"""

import time
import logging
import structlog
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from contextlib import contextmanager

from utils.config_loader import config


def setup_logging():
    """Configure structured logging"""
    log_level = config.get("observability.log_level", "INFO")
    log_path = config.get("observability.log_path", "logs")

    # Create logs directory
    Path(log_path).mkdir(parents=True, exist_ok=True)

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level),
        handlers=[logging.FileHandler(f"{log_path}/app.log"), logging.StreamHandler()],
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a logger instance"""
    return structlog.get_logger(name)


class CostTracker:
    """
    Track LLM and processing costs.

    CRITICAL for production systems handling millions of calls.
    This helps answer: "How much does it cost to process 1M calls?"
    """

    def __init__(self):
        self.logger = get_logger("cost_tracker")
        self.costs: Dict[str, float] = {}
        self.token_usage: Dict[str, int] = {}

    def track_llm_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_per_1k_input: float,
        cost_per_1k_output: float,
    ) -> float:
        """Track cost of an LLM call"""
        input_cost = (input_tokens / 1000) * cost_per_1k_input
        output_cost = (output_tokens / 1000) * cost_per_1k_output
        total_cost = input_cost + output_cost

        # Aggregate
        self.costs[model] = self.costs.get(model, 0) + total_cost
        self.token_usage[model] = (
            self.token_usage.get(model, 0) + input_tokens + output_tokens
        )

        self.logger.info(
            "llm_call_cost",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=total_cost,
        )

        return total_cost

    def track_embedding_call(
        self, model: str, num_texts: int, total_tokens: int, cost_per_1k: float
    ) -> float:
        """Track cost of embedding generation"""
        cost = (total_tokens / 1000) * cost_per_1k

        self.costs[f"embedding_{model}"] = (
            self.costs.get(f"embedding_{model}", 0) + cost
        )
        self.token_usage[f"embedding_{model}"] = (
            self.token_usage.get(f"embedding_{model}", 0) + total_tokens
        )

        self.logger.info(
            "embedding_call_cost",
            model=model,
            num_texts=num_texts,
            total_tokens=total_tokens,
            cost_usd=cost,
        )

        return cost

    def get_total_cost(self) -> float:
        """Get total accumulated cost"""
        return sum(self.costs.values())

    def get_summary(self) -> Dict[str, Any]:
        """Get cost summary"""
        return {
            "total_cost_usd": self.get_total_cost(),
            "costs_by_model": self.costs,
            "tokens_by_model": self.token_usage,
        }

    def reset(self):
        """Reset cost tracking"""
        self.costs = {}
        self.token_usage = {}


@contextmanager
def track_performance(
    operation_name: str, logger: Optional[structlog.BoundLogger] = None
):
    """
    Context manager to track operation performance.

    Usage:
        with track_performance("preprocessing"):
            # Do work
            pass
    """
    if logger is None:
        logger = get_logger("performance")

    start_time = time.time()
    start_timestamp = datetime.now()

    try:
        yield
    finally:
        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            "operation_completed",
            operation=operation_name,
            duration_ms=round(duration_ms, 2),
            timestamp=start_timestamp.isoformat(),
        )


class MetricsCollector:
    """
    Collect metrics for observability.

    In production, this would export to Prometheus, DataDog, etc.
    """

    def __init__(self):
        self.logger = get_logger("metrics")
        self.counters: Dict[str, int] = {}
        self.gauges: Dict[str, float] = {}

    def increment(
        self, metric_name: str, value: int = 1, labels: Optional[Dict[str, str]] = None
    ):
        """Increment a counter"""
        key = self._make_key(metric_name, labels)
        self.counters[key] = self.counters.get(key, 0) + value

    def set_gauge(
        self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None
    ):
        """Set a gauge value"""
        key = self._make_key(metric_name, labels)
        self.gauges[key] = value

    def _make_key(self, metric_name: str, labels: Optional[Dict[str, str]]) -> str:
        """Create metric key with labels"""
        if labels:
            label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
            return f"{metric_name}{{{label_str}}}"
        return metric_name

    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics"""
        return {"counters": self.counters, "gauges": self.gauges}


# Global instances
cost_tracker = CostTracker()
metrics_collector = MetricsCollector()


# Initialize logging on import
setup_logging()
