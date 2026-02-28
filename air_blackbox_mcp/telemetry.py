"""
OpenTelemetry instrumentation for AIR Blackbox compliance scanning.

Follows OpenTelemetry GenAI SIG semantic conventions for generative AI
applications. Instruments scan operations, trust layer events, and model
analysis with proper OTel traces and metrics.

Environment Variables:
  AIR_TELEMETRY_ENABLED: "true"/"false" (default: "true")
  AIR_TELEMETRY_ENDPOINT: OTLP endpoint (default: "http://localhost:4317")
  AIR_TELEMETRY_CONSOLE: "true" to print spans to console
  OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: "true" to capture code

All attributes use gen_ai.* namespace per OpenTelemetry GenAI SIG conventions.
"""

import os
import json
import time
import logging
import functools
import threading
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import OTel packages; gracefully degrade if not available
try:
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.trace import Status, StatusCode
    HAS_OTEL = True
except ImportError:
    HAS_OTEL = False


@dataclass
class SpanEvent:
    """Represents a telemetry span event for audit chain integration."""
    name: str
    timestamp: str
    attributes: Dict[str, Any]
    status: str
    duration_ms: float


class NoOpTracer:
    """No-op tracer when OTel is not available."""
    def start_as_current_span(self, name: str):
        return NoOpSpan()
    
    def start_span(self, name: str):
        return NoOpSpan()


class NoOpSpan:
    """No-op span context manager."""
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass
    
    def set_attribute(self, key: str, value: Any):
        pass
    
    def set_attributes(self, attrs: Dict[str, Any]):
        pass
    
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        pass
    
    def record_exception(self, exception: Exception):
        pass
    
    def set_status(self, status):
        pass


class NoOpMeter:
    """No-op meter when OTel is not available."""
    def create_counter(self, name: str, **kwargs):
        return NoOpCounter()
    
    def create_histogram(self, name: str, **kwargs):
        return NoOpHistogram()
    
    def create_gauge(self, name: str, **kwargs):
        return NoOpGauge()


class NoOpCounter:
    """No-op counter."""
    def add(self, value: float, attributes: Optional[Dict[str, Any]] = None):
        pass


class NoOpHistogram:
    """No-op histogram."""
    def record(self, value: float, attributes: Optional[Dict[str, Any]] = None):
        pass


class NoOpGauge:
    """No-op gauge."""
    def record(self, value: float, attributes: Optional[Dict[str, Any]] = None):
        pass


class AirTelemetry:
    """
    Singleton telemetry wrapper for AIR Blackbox compliance scanning.
    
    Initializes OpenTelemetry TracerProvider and MeterProvider if packages
    are available. Gracefully degrades to no-ops if OTel is not installed.
    Provides decorator-based instrumentation with GenAI SIG conventions.
    """
    
    _instance: Optional['AirTelemetry'] = None
    _lock: threading.Lock = threading.Lock()
    
    def __new__(cls, service_name: str = "air-blackbox", enable: bool = True):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, service_name: str = "air-blackbox", enable: bool = True):
        """
        Initialize OTel instrumentation for AIR Blackbox.
        
        Args:
            service_name: Service name for resource attributes
            enable: Enable telemetry (checks AIR_TELEMETRY_ENABLED env var)
        """
        if self._initialized:
            return
        
        self.service_name = service_name
        self.enabled = enable and os.getenv("AIR_TELEMETRY_ENABLED", "true").lower() == "true"
        self.console_export = os.getenv("AIR_TELEMETRY_CONSOLE", "false").lower() == "true"
        self.otlp_endpoint = os.getenv("AIR_TELEMETRY_ENDPOINT", "http://localhost:4317")
        self.capture_content = os.getenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "false").lower() == "true"
        
        # Span and metric collections for audit chain and export
        self._spans: List[SpanEvent] = []
        self._span_lock = threading.Lock()
        
        # Initialize OTel if available and enabled
        if HAS_OTEL and self.enabled:
            self._init_otel()
        else:
            self._tracer = NoOpTracer()
            self._meter = NoOpMeter()
        
        self._initialized = True
    
    def _init_otel(self):
        """Initialize OpenTelemetry TracerProvider and MeterProvider."""
        try:
            # Create resource with service attributes
            resource = Resource.create({
                "service.name": self.service_name,
                "service.version": self._get_package_version(),
                "gen_ai.system": "air-blackbox",
                "gen_ai.agent.name": "air-compliance-scanner",
                "gen_ai.agent.version": self._get_package_version(),
            })
            
            # Initialize TracerProvider
            tracer_provider = TracerProvider(resource=resource)
            
            # Add OTLP span exporter
            otlp_exporter = OTLPSpanExporter(endpoint=self.otlp_endpoint)
            tracer_provider.add_span_processor(SimpleSpanProcessor(otlp_exporter))
            
            # Add console exporter if enabled
            if self.console_export:
                tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
            
            trace.set_tracer_provider(tracer_provider)
            self._tracer = trace.get_tracer(__name__)
            
            # Initialize MeterProvider
            metric_reader = PeriodicExportingMetricReader(
                OTLPMetricExporter(endpoint=self.otlp_endpoint)
            )
            meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
            metrics.set_meter_provider(meter_provider)
            self._meter = metrics.get_meter(__name__)
            
            logger.info(f"OTel telemetry initialized: endpoint={self.otlp_endpoint}")
        except Exception as e:
            logger.warning(f"Failed to initialize OTel: {e}. Falling back to no-ops.")
            self._tracer = NoOpTracer()
            self._meter = NoOpMeter()
    
    @staticmethod
    def _get_package_version() -> str:
        """Get AIR Blackbox package version."""
        try:
            import pkg_resources
            return pkg_resources.get_distribution("air-blackbox").version
        except:
            return "unknown"
    
    def trace_scan(self, func: Callable) -> Callable:
        """
        Decorator for scan operations (code, file, project scans).
        
        Creates a span with gen_ai.operation.name="compliance.scan" and
        records scan results as attributes.
        
        Example:
            @telemetry.trace_scan
            def scan_code(code: str) -> Dict:
                ...
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            operation_name = f"compliance.{func.__name__.replace('scan_', '')}"
            span_name = f"AIR.{func.__name__}"
            
            start_time = time.time()
            span_attrs = {
                "gen_ai.system": "air-blackbox",
                "gen_ai.operation.name": operation_name,
                "gen_ai.agent.name": "air-compliance-scanner",
            }
            
            try:
                with self._tracer.start_as_current_span(span_name) as span:
                    span.set_attributes(span_attrs)
                    
                    # Execute the function
                    result = func(*args, **kwargs)
                    
                    # Record result attributes
                    if isinstance(result, dict):
                        if "framework" in result:
                            span.set_attribute("gen_ai.request.framework", result["framework"])
                        if "passed" in result and "total" in result:
                            span.set_attribute("air.compliance.articles_passed", result["passed"])
                            span.set_attribute("air.compliance.articles_total", result["total"])
                            score = (result["passed"] / result["total"] * 100) if result["total"] > 0 else 0
                            span.set_attribute("air.compliance.score", round(score, 2))
                    
                    # Record duration
                    duration_ms = (time.time() - start_time) * 1000
                    span.set_attribute("air.scan.duration_ms", round(duration_ms, 2))
                    span.set_status(Status(StatusCode.OK))
                    
                    # Record span event
                    self._record_span_event(span_name, span_attrs, "OK", duration_ms)
                    return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                with self._tracer.start_as_current_span(span_name) as span:
                    span.set_attributes(span_attrs)
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR))
                    self._record_span_event(span_name, span_attrs, "ERROR", duration_ms)
                raise
        
        return wrapper
    
    def trace_model_call(self, model_name: str) -> Callable:
        """
        Decorator for Ollama model API calls.
        
        Follows GenAI SIG conventions with:
        - gen_ai.request.model / gen_ai.response.model
        - gen_ai.usage.input_tokens / gen_ai.usage.output_tokens
        
        Example:
            @telemetry.trace_model_call("air-compliance-v2")
            def analyze_code(code: str) -> str:
                ...
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                span_name = f"AIR.model.{func.__name__}"
                start_time = time.time()
                
                span_attrs = {
                    "gen_ai.system": "air-blackbox",
                    "gen_ai.operation.name": "compliance.analyze",
                    "gen_ai.request.model": model_name,
                    "gen_ai.response.model": model_name,
                    "gen_ai.agent.name": "air-compliance-scanner",
                }
                
                try:
                    with self._tracer.start_as_current_span(span_name) as span:
                        span.set_attributes(span_attrs)
                        
                        # Capture input content if enabled
                        if self.capture_content and args:
                            input_content = str(args[0])[:500]  # First 500 chars
                            span.set_attribute("gen_ai.request.message.content", input_content)
                        
                        # Execute the model call
                        result = func(*args, **kwargs)
                        
                        # Capture output content if enabled
                        if self.capture_content and result:
                            output_content = str(result)[:500]  # First 500 chars
                            span.set_attribute("gen_ai.response.message.content", output_content)
                        
                        # Record latency
                        duration_ms = (time.time() - start_time) * 1000
                        span.set_attribute("gen_ai.request.latency_ms", round(duration_ms, 2))
                        span.set_status(Status(StatusCode.OK))
                        
                        # Record as metric
                        self._meter.create_counter(
                            "air.compliance.model.calls",
                            description="Ollama model invocations"
                        ).add(1, {"model": model_name})
                        
                        self._meter.create_histogram(
                            "air.compliance.model.latency",
                            unit="ms",
                            description="Model response time"
                        ).record(duration_ms, {"model": model_name})
                        
                        self._record_span_event(span_name, span_attrs, "OK", duration_ms)
                        return result
                except Exception as e:
                    duration_ms = (time.time() - start_time) * 1000
                    with self._tracer.start_as_current_span(span_name) as span:
                        span.set_attributes(span_attrs)
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR))
                        self._record_span_event(span_name, span_attrs, "ERROR", duration_ms)
                    raise
            
            return wrapper
        return decorator
    
    def trace_tool(self, tool_name: str) -> Callable:
        """
        Decorator for MCP tool invocations.
        
        Creates spans with gen_ai.tool.* attributes following GenAI SIG.
        
        Example:
            @telemetry.trace_tool("scan_code")
            async def scan_code(code: str) -> str:
                ...
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                span_name = f"AIR.tool.{tool_name}"
                start_time = time.time()
                
                span_attrs = {
                    "gen_ai.system": "air-blackbox",
                    "gen_ai.operation.name": f"compliance.{tool_name}",
                    "gen_ai.tool.type": "Function",
                    "gen_ai.tool.name": tool_name,
                    "gen_ai.agent.name": "air-compliance-scanner",
                }
                
                try:
                    with self._tracer.start_as_current_span(span_name) as span:
                        span.set_attributes(span_attrs)
                        
                        # Execute the tool
                        result = func(*args, **kwargs)
                        
                        # Record result
                        if isinstance(result, str) and len(result) > 0:
                            span.set_attribute("gen_ai.tool.result", result[:100])
                        
                        duration_ms = (time.time() - start_time) * 1000
                        span.set_attribute("gen_ai.tool.duration_ms", round(duration_ms, 2))
                        span.set_status(Status(StatusCode.OK))
                        
                        self._record_span_event(span_name, span_attrs, "OK", duration_ms)
                        return result
                except Exception as e:
                    duration_ms = (time.time() - start_time) * 1000
                    with self._tracer.start_as_current_span(span_name) as span:
                        span.set_attributes(span_attrs)
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR))
                        self._record_span_event(span_name, span_attrs, "ERROR", duration_ms)
                    raise
            
            return wrapper
        return decorator
    
    def record_scan_metrics(
        self,
        framework: str,
        passed: int,
        total: int,
        duration_ms: float
    ):
        """
        Record scan results as metrics.
        
        Records:
        - Counter for total scans
        - Histogram for scan duration
        - Gauge for compliance score
        
        Args:
            framework: Detected framework name
            passed: Number of articles passed
            total: Total articles scanned
            duration_ms: Scan duration in milliseconds
        """
        # Counter for total scans
        counter = self._meter.create_counter(
            "air.compliance.scans",
            description="Total compliance scans performed"
        )
        counter.add(1, {"framework": framework})
        
        # Histogram for duration
        histogram = self._meter.create_histogram(
            "air.compliance.scan.duration",
            unit="ms",
            description="Scan duration"
        )
        histogram.record(duration_ms, {"framework": framework})
        
        # Gauge for compliance score
        score = (passed / total * 100) if total > 0 else 0
        gauge = self._meter.create_gauge(
            "air.compliance.score",
            unit="%",
            description="Compliance score percentage"
        )
        gauge.record(score, {"framework": framework})
    
    def record_finding(self, article: int, severity: str, passed: bool):
        """
        Record individual compliance finding as metric.
        
        Args:
            article: EU AI Act article number (9, 10, 11, 12, 14, 15)
            severity: Finding severity ("CRITICAL", "HIGH", "MEDIUM", "LOW")
            passed: Whether the finding passed
        """
        counter = self._meter.create_counter(
            "air.compliance.findings",
            description="Compliance findings by article and severity"
        )
        counter.add(1, {
            "article": str(article),
            "severity": severity,
            "passed": str(passed)
        })
    
    def _record_span_event(
        self,
        name: str,
        attributes: Dict[str, Any],
        status: str,
        duration_ms: float
    ):
        """Record a span event for audit chain integration."""
        event = SpanEvent(
            name=name,
            timestamp=datetime.utcnow().isoformat() + "Z",
            attributes=attributes,
            status=status,
            duration_ms=duration_ms
        )
        
        with self._span_lock:
            self._spans.append(event)
            
            # Keep only last 1000 spans in memory
            if len(self._spans) > 1000:
                self._spans = self._spans[-1000:]
    
    def export_spans_json(self) -> List[Dict[str, Any]]:
        """
        Export collected spans as JSON for audit chain integration.
        
        Returns:
            List of span events as dictionaries
        """
        with self._span_lock:
            return [asdict(span) for span in self._spans]
    
    def clear_spans(self):
        """Clear recorded spans from memory."""
        with self._span_lock:
            self._spans.clear()


class ConsoleSpanExporter:
    """Simple console span exporter for debugging."""
    
    def export(self, spans):
        """Export spans to console."""
        for span in spans:
            print(f"[SPAN] {span.name}")
            print(f"  Status: {span.status.status_code}")
            print(f"  Duration: {span.end_time - span.start_time}s")
            print(f"  Attributes: {span.attributes}")
        return 0
    
    def shutdown(self):
        """Shutdown the exporter."""
        pass
    
    def force_flush(self, timeout_millis=30000):
        """Force flush any pending spans."""
        return True


# Global singleton instance
telemetry = AirTelemetry()
