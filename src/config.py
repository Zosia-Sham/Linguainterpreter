from __future__ import annotations
import os
import re
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any, List, Dict

# -------------------- YAML loader with auto-install --------------------
def _ensure_yaml():
    try:
        import yaml  # type: ignore
        return yaml
    except Exception:
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "pip", "install", "-U", "pyyaml"], check=False)
        import yaml  # type: ignore
        return yaml

_yaml = _ensure_yaml()

# ${VAR} expansion in YAML strings
_env_pattern = re.compile(r"\$\{([A-Z0-9_]+)\}")

def _expand_env(v: Any) -> Any:
    if isinstance(v, str):
        # Expand ${VAR}, then ~ and %VAR%/$(VAR) via os.path.expandvars
        s = _env_pattern.sub(lambda m: os.getenv(m.group(1), ""), v)
        s = os.path.expanduser(s)
        s = os.path.expandvars(s)
        return s
    if isinstance(v, dict):
        return {k: _expand_env(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_expand_env(x) for x in v]
    return v


def load_dotenv_from_cwd() -> None:
    """Load `.env` from cwd so `${VAR}` placeholders in config YAML resolve before `from_yaml`."""
    try:
        from dotenv import load_dotenv

        load_dotenv(Path.cwd() / ".env", override=False)
    except Exception:
        pass


# -------------------- Config dataclasses --------------------
@dataclass
class RuntimeConfig:
    project_name: str = "ml_project"
    project_root: str = ""
    create_env: bool = True
    code_timeout_min: int = 30
    pip_timeout_min: int = 10
    predictive_buffer_pct: int = 50
    checker_timeout_cap_sec: int = 60
    verifier_timeout_cap_sec: int = 60
    default_task_budget_sec: int = 1800
    prediction_fallback_sec: int = 300
    min_exec_timeout_sec: int = 180
    bash_timeout_sec: int = 600
    metric_validation_retry_limit: int = 3
    router_retry_limit: int = 10
    generation_retry_limit: int = 20
    llm_retry_attempts: int = 30
    llm_retry_initial_delay_sec: int = 2
    llm_retry_max_delay_sec: int = 300
    execution_output_shorten_threshold: int = 16000
    execution_output_shorten_target: int = 10000
    replan_context_chars: int = 5000
    aggregate_tail_chars: int = 20000
    attach_hardware_limit_files: int = 3000

    @property
    def code_timeout_sec(self) -> int:
        return self.code_timeout_min * 60

    @property
    def pip_timeout_sec(self) -> int:
        return self.pip_timeout_min * 60

@dataclass
class PathsConfig:
    data_dir: str = "data"          # may be relative or absolute
    src_dir: str = "src"
    artifacts_dir: str = "artifacts"
    # Canonical submission path = project_root / submission_dir / submission_filename.
    # By default keep legacy behavior: project_root/submission.csv (submission_dir="").
    submission_dir: str = ""
    submission_filename: str = "submission.csv"
    logs_dir: str = "logs"
    tests_dir: str = "tests"
    scripts_dir: str = "scripts"
    venv_dir: str = ".venv"

@dataclass
class ProxyConfig:
    http: str = ""
    https: str = ""

@dataclass
class HardwareConfig:
    require_cuda: bool = False
    fail_if_no_cuda: bool = False
    cuda_devices: str = ""   # e.g., "0" or "0,1"

@dataclass
class OpenAIConfig:
    api_key: str = ""
    base_url: str = ""
    chat_model_strong: str = "gpt-4o"
    chat_model_fast: str = "gpt-4o-mini"
    temperature: float = 0.2

@dataclass
class AnthropicConfig:
    api_key: str = ""
    base_url: str = ""
    chat_model_strong: str = "claude-3-opus-20240229"
    chat_model_fast: str = "claude-3-haiku-20240307"
    temperature: float = 0.2

@dataclass
class GoogleConfig:
    api_key: str = ""
    cse_id: str = ""
    application_credentials: str = ""
    model_flash: str = "gemini-2.5-flash"
    model_pro: str = "gemini-2.5-pro"
    temperature: float = 0.2

@dataclass
class VertexConfig:
    project_id: str = ""
    location: str = "us-central1"
    model_flash: str = "gemini-2.5-flash"
    model_pro: str = "gemini-2.5-pro"
    temperature: float = 0.2
    application_credentials: str = ""

@dataclass
class OllamaConfig:
    enabled: bool = False
    base_url: str = "http://127.0.0.1:11434"
    model: str = "deepcoder"
    temperature: float = 0.4

@dataclass
class LLMConfig:
    prefer: str = "openai"  # openai | google | vertex | ollama | anthropic
    disable_ssl: bool = False
    # Relative to the config YAML directory (e.g. config.yaml next to llm_model_pricing.json).
    model_pricing_file: str = "llm_model_pricing.json"
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    google: GoogleConfig = field(default_factory=GoogleConfig)
    vertex: VertexConfig = field(default_factory=VertexConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    anthropic: AnthropicConfig = field(default_factory=AnthropicConfig)

# NEW: Orchestration section
@dataclass
class OrchestrationConfig:
    enforce_single_stack: bool = True
    allow_ensembles: bool = True
    metric_source: str = "auto"          # "auto" | "manual"
    require_metrics_json: bool = True
    min_metric_improvement_rel: float = 0.05
    check_fail_threshold: float = 0.25
    optimize_iters: int = 4
    max_tree_depth: int = 5
    max_tree_width: int = 4
    # If global remaining time falls below this, do not subdivide tasks (execute leaf-only).
    min_remaining_sec_to_split: int = 600
    # Main-branch tail replanning anti-thrashing controls.
    replan_max_calls: int = 3
    # Require at least N executed subtasks before allowing another replan.
    replan_cooldown_steps: int = 1
    # Improve loop: max YAML tasks per iteration (after ordering); planner must respect this.
    max_improve_tasks_per_iter: int = 7
    # ReAct meta-planner for the improver loop.
    # It receives a slice of remaining improve budget (time_budget_sec passed to prompt).
    meta_planner_time_pct: float = 0.25
    # How many times to retry meta-planning when output is missing/invalid.
    meta_planner_max_attempts: int = 3
    main_verifier_max_steps: int = 15
    improve_verifier_max_steps: int = 10
    total_budget_min: int = 480
    improve_budget_min: int = 120

    @property
    def total_budget_sec(self) -> int:
        return self.total_budget_min * 60

    @property
    def improve_budget_sec(self) -> int:
        return self.improve_budget_min * 60

# NEW: Preinstall section
@dataclass
class PreinstallConfig:
    enable: bool = True
    torch_cuda_index_url: str = ""       # e.g., https://download.pytorch.org/whl/cu128
    pkgs: List[str] = field(default_factory=list)

# NEW: Data-check section
@dataclass
class DataCheckConfig:
    enabled: bool = True
    max_samples_per_dir: int = 3
    probe_timeout_sec: int = 180
    react_max_rounds: int = 3

# NEW: MCP configuration
@dataclass
class MCPConfig:
    enabled: bool = False
    servers: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class AppConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    proxy: ProxyConfig = field(default_factory=ProxyConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    orchestration: OrchestrationConfig = field(default_factory=OrchestrationConfig)
    preinstall: PreinstallConfig = field(default_factory=PreinstallConfig)
    data_check: DataCheckConfig = field(default_factory=DataCheckConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)

    # ---------- Load & normalize ----------
    @classmethod
    def from_yaml(cls, path: str | Path) -> "AppConfig":
        data = _yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        data = _expand_env(data)
        js = json.dumps(data)
        obj = json.loads(js)

        conf = cls()

        if "runtime" in obj:
            conf.runtime = RuntimeConfig(**obj["runtime"])

        if "paths" in obj:
            conf.paths = PathsConfig(**obj["paths"])

        if "proxy" in obj:
            conf.proxy = ProxyConfig(**obj["proxy"])

        if "llm" in obj:
            llm_obj = obj["llm"]
            llm = LLMConfig(**{k: v for k, v in llm_obj.items() if k in {"prefer", "disable_ssl", "model_pricing_file"}})
            if "openai" in llm_obj:
                llm.openai = OpenAIConfig(**llm_obj["openai"])
            if "google" in llm_obj:
                llm.google = GoogleConfig(**llm_obj["google"])
            if "vertex" in llm_obj:
                llm.vertex = VertexConfig(**llm_obj["vertex"])
            if "ollama" in llm_obj:
                llm.ollama = OllamaConfig(**llm_obj["ollama"])
            if "anthropic" in llm_obj:
                llm.anthropic = AnthropicConfig(**llm_obj["anthropic"])
            conf.llm = llm

        if "hardware" in obj:
            conf.hardware = HardwareConfig(**obj["hardware"])

        if "orchestration" in obj:
            conf.orchestration = OrchestrationConfig(**obj["orchestration"])

        if "preinstall" in obj:
            # ensure list type for pkgs
            pre = obj["preinstall"]
            if "pkgs" in pre and not isinstance(pre["pkgs"], list):
                pre["pkgs"] = list(pre["pkgs"])
            conf.preinstall = PreinstallConfig(**pre)

        if "data_check" in obj:
            conf.data_check = DataCheckConfig(**obj["data_check"])

        if "mcp" in obj:
            conf.mcp = MCPConfig(**obj["mcp"])

        return conf

    # ---------- Env export ----------
    def apply_env(self) -> None:
        # CUDA mask
        if self.hardware.cuda_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.hardware.cuda_devices

        # proxies
        if self.proxy.http:
            os.environ["http_proxy"] = self.proxy.http
            os.environ["HTTP_PROXY"] = self.proxy.http
        if self.proxy.https:
            os.environ["https_proxy"] = self.proxy.https
            os.environ["HTTPS_PROXY"] = self.proxy.https

        # LLM keys
        if self.llm.openai.api_key:
            os.environ["OPENAI_API_KEY"] = self.llm.openai.api_key
        if self.llm.anthropic.api_key:
            os.environ["ANTHROPIC_API_KEY"] = self.llm.anthropic.api_key
        if self.llm.google.api_key:
            os.environ["GOOGLE_API_KEY"] = self.llm.google.api_key
        if self.llm.google.cse_id:
            os.environ["GOOGLE_CSE_ID"] = self.llm.google.cse_id
        if self.llm.google.application_credentials:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.llm.google.application_credentials

    # ---------- Path helpers ----------
    def abspath(self, root: Path, *rel: str) -> Path:
        return (root / Path(*rel)).resolve()

    def as_dict(self) -> Dict[str, Any]:
        # Comment translated to English.
        return {
            "runtime": self.runtime.__dict__,
            "paths": self.paths.__dict__,
            "proxy": self.proxy.__dict__,
            "llm": {
                "prefer": self.llm.prefer,
                "disable_ssl": self.llm.disable_ssl,
                "model_pricing_file": self.llm.model_pricing_file,
                "openai": self.llm.openai.__dict__,
                "anthropic": self.llm.anthropic.__dict__,
                "google": self.llm.google.__dict__,
                "vertex": self.llm.vertex.__dict__,
                "ollama": self.llm.ollama.__dict__,
            },
            "hardware": self.hardware.__dict__,
            "orchestration": self.orchestration.__dict__,
            "preinstall": {"enable": self.preinstall.enable,
                           "torch_cuda_index_url": self.preinstall.torch_cuda_index_url,
                           "pkgs": list(self.preinstall.pkgs)},
            "data_check": self.data_check.__dict__,
            "mcp": {
                "enabled": self.mcp.enabled,
                "servers": self.mcp.servers
            }
        }
