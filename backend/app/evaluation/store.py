import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from backend.app.config import settings
from backend.app.evaluation.schemas import RunReport

class EvaluationStore:
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or (Path(settings.DATA_DIR) / "evaluation")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, RunReport] = {}
        self._load_all()

    def _load_all(self):
        if not self.data_dir.exists():
            return
        for file_path in self.data_dir.glob("run_*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    report = RunReport(**data)
                    self._cache[report.run_id] = report
            except Exception:
                pass

    def save_run(self, report: RunReport) -> RunReport:
        self._cache[report.run_id] = report
        file_path = self.data_dir / f"run_{report.run_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))
        return report

    def get_run(self, run_id: str) -> Optional[RunReport]:
        return self._cache.get(run_id)

    def list_runs(self) -> List[RunReport]:
        return sorted(list(self._cache.values()), key=lambda r: r.timestamp, reverse=True)

    def delete_run(self, run_id: str) -> bool:
        if run_id in self._cache:
            del self._cache[run_id]
            file_path = self.data_dir / f"run_{run_id}.json"
            if file_path.exists():
                file_path.unlink()
            return True
        return False
