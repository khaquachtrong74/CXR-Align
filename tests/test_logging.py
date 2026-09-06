import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.train import Trainer
from src.visualize_utils import Visualizer


def test_log_metrics_does_not_mutate_visualization_metrics(tmp_path):
    trainer = Trainer.__new__(Trainer)
    trainer.step = 7
    trainer.epoch = 2
    trainer.log_file = tmp_path / "train_log.jsonl"
    metrics = {"total_loss": 1.25, "temperature_cross": 0.07}

    trainer.log_metrics(metrics, "train_epoch")

    assert metrics == {"total_loss": 1.25, "temperature_cross": 0.07}
    logged = json.loads(trainer.log_file.read_text().strip())
    assert logged["phase"] == "train_epoch"
    assert logged["step"] == 7
    assert logged["epoch"] == 2


def test_epoch_summary_metric_formatter_accepts_metadata():
    assert Visualizer._format_metric(0.123456) == "0.1235"
    assert Visualizer._format_metric("train_epoch") == "train_epoch"
