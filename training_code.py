from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def load_training_module():
    script_path = Path(__file__).resolve().parent / "SCRIPTS" / "training_code.py"
    spec = spec_from_file_location("training_script", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load training script from {script_path}")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    module = load_training_module()
    args = module.parse_args()
    module.train_model(
        input_csv=Path(args.input),
        output_model=Path(args.model_output),
        output_metrics=Path(args.metrics_output),
        max_rows=args.max_rows,
        n_estimators=args.n_estimators,
        n_jobs=args.n_jobs,
    )
