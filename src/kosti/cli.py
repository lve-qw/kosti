"""Command-line entry points for offline audit, inference and serving."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="kosti")
    sub = result.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit", help="Inspect DICOM files and optional expert workbook")
    audit.add_argument("input", type=Path)
    audit.add_argument("--labels", type=Path)
    audit.add_argument("--output", type=Path, required=True)
    template = sub.add_parser("annotation-template", help="Create blank review CSV from an audit; no inferred labels")
    template.add_argument("audit", type=Path)
    template.add_argument("--output", type=Path, required=True)
    folds = sub.add_parser("prepare-folds", help="Validate reviewed labels and freeze grouped folds; no training")
    folds.add_argument("audit", type=Path)
    folds.add_argument("--annotations", type=Path, required=True)
    folds.add_argument("--output", type=Path, required=True)
    folds.add_argument("--n-splits", type=int, default=3)
    folds.add_argument("--seed", type=int, default=42)
    evaluate = sub.add_parser("evaluate", help="Evaluate existing held-out probabilities; no training")
    evaluate.add_argument("manifest", type=Path)
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    configure = sub.add_parser("training-config", help="Create an offline training configuration; no training")
    configure.add_argument("manifest", type=Path)
    configure.add_argument("--data-root", type=Path, required=True)
    configure.add_argument("--initial-weights", type=Path, required=True)
    configure.add_argument("--output", type=Path, required=True, help="Future run directory")
    configure.add_argument("--config", type=Path, required=True, help="New configuration JSON")
    configure.add_argument("--mode", choices=("cross_validation", "final"), default="cross_validation")
    train = sub.add_parser("train", help="Validate training inputs; optimization requires --execute")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--execute", action="store_true", help="Actually optimize models; omitted by default")
    batch = sub.add_parser("batch", help="Run an explicit trained quality model offline")
    batch.add_argument("input", type=Path)
    batch.add_argument("--model", type=Path, required=True, help="Quality bundle manifest JSON")
    batch.add_argument("--output", type=Path, required=True)
    serve = sub.add_parser("serve", help="Serve batch API; no model means readiness is 503")
    serve.add_argument("--model", type=Path, default=os.getenv("KOSTI_MODEL_MANIFEST"))
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "audit":
            from kosti.dataset import audit_dataset
            report = audit_dataset(args.input, labels_path=args.labels)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Audit written to {args.output}")
            return 0

        if args.command in ("annotation-template", "prepare-folds", "evaluate"):
            from kosti.preparation import annotation_template, prepare_folds, evaluate_predictions
            if args.command == "annotation-template":
                annotation_template(args.audit, args.output)
            elif args.command == "prepare-folds":
                prepare_folds(args.audit, args.annotations, args.output, args.n_splits, args.seed)
            else:
                evaluate_predictions(args.manifest, args.predictions, args.output)
            print(f"Written to {args.output}; no training performed")
            return 0

        if args.command == "training-config":
            from kosti.training import create_config
            create_config(args.manifest, args.data_root, args.initial_weights, args.output, args.config, args.mode)
            print(f"Configuration written to {args.config}; no training performed")
            return 0

        if args.command == "train":
            from kosti.training import run_training
            print(json.dumps(run_training(args.config, execute=args.execute), ensure_ascii=False, indent=2))
            return 0

        from kosti.ml import load_quality_predictor
        predictor = load_quality_predictor(args.model) if args.model else None
        if args.command == "serve":
            import uvicorn
            from kosti.api import create_app
            uvicorn.run(create_app(predictor=predictor), host=args.host, port=args.port)
            return 0

        from kosti.archive import safe_extract_zip
        from kosti.pipeline import process_files, write_csv
        if not args.input.exists():
            raise ValueError(f"Input does not exist: {args.input}")
        with tempfile.TemporaryDirectory(prefix="kosti-cli-") as temporary:
            if args.input.is_file() and args.input.suffix.lower() == ".zip":
                root = Path(temporary)
                paths = safe_extract_zip(args.input, root)
            elif args.input.is_dir():
                root = args.input
                paths = sorted(p for p in root.rglob("*") if p.is_file() and not p.name.startswith("."))
            else:
                root = args.input.parent
                paths = [args.input]
            if not paths:
                raise ValueError("Input contains no files")
            rows, diagnostics = process_files(paths, predictor, display_root=root)
            args.output.mkdir(parents=True, exist_ok=True)
            write_csv(rows, args.output / "results.csv")
            (args.output / "diagnostics.json").write_text(
                json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"Processed {len(rows)} files; failures: {sum(r.processing_status == 'Failure' for r in rows)}")
            return 1 if any(r.processing_status == "Failure" for r in rows) else 0
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        print(f"kosti: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
