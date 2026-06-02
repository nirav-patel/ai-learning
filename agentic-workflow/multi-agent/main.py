"""
Entry point for the multi-agent sunglasses campaign pipeline.

Usage:
    python main.py
    python main.py --output-dir ./my-output --output-file my_report.md
    python main.py --log-level DEBUG
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="multi-agent-campaign",
        description="Run the autonomous multi-agent sunglasses campaign pipeline.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directory to save generated image and report (default: ./output)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        metavar="FILENAME",
        help="Markdown report filename (default: campaign_summary_<timestamp>.md)",
    )
    parser.add_argument(
        "--caption-style",
        type=str,
        default="short and punchy",
        metavar="STYLE",
        help='Style hint for the campaign caption (default: "short and punchy")',
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        # Lazy import so --help works without API keys configured
        from pipeline.campaign_pipeline import CampaignPipeline

        pipeline = CampaignPipeline.from_env(output_dir=args.output_dir)
        result = pipeline.run(
            output_filename=args.output_file,
            caption_style=args.caption_style,
        )

        print("\n✅ Pipeline complete!")
        print(f"   📊 Trend summary: {len(result.trend_summary)} chars")
        print(f"   🖼️  Image:         {result.image_path}")
        print(f"   ✍️  Quote:         {result.quote}")
        print(f"   📄 Report:        {result.report_path}")
        print(f"   ⏱️  Duration:      {result.metadata.get('duration_seconds', 0):.1f}s")
        return 0

    except EnvironmentError as exc:
        print(f"\n❌ Configuration error: {exc}", file=sys.stderr)
        print("   See .env.example for required environment variables.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
