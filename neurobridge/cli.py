"""NeuroBridge CLI — analyze organoid data from the command line.

Usage:
    neurobridge analyze recording.csv --analysis organoid_iq
    neurobridge analyze recording.csv --full-report
    neurobridge generate --duration 60 --output synthetic.csv
    neurobridge server --port 8847
"""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="neurobridge",
        description="NeuroBridge — Analysis engine for brain organoid data",
    )
    sub = parser.add_subparsers(dest="command")

    # analyze
    analyze_p = sub.add_parser("analyze", help="Analyze a spike data file")
    analyze_p.add_argument("file", help="Path to spike data file (CSV, HDF5, Parquet, JSON)")
    analyze_p.add_argument("--analysis", "-a", default="summary", help="Analysis to run (default: summary)")
    analyze_p.add_argument("--full-report", "-f", action="store_true", help="Run all analyses")
    analyze_p.add_argument("--output", "-o", help="Output JSON file (default: stdout)")

    # generate
    gen_p = sub.add_parser("generate", help="Generate synthetic spike data")
    gen_p.add_argument("--duration", type=float, default=30.0, help="Duration in seconds")
    gen_p.add_argument("--electrodes", type=int, default=8, help="Number of electrodes")
    gen_p.add_argument("--output", "-o", default="synthetic_spikes.csv", help="Output CSV file")

    # server
    server_p = sub.add_parser("server", help="Start the NeuroBridge API server")
    server_p.add_argument("--port", type=int, default=8847, help="Port number")
    server_p.add_argument("--host", default="0.0.0.0", help="Host to bind")

    # list
    sub.add_parser("list", help="List all available analyses")

    args = parser.parse_args()

    if args.command == "analyze":
        from neurobridge import load, analyze, full_report
        data = load(args.file)
        print(f"Loaded {data.n_spikes} spikes, {data.n_electrodes} electrodes, {data.duration:.1f}s", file=sys.stderr)

        if args.full_report:
            result = full_report(data)
        else:
            result = analyze(data, args.analysis)

        output = json.dumps(result, indent=2, default=str)
        if args.output:
            with open(args.output, "w") as f:
                f.write(output)
            print(f"Results written to {args.output}", file=sys.stderr)
        else:
            print(output)

    elif args.command == "generate":
        from neurobridge import generate_synthetic
        import numpy as np
        data = generate_synthetic(duration=args.duration, n_electrodes=args.electrodes)
        # Save as CSV
        import pandas as pd
        df = pd.DataFrame({
            "time": data.times,
            "electrode": data.electrodes,
            "amplitude": data.amplitudes,
        })
        df.to_csv(args.output, index=False)
        print(f"Generated {data.n_spikes} spikes → {args.output}", file=sys.stderr)

    elif args.command == "server":
        import uvicorn
        print(f"Starting NeuroBridge API on {args.host}:{args.port}", file=sys.stderr)
        uvicorn.run("main:app", host=args.host, port=args.port, reload=True)

    elif args.command == "list":
        from neurobridge.core import list_analyses
        for name in list_analyses():
            print(f"  {name}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
