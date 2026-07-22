"""
DP Partition Simulation CLI  (Method 2)

Usage:
  python run_dp_partition.py \
      --cluster-config cluster_config/single_node_multi_instance.json \
      --dataset dataset/sharegpt_req100_rate10_llama.jsonl \
      --num-req 100 --dp-count 2 \
      --output output/dp_partition_result.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inference_serving.dp_partition_sim import run_dp_partition_sim


def main():
    parser = argparse.ArgumentParser(description="LLMServingSim DP Partition Mode (Method 2)")
    parser.add_argument("--cluster-config", required=True,
                        help="DP cluster config (same as main.py)")
    parser.add_argument("--dataset",        required=True,
                        help="Request dataset JSONL (relative to repo root)")
    parser.add_argument("--output",         default=None,
                        help="Merged output CSV path")
    parser.add_argument("--num-req",        type=int, default=100)
    parser.add_argument("--dp-count",       type=int, default=2,
                        help="Number of DP partitions (usually = num_instances)")
    parser.add_argument("--fp",             type=int, default=16)
    parser.add_argument("--block-size",     type=int, default=16)
    parser.add_argument("--log-interval",   type=float, default=1.0)
    parser.add_argument("--verbose",        action="store_true",
                        help="Show each partition's simulation stdout")
    args = parser.parse_args()

    extra = [
        "--fp",           str(args.fp),
        "--block-size",   str(args.block_size),
        "--log-interval", str(args.log_interval),
    ]

    run_dp_partition_sim(
        dp_config_path=args.cluster_config,
        dp_count=args.dp_count,
        dataset_path=args.dataset,
        num_req=args.num_req,
        output_csv=args.output,
        extra_args=extra,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
