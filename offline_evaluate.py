import argparse
import os
from src.scorer.phybench import phybench_evaluate
from src.scorer.ABench import ABench_evaluate

def main():
    parser = argparse.ArgumentParser(description="Offline Evaluation for LLMs on Physics Benchmarks")
    parser.add_argument("--input", required=True, help="Path to the test questions (JSONL format)")
    parser.add_argument("--cleaned_input", default=None, help="Path to cleaned test questions (JSONL format), optional")
    parser.add_argument("--results", required=True, help="Path to the model's answers (JSONL format)")
    parser.add_argument("--output_dir", required=True, help="Directory to save evaluation results")
    parser.add_argument("--scheme", choices=["phybench", "ABench"], default="phybench", 
                      help="Evaluation scheme to use (default: phybench)")

    args = parser.parse_args()

    try:
        if args.scheme == "phybench":
            phybench_evaluate.run_evaluation(
                test_path=args.input,
                cleaned_input=args.cleaned_input,
                results_path=args.results,
                output_dir=args.output_dir
            )
        elif args.scheme == "ABench":
            ABench_evaluate.run_evaluation(
                test_path=args.input,
                results_path=args.results,
                output_dir=args.output_dir
            )
        else:
            raise ValueError("Unsupported evaluation scheme specified.")
            
        print(f"Finished！Results saved at: {os.path.abspath(args.output_dir)}")
        
    except Exception as e:
        import traceback
        print(f"Failed: {repr(e)}")
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    main()
