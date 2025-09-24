#!/usr/bin/env python3

import json
import os
import argparse
import sys
from pathlib import Path


def process_jsonl_file(input_path: str, output_path: str) -> None:
    """
    Process a JSONL file and create output with ID extraction and correctness marking.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created directory: {output_dir}")
    
    # Process the input file
    processed_count = 0
    error_count = 0
    
    try:
        with open(input_path, 'r', encoding='utf-8') as input_file, \
             open(output_path, 'w', encoding='utf-8') as output_file:
            
            for line_num, line in enumerate(input_file, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                
                try:
                    # Parse JSON line
                    data = json.loads(line)
                    
                    # Extract ID if present
                    extracted_id = data.get("id", None)
                    
                    # Create output record
                    output_record = {
                        "id": extracted_id,
                        "original_answer_correctness": "✅"
                    }
                    
                    # Write to output file
                    output_file.write(json.dumps(output_record, ensure_ascii=False) + '\n')
                    processed_count += 1
                    
                    if line_num % 100 == 0:
                        print(f"Processed {line_num} lines...")
                        
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON on line {line_num}: {e}")
                    error_count += 1
                    continue
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}")
                    error_count += 1
                    continue
    
    except FileNotFoundError:
        print(f"Error: Input file '{input_path}' not found.")
        sys.exit(1)
    except PermissionError:
        print(f"Error: Permission denied accessing files.")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)
    
    # Print summary
    print(f"\nProcessing completed:")
    print(f"  Input file: {input_path}")
    print(f"  Output file: {output_path}")
    print(f"  Successfully processed: {processed_count} lines")
    if error_count > 0:
        print(f"  Errors encountered: {error_count} lines")


def validate_files(input_path: str, output_path: str) -> bool:
    """
    Validate input and output file paths.
    
    Args:
        input_path: Path to input file
        output_path: Path to output file
        
    Returns:
        True if validation passes, False otherwise
    """
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' does not exist.")
        return False
    
    # Check if input file is readable
    if not os.access(input_path, os.R_OK):
        print(f"Error: Input file '{input_path}' is not readable.")
        return False
    
    # Check if output path is valid
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            # Test if we can create the directory
            os.makedirs(output_dir, exist_ok=True)
            os.rmdir(output_dir)  # Remove test directory
        except Exception as e:
            print(f"Error: Cannot create output directory '{output_dir}': {e}")
            return False
    
    return True


def main():
    """Main function to handle command line arguments and execute processing."""
    parser = argparse.ArgumentParser(
        description="Process JSONL file to extract IDs and mark correctness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/check_jsonl_ids.py input.jsonl output.jsonl
  python scripts/check_jsonl_ids.py --input data/test.jsonl --output results/processed.jsonl
        """
    )
    
    parser.add_argument(
        'input_file',
        nargs='?',
        help='Input JSONL file path'
    )
    
    parser.add_argument(
        'output_file',
        nargs='?',
        help='Output JSONL file path'
    )
    
    parser.add_argument(
        '--input', '-i',
        dest='input_path',
        help='Input JSONL file path (alternative to positional argument)'
    )
    
    parser.add_argument(
        '--output', '-o',
        dest='output_path',
        help='Output JSONL file path (alternative to positional argument)'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0.0'
    )
    
    args = parser.parse_args()
    
    # Determine input and output paths
    input_path = args.input_path or args.input_file
    output_path = args.output_path or args.output_file
    
    # Validate arguments
    if not input_path or not output_path:
        parser.print_help()
        print("\nError: Both input and output file paths are required.")
        sys.exit(1)
    
    # Convert to absolute paths
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)
    
    print(f"JSONL ID Checker and Correctness Marker")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print("-" * 50)
    
    # Validate files
    if not validate_files(input_path, output_path):
        sys.exit(1)
    
    # Process the files
    try:
        process_jsonl_file(input_path, output_path)
        print("\n✅ Processing completed successfully!")
    except KeyboardInterrupt:
        print("\n❌ Processing interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
