#!/usr/bin/env python3
import argparse
import sys
from mt_benchmark.models.factory import ModelFactory

def main():
    parser = argparse.ArgumentParser(description='Machine Translation Evaluation Pipeline')
    parser.add_argument('model_id', help='Model ID (e.g., nllb_200_3.3B)')
    
    args = parser.parse_args()
    
    try:
        # Load model
        print(f"Loading model: {args.model_id}")
        model = ModelFactory.create_model(args.model_id)
        
        # check for languages supported
        supported_languages = model.get_supported_languages()
        if not supported_languages:
            print("No supported languages found.", file=sys.stderr)
            sys.exit(1)

        print("Supported languages:")
        for lang in supported_languages:
            print(f" - {lang}: {supported_languages[lang]['name']}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()