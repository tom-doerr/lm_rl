#!/usr/bin/env python3

'''
This is a very simple script that uses huggingfaces to download a transformer 
which completes the text the user inputs.
'''

import argparse
import os
import sys
import logging

from transformers import pipeline

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-s', '--seed', help='The seed text to complete.', required=True)
    parser.add_argument('-n', '--number', help='The number of completions to generate.', type=int, default=5)
    parser.add_argument('-l', '--length', help='The number of tokens to complete to.', type=int, default=20)
    parser.add_argument('-o', '--output', help='The file to write completions to. If not specified, completions will be written to stdout.')
    parser.add_argument('-v', '--verbose', help='Print debug information to stderr.', action='store_true')
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    logger.info("Downloading pipeline...")
    # Set the pipline to complete the text.
    # pipeline_name = "fill-mask"
    # pipeline_name = "ner"
    # pipeline_name = "feature-extraction"
    # pipeline_name = "question-answering"
    pipeline_name = "text-generation"

    # pipeline_kwargs = {'max_length': args.length}
    pipeline_kwargs = {}
    pipeline_class = pipeline(pipeline_name, **pipeline_kwargs)

    logger.info("Generating completions...")
    # completions = pipeline_class(args.seed)
    # completions = pipeline_class('Customer complaint: <mask>')
    completions = pipeline_class('Customer complaint: ')

    if args.output is None:
        for completion in completions:
            print(completion)
    else:
        with open(args.output, 'w') as f:
            for completion in completions:
                f.write(completion + '\n')

    logger.info("Done.")

if __name__ == '__main__':
    main()

