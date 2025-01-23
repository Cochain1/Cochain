# main.py

import logging
from tqdm import tqdm
from typing import List, Dict
from data_loader import load_eval_data, load_tree_data, load_config
from tree_retrieval import find_most_similar_paths, merge_similar_paths
from kg_qa_runner import KGQARunner
from instruction_integrator import integrate_instruction
import json
import os

def main():
    config = load_config('config.yaml')

    logging_level = getattr(logging, config.get('logging', {}).get('level', 'INFO').upper(), logging.INFO)
    logging.basicConfig(
        level=logging_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    if config.get('logging', {}).get('use_coloredlogs', False):
        import coloredlogs
        coloredlogs.install(
            level=logging_level,
            fmt='%(asctime)s - %(levelname)s - %(message)s'
        )

    eval_data = load_eval_data(config['paths']['eval_data'])

    updated_eval_data = []

    kg_output_file = config['paths']['kg_output']

    if os.path.exists(kg_output_file):
        logging.info(f"Detected that the file '{kg_output_file}' already exists, skipping the execution of the Knowledge Graph QA system.")
        with open(kg_output_file, 'r', encoding='utf-8') as f:
            kg_results = json.load(f)
    else:
        kg_runner = KGQARunner(config)

        kg_results = []
        for item in tqdm(eval_data, desc="Processing Knowledge Graph QA", unit="item"):
            query_text = item.get('instruction', "")
            if not query_text:
                logging.warning("The 'instruction' field is missing in the evaluation data.")
                continue

            kg_output = kg_runner.get_knowledge(query_text)
            integrated_knowledge = kg_output.get('integrated_knowledge', '')
            causal_chain = kg_output.get('causal_chain', '')
            result = {
                'instruction': query_text,
                'integrated_knowledge': integrated_knowledge,
                'causal_chain': causal_chain
            }
            kg_results.append(result)

        with open(kg_output_file, 'w', encoding='utf-8') as f:
            json.dump(kg_results, f, ensure_ascii=False, indent=4)
        logging.info(f"Knowledge Graph QA results have been saved to '{kg_output_file}'.")

    kg_results_dict = {item['instruction']: item for item in kg_results}

    tree_data = load_tree_data(config['paths']['tree_data'])

    parameters = config.get('parameters', {})
    max_merged_paths = parameters.get('max_merged_paths', None)
    max_causal_chains = parameters.get('max_causal_chains', None)

    for item in tqdm(eval_data, desc="Processing Final Output", unit="item"):
        query_text = item.get('instruction', "")
        if not query_text:
            logging.warning("The 'instruction' field is missing in the evaluation data.")
            continue

        kg_result = kg_results_dict.get(query_text, {})
        integrated_knowledge = kg_result.get('integrated_knowledge', '')
        causal_chain = kg_result.get('causal_chain', '')

        similar_paths = find_most_similar_paths(query_text, tree_data, max_merged_paths)

        merged_paths = merge_similar_paths(similar_paths, max_merged_paths)

        final_instruction = integrate_instruction(
            query_text,
            integrated_knowledge,
            causal_chain,
            merged_paths,
            max_causal_chains=max_causal_chains,
            max_merged_paths=max_merged_paths
        )

        updated_item = item.copy()
        updated_item['instruction'] = final_instruction
        updated_eval_data.append(updated_item)

    try:
        with open(config['paths']['output_file'], 'w', encoding='utf-8') as f_out:
            json.dump(updated_eval_data, f_out, ensure_ascii=False, indent=4)
        logging.info(f"\nIntegrated content has been saved to '{config['paths']['output_file']}'.")
    except Exception as e:
        logging.error(f"Error saving integrated content: {e}")

if __name__ == "__main__":
    main()