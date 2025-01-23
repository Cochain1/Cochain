# kg_qa_runner.py

import logging
from typing import Dict
from knowledge_graph_qa import KnowledgeGraphQA

class KGQARunner:
    """
    Wrapper class for the Knowledge Graph Question Answering System.
    """

    def __init__(self, config: Dict):
        """
        Initialize the KGQARunner instance.

        Args:
            config (Dict): Dictionary of configuration parameters.
        """
        self.config = config
        self.qa_system = KnowledgeGraphQA(
            uri=config['database']['uri'],
            username=config['database']['username'],
            password=config['database']['password'],
            model_path=config['paths']['model_path'],
            category_map_path=config['paths']['category_map_path'],
            tmp_dir=config['paths']['tmp_dir'],
            custom_dict_path=config['paths']['custom_dict_path'],
            parameters=config.get('parameters', {})
        )
        logging.info("Knowledge Graph QA system has been initialized.")

    def get_knowledge(self, query_text: str) -> Dict[str, str]:
        """
        Retrieve knowledge and causal chain.

        Args:
            query_text (str): Query text.

        Returns:
            Dict[str, str]: Dictionary containing integrated knowledge and causal chain.
        """
        top_n = self.config.get('parameters', {}).get('top_n_nodes', 3)
        output = self.qa_system.main(query_text, top_n=top_n)
        integrated_knowledge = output.get('integrated_knowledge', '')
        causal_chain = output.get('causal_chain', '')
        return {
            'integrated_knowledge': integrated_knowledge,
            'causal_chain': causal_chain
        }