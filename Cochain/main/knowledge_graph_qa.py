# knowledge_graph_qa.py

import os
import re
import logging
import json
import time
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import jieba
from neo4j import GraphDatabase
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import argparse
import coloredlogs


coloredlogs.install(
    level='INFO',
    fmt='%(asctime)s - %(levelname)s - %(message)s'
)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class KnowledgeGraphQA:
    """
    Knowledge Graph Question Answering System.
    """

    def __init__(self, uri: str, username: str, password: str, model_path: str, category_map_path: str, tmp_dir: str = './tmp', custom_dict_path: str = 'custom_dict.txt', parameters: Dict = {}):
        """
        Initialize the KnowledgeGraphQA instance.

        Args:
            uri (str): Neo4j database URI.
            username (str): Database username.
            password (str): Database password.
            model_path (str): Path to the pre-trained model.
            category_map_path (str): Path to the category mapping file.
            tmp_dir (str, optional): Path to the temporary directory.
            custom_dict_path (str, optional): Path to the custom dictionary file. Default is 'custom_dict.txt'.
            parameters (Dict, optional): Configuration parameters dictionary.
        """
        # Configure Neo4j connection information
        self.uri = uri
        self.username = username
        self.password = password
        self.model_path = model_path
        self.tmp_dir = tmp_dir
        self.custom_dict_path = custom_dict_path
        self.parameters = parameters

        # Set the temporary directory
        os.environ['TMPDIR'] = self.tmp_dir

        # Connect to the Neo4j database
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
                connection_timeout=60
            )
            logging.info("Successfully connected to the Neo4j database.")
        except Exception as e:
            logging.error(f"Error connecting to the Neo4j database: {e}")
            exit()

        # Initialize the pre-trained text encoder
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModel.from_pretrained(self.model_path)
            logging.info("Model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading the model: {e}")
            exit()

        self._create_fulltext_index_if_not_exists()

        try:
            with open(category_map_path, 'r', encoding='utf-8') as f:
                self.causal_category_map = json.load(f)
            logging.info("Category causality mapping loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading category causality mapping: {e}")
            self.causal_category_map = {}
            exit()

        self.load_custom_dict()

    def _create_fulltext_index_if_not_exists(self):
        """
        Check and create a full-text index if it does not exist.
        """
        with self.driver.session() as session:
            try:
                check_index_query = """
                SHOW INDEXES YIELD name
                WHERE name = 'entityNameIndex'
                RETURN name
                """
                index_exists = session.run(check_index_query).single()
                if not index_exists:
                    logging.info("Creating full-text index...")
                    session.write_transaction(self.create_fulltext_index)
                    logging.info("Full-text index created.")
                else:
                    logging.info("Full-text index already exists.")
            except Exception as e:
                logging.error(f"Error creating full-text index: {e}")

    @staticmethod
    def create_fulltext_index(tx):
        """
        Transaction function to create a full-text index.

        Args:
            tx: Neo4j transaction.
        """
        create_index_query = """
        CALL db.index.fulltext.createNodeIndex('entityNameIndex', ['Entity'], ['name'])
        """
        try:
            tx.run(create_index_query)
        except Exception as e:
            if "There already exists an index called" in str(e):
                logging.info("Full-text index already exists, skipping creation.")
            else:
                raise e

    def load_custom_dict(self):
        """
        Load a custom dictionary and add it to Jieba's dictionary.
        """
        if not os.path.exists(self.custom_dict_path):
            logging.warning(f"Custom dictionary file '{self.custom_dict_path}' does not exist.")
            return

        try:
            with open(self.custom_dict_path, 'r', encoding='utf-8') as f:
                custom_words = [line.strip() for line in f if line.strip()]
            for word in custom_words:
                jieba.add_word(word)
            logging.info(f"Custom dictionary loaded successfully, {len(custom_words)} words added.")
        except Exception as e:
            logging.error(f"Error loading custom dictionary: {e}")

    def extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from the input text.

        Args:
            text (str): Input text.

        Returns:
            List[str]: List of extracted keywords.
        """
        tokens = jieba.lcut(text)
        stopwords = {'how', 'of', 'and', 'is', 'in', 'through', 'to', 'improve depends on', 'balance depends on', '?'}
        keywords = [token for token in tokens if token not in stopwords and token.strip()]
        logging.debug(f"Extracted keywords: {keywords}")
        return keywords

    def filter_keywords(self, keywords: List[str], stopwords: set) -> List[str]:
        """
        Filter out stopwords from the keyword list.

        Args:
            keywords (List[str]): Original keyword list.
            stopwords (set): Set of stopwords.

        Returns:
            List[str]: Filtered keyword list.
        """
        filtered_keywords = [word for word in keywords if word not in stopwords]
        logging.debug(f"Filtered keywords: {filtered_keywords}")
        return filtered_keywords

    def run_query_with_retry(self, session, query: str, parameters: Dict[str, Any], retries: int = 3, delay: int = 5) -> List[Dict[str, Any]]:
        """
        Execute a query with a retry mechanism.

        Args:
            session: Neo4j session.
            query (str): Cypher query.
            parameters (Dict[str, Any]): Query parameters.
            retries (int, optional): Number of retries. Default is 3.
            delay (int, optional): Delay between retries in seconds. Default is 5.

        Returns:
            List[Dict[str, Any]]: List of query results.
        """
        for attempt in range(retries):
            try:
                result = session.run(query, parameters)
                return [record.data() for record in result]
            except Exception as e:
                logging.error(f"Error executing query: {e}")
                if attempt < retries - 1:
                    logging.info(f"Retrying attempt {attempt + 1}...")
                    time.sleep(delay)
                else:
                    logging.error("All retries failed.")
        return []

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of texts and return their embeddings.

        Args:
            texts (List[str]): List of texts.

        Returns:
            np.ndarray: NumPy array of text embeddings.
        """
        texts = [str(text) for text in texts]
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        token_embeddings = model_output.last_hidden_state  # (batch_size, sequence_length, hidden_size)
        attention_mask = encoded_input['attention_mask']  # (batch_size, sequence_length)

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embeddings = sum_embeddings / sum_mask  # (batch_size, hidden_size)

        return embeddings.numpy()

    def escape_special_characters(self, text):
        """
        Escape special characters in Lucene query syntax.
        """
        special_chars = '+-&&||!(){}[]^"~*?:\\/'
        escaped_text = ''
        for char in text:
            if char in special_chars:
                escaped_text += f'\\{char}'
            else:
                escaped_text += char
        return escaped_text

    def find_most_similar_nodes(self, session, query_text: str, k: int = 50, top_n: int = 3) -> Optional[List[Dict[str, Any]]]:
        """
        Find the top_n most relevant nodes (question concept nodes) based on the query text.

        Args:
            session: Neo4j session.
            query_text (str): Query text.
            k (int, optional): Initial number of nodes to retrieve. Default is 50.
            top_n (int, optional): Number of most similar nodes to return. Default is 3.

        Returns:
            Optional[List[Dict[str, Any]]]: List of most similar nodes, or None if none are found.
        """
        # Extract keywords
        keywords = self.extract_keywords(query_text)

        # Filter out stopwords
        stopwords = {"car design", "car production", "car manufacturing", "car quality inspection", "car supply chain"}
        filtered_keywords = self.filter_keywords(keywords, stopwords)
        logging.debug(f"Filtered keywords: {filtered_keywords}")

        if not filtered_keywords:
            logging.info("No keywords remaining after filtering.")
            return None

        escaped_keywords = [self.escape_special_characters(keyword) for keyword in filtered_keywords]
        logging.debug(f"Escaped keywords: {escaped_keywords}")

        search_string = ' OR '.join(escaped_keywords)
        logging.debug(f"Constructed search string: {search_string}")

        search_query = """
        CALL db.index.fulltext.queryNodes('entityNameIndex', $search_string) YIELD node, score
        RETURN node.name AS name, labels(node) AS labels, score
        ORDER BY score DESC
        LIMIT $k
        """

        results = self.run_query_with_retry(session, search_query, {'search_string': search_string, 'k': k})

        if not results:
            logging.info("No relevant entity nodes found.")
            return None

        # Get node texts (only use name as description to avoid noise)
        node_texts = [record['name'] for record in results]
        node_names = [record['name'] for record in results]

        logging.debug(f"Node texts: {node_texts}")

        filtered_text = ' '.join(filtered_keywords)
        logging.debug(f"Filtered query text: {filtered_text}")

        question_embedding = self.encode_texts([filtered_text])[0]
        node_embeddings = self.encode_texts(node_texts)

        logging.debug(f"Filtered query embedding: {question_embedding}")
        for i, node_name in enumerate(node_names):
            logging.debug(f"Node name: {node_name}, Embedding: {node_embeddings[i]}")

        # Calculate cosine similarity
        similarities = cosine_similarity([question_embedding], node_embeddings)[0]
        logging.debug(f"Similarities: {similarities}")

        # Find the top_n most similar nodes
        top_n_indices = np.argsort(similarities)[-top_n:][::-1]
        most_similar_nodes = []
        for idx in top_n_indices:
            most_similar_nodes.append({
                'name': node_names[idx],
                'score': similarities[idx]
            })

        logging.debug(f"Node similarities: {most_similar_nodes}")
        return most_similar_nodes

    def get_one_hop_neighbors(self, session, node_name: str) -> List[Dict[str, Any]]:
        """
        Get one-hop neighbors of the specified node, including incoming and outgoing edges.

        Args:
            session: Neo4j session.
            node_name (str): Node name.

        Returns:
            List[Dict[str, Any]]: List of one-hop neighbor relationships.
        """
        query = """
        MATCH (n:Entity {name: $node_name})-[r]->(m:Entity)
        RETURN n.name AS subject, type(r) AS predicate, m.name AS object, 'outbound' AS direction
        UNION
        MATCH (m:Entity)-[r]->(n:Entity {name: $node_name})
        RETURN m.name AS subject, type(r) AS predicate, n.name AS object, 'inbound' AS direction
        """
        return self.run_query_with_retry(session, query, {'node_name': node_name})

    def verbalize_triples(self, triples: List[Dict[str, Any]]) -> List[str]:
        """
        Convert triples into natural language sentences.

        Args:
            triples (List[Dict[str, Any]]): List of triples.

        Returns:
            List[str]: List of natural language sentences.
        """
        sentences = []
        for triple in triples:
            subject = triple['subject'].replace(" ", "")
            predicate = triple['predicate'].replace(" ", "")
            object_ = triple['object'].replace(" ", "")
            sentence = f"{subject}{predicate}{object_}"
            sentences.append(sentence)
        logging.debug(f"Verbalized triples: {sentences}")
        return sentences

    def integrate_triples(self, triples: List[str]) -> str:
        """
        Integrate all triples into a unified knowledge string, ensuring it ends with a period.

        Args:
            triples (List[str]): List of triple sentences.

        Returns:
            str: Integrated knowledge string.
        """
        if not triples:
            return ""
        if len(triples) == 1:
            return triples[0] + "."
        return ";".join(triples[:-1]) + ";" + triples[-1] + "."

    def get_causal_chains(self, session, node_name: str, min_depth: int = 2, max_depth: int = 5) -> List[List[Dict[str, Any]]]:
        """
        Get causal chains starting from the specified node, with a path length of at least min_depth steps.

        Args:
            session: Neo4j session.
            node_name (str): Starting node name.
            min_depth (int, optional): Minimum path depth. Default is 2.
            max_depth (int, optional): Maximum path depth. Default is 5.

        Returns:
            List[List[Dict[str, Any]]]: List of causal chains, each consisting of multiple triples.
        """
        # Build the starting node category condition
        start_category_condition = "'car design' IN start.category"

        # Cypher query to find all paths starting from the specified node with a path length between min_depth and max_depth
        query = f"""
        MATCH path = (start:Entity {{name: $node_name}})-[*{min_depth}..{max_depth}]->(end:Entity)
        WHERE {start_category_condition}
        RETURN nodes(path) AS nodes, [rel IN relationships(path) | type(rel)] AS rel_types
        LIMIT $limit
        """
        params = {
            'node_name': node_name,
            'limit': self.parameters.get('max_causal_chains', 3)
        }

        logging.debug(f"Generated query: {query}")
        logging.debug(f"With parameters: {params}")

        paths = self.run_query_with_retry(session, query, params)

        if not paths:
            logging.debug("No paths returned by the query.")
            return []

        causal_chains = []
        for path_record in paths:
            nodes = path_record['nodes']
            rel_types = path_record['rel_types']
            chain = []

            if len(rel_types) != len(nodes) - 1:
                logging.warning("Mismatch between node and relationship counts, skipping this path.")
                continue

            for i in range(len(rel_types)):
                subject_node = nodes[i]
                object_node = nodes[i + 1]
                relationship_type = rel_types[i]
                subject = subject_node.get('name', 'Unknown Entity')
                predicate = relationship_type or 'Unknown Relationship'
                object_ = object_node.get('name', 'Unknown Entity')
                chain.append({
                    "Entity1": subject,
                    "Relationship": predicate,
                    "Entity2": object_
                })
            causal_chains.append(chain)
            logging.debug(f"Identified causal chain: {chain}")

        logging.debug(f"Causal chains: {causal_chains}")
        return causal_chains

    def format_causal_chain(self, chain: List[Dict[str, Any]]) -> str:
        """
        Convert a causal chain into a natural language sentence, merging different predicates and objects for the same entity to reduce repetition.

        Args:
            chain (List[Dict[str, Any]]): List of causal chains.

        Returns:
            str: Formatted causal chain string.
        """
        sentences = []
        grouped_relations = {}

        # Group causal chains by Entity1
        for relation in chain:
            subject = relation["Entity1"]
            predicate = relation["Relationship"]
            object_ = relation["Entity2"]
            if subject not in grouped_relations:
                grouped_relations[subject] = {}
            if predicate not in grouped_relations[subject]:
                grouped_relations[subject][predicate] = set()
            grouped_relations[subject][predicate].add(object_)

        for subject, predicates in grouped_relations.items():
            parts = []
            for predicate, objects in predicates.items():
                object_str = ", ".join(objects)
                part = f"{predicate}{object_str}"
                parts.append(part)
            sentence = subject + "".join(f"{part}" if i == 0 else f", {part}" for i, part in enumerate(parts)) + "."
            sentences.append(sentence)

        result = "".join(sentences)
        return result

    def main(self, query_text: str, top_n: int) -> Dict[str, str]:
        """
        Main function to execute the knowledge graph question answering process.

        Args:
            query_text (str): Query text.
            top_n (int): Number of most similar nodes to return.

        Returns:
            Dict[str, str]: Result dictionary containing integrated knowledge and causal chain.
        """
        with self.driver.session() as session:
            most_similar_nodes = self.find_most_similar_nodes(session, query_text, top_n=top_n)
            if not most_similar_nodes:
                logging.info("No relevant nodes found.")
                return {'integrated_knowledge': '', 'causal_chain': ''}

            logging.debug("Most relevant nodes:")
            for node in most_similar_nodes:
                logging.debug(f"{node['name']}, Similarity: {node['score']:.4f}")

            all_triples = []
            for node in most_similar_nodes:
                triples = self.get_one_hop_neighbors(session, node['name'])
                verbalized_triples = self.verbalize_triples(triples)
                # Limit the number of triples integrated per node
                MAX_TRIPLES_PER_NODE = 10  # Set the maximum number of triples to integrate per node
                verbalized_triples = verbalized_triples[:MAX_TRIPLES_PER_NODE]
                all_triples.extend(verbalized_triples)

                logging.debug(f"Generated triples for node【{node['name']}】:")
                for sentence in verbalized_triples:
                    logging.debug(sentence)

            integrated_knowledge = self.integrate_triples(all_triples)
            logging.debug("Integrated knowledge:")
            logging.debug(integrated_knowledge)

            all_causal_relations = []
            causal_chain_text = ''
            for node in most_similar_nodes:
                causal_chains = self.get_causal_chains(
                    session,
                    node['name'],
                    min_depth=2,
                    max_depth=5  
                )
                if causal_chains:
                    logging.debug(f"Causal chains for node【{node['name']}】:")

                    for chain in causal_chains:
                        all_causal_relations.extend(chain)
                else:
                    logging.debug(f"No causal chains found for node【{node['name']}】.")

            if all_causal_relations:
                causal_chain_text = self.format_causal_chain(all_causal_relations)
                logging.debug("Integrated causal chain:")
                logging.debug(causal_chain_text)
            else:
                causal_chain_text = ''
                logging.debug("No causal chains found.")

            return {
                'integrated_knowledge': integrated_knowledge,
                'causal_chain': causal_chain_text
            }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge Graph Question Answering System")
    parser.add_argument('--query', type=str, help='Query question', required=False)
    parser.add_argument('--top_n', type=int, help='Top n most relevant nodes', required=False, default=3)
    parser.add_argument('--json_file', type=str, help='JSON file containing query questions', required=False, default='original_eval_data.json')
    parser.add_argument('--output_file', type=str, help='Output JSON file for results', required=False, default='kg_output_results.json')
    args = parser.parse_args()

    from data_loader import load_config
    config = load_config('config.yaml')

    qa_system = KnowledgeGraphQA(
        uri=config['database']['uri'],
        username=config['database']['username'],
        password=config['database']['password'],
        model_path=config['paths']['model_path'],
        category_map_path=config['paths']['category_map_path'],
        tmp_dir=config['paths']['tmp_dir'],
        custom_dict_path=config['paths'].get('custom_dict_path', 'custom_dict.txt')
    )

    results = []

    if args.json_file:
        with open(args.json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for item in tqdm(data, desc="Processing Knowledge Graph QA", unit="item"):
            query = item['instruction'].strip()
            output = qa_system.main(query, args.top_n)
            result = {
                'instruction': query,
                'integrated_knowledge': output['integrated_knowledge'],
                'causal_chain': output['causal_chain']
            }
            results.append(result)

    elif args.query:
        query = args.query.strip()
        logging.info(f"Processing question: {query}")
        output = qa_system.main(query, args.top_n)
        result = {
            'instruction': query,
            'integrated_knowledge': output['integrated_knowledge'],
            'causal_chain': output['causal_chain']
        }
        results.append(result)
    else:
        logging.error("Please provide a query question or an input JSON file.")
        exit(1)

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    logging.info(f"Results written to file: {args.output_file}")