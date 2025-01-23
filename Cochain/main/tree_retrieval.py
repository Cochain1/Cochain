# tree_retrieval.py

import logging
from collections import defaultdict
from difflib import SequenceMatcher
from typing import List, Dict, Any

def find_most_similar_paths(
    query: str,
    tree_data: Dict[str, Any],
    max_merged_paths: int = None
) -> List[str]:
    """
    Find all paths that are most similar to the query.

    Args:
        query (str): The query string.
        tree_data (Dict[str, Any]): The tree data loaded from tree_data.json.
        max_merged_paths (int, optional): The maximum number of merged paths.

    Returns:
        List[str]: A list of the most similar paths.
    """
    def similarity(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

    def dfs(node: Dict[str, Any], node_type: str, current_path: List[str], all_paths: List[str]):
        """
        Perform a depth-first search to traverse the tree structure and generate all paths.

        Args:
            node (Dict[str, Any]): The current node.
            node_type (str): The type of the current node (e.g., "User Need", "Design Method", etc.).
            current_path (List[str]): The current path.
            all_paths (List[str]): The list of all paths.
        """
        # Add the current node to the path and prefix it with the node type
        if 'name' in node:
            current_path.append(f"{node_type}: {node['name']}")

        if node_type == "User Need":
            # Traverse design methods
            for design_method in node.get('design_methods', []):
                dfs(design_method, "Design Method", current_path, all_paths)
        elif node_type == "Design Method":
            # Traverse supply chain methods
            for supply_method in node.get('supply_methods', []):
                dfs(supply_method, "Supply Chain Method", current_path, all_paths)
        elif node_type == "Supply Chain Method":
            # Traverse production methods
            for produce_method in node.get('produce_methods', []):
                dfs(produce_method, "Production Method", current_path, all_paths)
        elif node_type == "Production Method":
            # Traverse quality inspection methods
            for quality_method in node.get('quality_methods', []):
                dfs(quality_method, "Quality Inspection Method", current_path, all_paths)

        if len(current_path) > 1: 
            all_paths.append(" -> ".join(current_path))

        if current_path:
            current_path.pop()

    user_needs = tree_data.get('user_need', [])
    most_similar_need = None
    highest_similarity = 0.0
    for user_need in user_needs:
        sim = similarity(query, user_need['name'])
        if sim > highest_similarity:
            highest_similarity = sim
            most_similar_need = user_need

    all_paths = []
    if most_similar_need:
        dfs(most_similar_need, "User Need", [], all_paths)
    else:
        logging.warning("No user need node similar to the query was found.")

    merged_paths = merge_similar_paths(all_paths, max_merged_paths)

    return merged_paths

def merge_similar_paths(paths: List[str], max_merged_paths: int = None) -> List[str]:
    """
    Merge identical parts of the paths.

    Args:
        paths (List[str]): A list of paths.
        max_merged_paths (int, optional): The maximum number of merged paths.

    Returns:
        List[str]: A list of merged paths.
    """
    merged_paths = defaultdict(set)
    for path in paths:
        parts = path.split(" -> ")
        if len(parts) >= 4:
            key = " -> ".join(parts[:4])
            rest = parts[4:]
            if rest:
                merged_paths[key].add(" -> ".join(rest))
            else:
                merged_paths[key].add('')
        else:
            key = " -> ".join(parts)
            merged_paths[key].add('')

    result = []
    for key, rest_parts in merged_paths.items():
        if '' in rest_parts and len(rest_parts) == 1:
            result.append(key)
        else:
            filtered_parts = [rp for rp in rest_parts if rp]
            if filtered_parts:
                unique_methods = sorted(set(filtered_parts))
                methods_str = ", ".join(unique_methods)
                result.append(f"{key} -> {methods_str}")
            else:
                result.append(key)

    if max_merged_paths is not None:
        result = result[:max_merged_paths]

    return result