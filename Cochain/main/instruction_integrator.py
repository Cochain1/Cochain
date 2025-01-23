# instruction_integrator.py

from typing import List

def integrate_instruction(
    query_text: str,
    integrated_knowledge: str,
    causal_chain: str,
    hints: List[str],
    max_causal_chains: int = None,
    max_merged_paths: int = None
) -> str:
    """
    Integrate knowledge, causal chains, and hints into the final instruction.

    Args:
        query_text (str): Original query text.
        integrated_knowledge (str): Integrated knowledge text.
        causal_chain (str): Causal chain text.
        hints (List[str]): List of hints.
        max_causal_chains (int, optional): Maximum number of causal chains.
        max_merged_paths (int, optional): Maximum number of merged paths.

    Returns:
        str: Final integrated instruction text.
    """
    def determine_level(path: str) -> int:
        return path.count("->")

    hints.sort(key=lambda x: determine_level(x))
    if max_merged_paths is not None:
        hints = hints[:max_merged_paths]

    hint_text = "\n".join([f"Level {determine_level(path)}: {path}" for path in hints])
    if max_causal_chains is not None and causal_chain:
        causal_chain_lines = causal_chain.split('\n')
        causal_chain = '\n'.join(causal_chain_lines[:max_causal_chains])

    if causal_chain:
        integrated_output = (
            f"You are an expert in the field of automotive design. When addressing the following user needs relating to automotive design,"
            f"please provide a comprehensive answer that considers the entire business workflow, utilizing the provided knowledge and following the existing prompts."
            f"\nUser Need:{query_text}"
            f"\nKnowledge:{integrated_knowledge}"
            f"\nCausal chains:{causal_chain}"
            f"\nPrompts regarding the entire automotive business workflow:{hint_text}"
            f"\nPlease provide an answer to the question based on the above knowledge and prompts."
        )
    else:
        integrated_output = (
            f"You are an expert in the field of automotive design. When addressing the following user needs relating to automotive design,"
            f"please provide a comprehensive answer that considers the entire business workflow, utilizing the provided knowledge and following the existing prompts."
            f"\nUser Need:{query_text}"
            f"\nKnowledge:{integrated_knowledge}"
            f"\nPrompts regarding the entire automotive business workflow:{hint_text}"
            f"\nPlease provide an answer to the question based on the above knowledge and prompts."
        )

    return integrated_output