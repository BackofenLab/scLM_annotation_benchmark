

def get_prompt_template():
    prompt_template = """You are a scientific expert in the field of cell biology.
You are given a gene perturbation experiment in K562 cells, where the CCND3 gene is perturbed.
Your task is to analyze the effect of this perturbation on the expression of the SDE2 gene.
Please provide a response indicating whether the expression of SDE2 is upregulated or downregulated as a result of the perturbation.
Respond with 'Yes' if SDE2 is upregulated, and 'No' if it is downregulated.
"""    
    return prompt_template