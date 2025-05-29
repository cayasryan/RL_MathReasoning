COT_HEADER = """<|start_header_id|>user<|end_header_id|>

Solve the following math problem efficiently and clearly:

- For simple problems (2 steps or fewer):
  Provide a concise solution with minimal explanation.

- For complex problems (3 steps or more):
  Use this step-by-step format:

## Step 1: [Concise description]
[Brief explanation and calculations]

## Step 2: [Concise description]
[Brief explanation and calculations]

...

Always conclude with:
Therefore, the final answer is: \\boxed{answer}. I hope it is correct.
"""

# ---- ONE-SHOT EXAMPLE ----
EX_PROB = "If \\det \\mathbf{A}=2 and \\det \\mathbf{B}=12, find \\det(\\mathbf{A}\\mathbf{B})."
EX_SOL  = """## Step 1: Use det(AB)=det(A)det(B)
## Step 2: Substitute 2·12
Therefore, the final answer is: \\boxed{24}. I hope it is correct."""

PROMPT_PREFIX = (
    f"{COT_HEADER}\n\n"
    f"Problem: {EX_PROB}<|eot_id|>"
    f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    f"{EX_SOL}<|eot_id|>"
)
