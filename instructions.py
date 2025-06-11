detailedOneShot = """
You are given a math or physics problem. Please solve the problem in a structured, step-by-step manner as follows:

Step 1: Problem Interpretation and Variable Definition
Read the problem and translate it into mathematical symbols. Clearly list all known variables, conditions, and the unknowns to be solved.
Use letter (a, b, c, \alpha, \beta, \gamma, etc.) rather than natual language for variables.
Use # for comments to explain the meaning of variables.
Conditions are optional, only include them as a additional informtion known variables.
Place the step within <abstract> </abstract> tag.

Step 2: Calculation Process
Based on the variables and conditions from Step 1, provide detailed mathematical derivation or computation, showing each step and explanations if necessary.
Use LaTeX for mathematical expressions and symbols.
Place the step within <calculation> </calculation> tag.

Final: 
Answer the question with a number
Place the step within <answer> </answer> tag.

Example:

Input:
Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?

Response:
<symbols>
[Known variables]

- a = 48 # Clips sold in April
  9
  [Unknown variables]
- b # Clips sold in May
- c # Total clips sold in April and May, the final answer

[Conditions]

- clips sold in May is half of clips sold in April
</symbols>

<calculation>

- b = a / 2 = 48 / 2 = 24
- c = a + b = 48 + 24 = 72
</calculation>

<answer>
72
</answer>
"""

detailedZeroShot2 = """
You are given a math or physics problem. Please solve the problem in a structured, step-by-step manner as follows:

Step 1: Problem Interpretation and Variable Definition
Read the problem and translate it into mathematical symbols. Clearly list all known variables, conditions, and the unknowns to be solved.
Use letter (a, b, c, \alpha, \beta, \gamma, etc.) rather than natual language for variables.
Use # for comments to explain the meaning of variables.
Conditions are optional, only include them as a additional informtion known variables.
Place the step within <abstract> </abstract> tag.

Step 2: Calculation Process
Based on the variables and conditions from Step 1, provide detailed mathematical derivation or computation, showing each step and explanations if necessary.
Use LaTeX for mathematical expressions and symbols.
Place the step within <calculation> </calculation> tag.

Final: 
Answer the question with a number
Place the step within <answer> </answer> tag.

Then your response should be in the following format:

<symbols>
mathematical symbols and variable definition
</symbols>

<calculation>
Detailed calculation here
</calculation>

<answer>
Number as the answer
</answer>
"""

detailedZeroShot3 = """You are given a math or physics problem. Please solve the problem in a structured, step-by-step manner as follows:

Step 1: Variable Extraction
Carefully read the problem and extract all relevant variables given in the question.
Clearly list these variables with their symbols (use letters like a, b, c, \alpha, \beta, \gamma, etc.) and their meanings or values.
Place this list between {variablesStart} and {variablesEnd} tags.

Step 2: Formula Derivation
Based on the extracted variables, identify and derive the appropriate mathematical or physical formula(s) needed to solve the problem.
Explain the reasoning behind choosing or deriving the formula.
Use LaTeX for all formulas and mathematical expressions.
Place the formula derivation between {formulaStart} and {formulaEnd} tags.

Step 3: Calculation Process
Using the derived formula(s) and extracted variables, perform the detailed calculation step-by-step.
Show all intermediate steps and provide explanations if necessary.
Use LaTeX for all calculations.
Place this calculation process between {calculationStart} and {calculationEnd} tags.

Final:
Answer the question with a number
Place the step within <answer> </answer> tag.

Then your response should be in the following format:

<variables>
Extracted variables and their definitions
</variables>

<formula>
Derived formula(s) with explanation
</formula>

<calculation>
Step-by-step calculation
</calculation>

<answer>
Number as the answer
</answer>
"""

deepseekStyle = """
You are given a math or physics problem and try to solve the problem.
First, you interpret the problem into mathematical symbols including known variable, conditions and the unknowns to be solved.Then, you do calculation based on the known variables and conditions, to solve the unknowns.Finally, you answer the question with a number.
The interpretation, calculation and answer should be enclosed in <symbols></symbols>, <calculation></calculation> and <answer></answer> tags respectively, i.e.,
<symbols>
mathematical symbols here
</symbols>
<calculation>
calculation here
</calculation>
<answer>
number as answer here
</answer>
"""

minimal2 = """
You are given a math or physics problem. Please solve the problem and response in the following format:
<symbols>
mathematical symbols and variable definition
</symbols>
<calculation>
Detailed calculation here
</calculation>
<answer>
number as answer here
</answer>
"""

minimal3 = """
You are given a math or physics problem. Please solve the problem and response in the following format:
<variables>
extract variables from the question
</variables>
<formula>
derive formula based on variables you extracted
</formula>
<calculation>
compute the answer based on the formula
</calculation>
<answer>
number as answer here
</answer>
"""