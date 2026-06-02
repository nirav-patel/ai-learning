# GRADED FUNCTION: reflection_and_rewrite
def reflection_and_rewrite(report, model: str = "gpt-4o-mini", temperature: float = 0.3) -> dict:
    """
    Generates a structured reflection AND a revised research report.
    Accepts raw text OR the messages list returned by generate_research_report_with_tools.

    Returns:
        dict with keys:
          - "reflection": structured reflection text
          - "revised_report": improved version of the input report
    """

    # Input can be plain text or a list of messages, this function detects and parses accordingly
    report = research_tools.parse_input(report)

    ### START CODE HERE ###

    # Define the prompt. A multi-line f-string is typically used for this.
    # Remember it should ask the model to output ONLY valid JSON with this structure:
    # {{ "reflection": "<text>", "revised_report": "<text>" }}
    user_prompt = f"""You are an academic reviewer and editor.

Analyse the research report below and produce an improved version.

Output ONLY valid JSON — no markdown fences, no extra commentary. Use exactly this structure:
{{ "reflection": "<structured critique covering: Strengths, Limitations, Suggestions, Opportunities>", "revised_report": "<full improved report text>" }}

Rules for the reflection:
- Use the four headings: Strengths | Limitations | Suggestions | Opportunities
- Be specific: point to concrete passages and explain how to improve them.

Rules for the revised_report:
- Address every issue raised in the reflection.
- Keep the academic tone and all citations / URLs from the original.
- Return the FULL revised text (not a summary or diff).

Research report to review:
{report}
"""

    # Get a response from the LLM
    response = CLIENT.chat.completions.create( 
        # Pass in the model
        model=model,
        messages=[ 
            # System prompt is already defined
            {"role": "system", "content": "You are an academic reviewer and editor."},
            # Add user prompt
            {"role": "user", "content": user_prompt},
        ],
        # Set the temperature equal to the temperature parameter passed to the function
        temperature=temperature
    )

    ### END CODE HERE ###

    # Extract output
    llm_output = response.choices[0].message.content.strip()

    # Check if output is valid JSON
    try:
        data = json.loads(llm_output)
    except json.JSONDecodeError:
        raise Exception("The output of the LLM was not valid JSON. Adjust your prompt.")

    return {
        "reflection": str(data.get("reflection", "")).strip(),
        "revised_report": str(data.get("revised_report", "")).strip(),
    }
