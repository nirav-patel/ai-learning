# GRADED FUNCTION: convert_report_to_html
def convert_report_to_html(report, model: str = "gpt-4o", temperature: float = 0.5) -> str:
    """
    Converts a plaintext research report into a styled HTML page using OpenAI.
    Accepts raw text OR the messages list from the tool-calling step.
    """

    # Input can be plain text or a list of messages, this function detects and parses accordingly
    report = research_tools.parse_input(report)

    # System prompt is already provided
    system_prompt = "You convert plaintext reports into full clean HTML documents."

    ### START CODE HERE ###
    
    # Build the user prompt instructing the model to return ONLY valid HTML
    user_prompt = f"""Convert the research report below into a complete, well-structured HTML document.

Requirements:
- Return ONLY valid HTML — no markdown, no code fences, no commentary.
- Include a proper <html>, <head> (with <meta charset="UTF-8"> and a <style> block), and <body>.
- Use semantic tags: <h1> for the title, <h2> for section headers, <p> for paragraphs.
- All URLs must be wrapped in <a href="..."> tags so they are clickable.
- Preserve all citations and references from the original report.
- Add a clean, readable CSS style (e.g., max-width, line-height, font-family, link colours).
- Do NOT truncate or omit any content from the report.

Research report:
{report}
"""

    # Call the LLM by interacting with the CLIENT. 
    # Remember to set the correct values for the model, messages (system and user prompts) and temperature
    response = CLIENT.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature
    )

    ### END CODE HERE ###

    # Extract the HTML from the assistant message
    html = response.choices[0].message.content.strip()  

    return html
