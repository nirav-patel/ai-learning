# GRADED FUNCTION: generate_research_report_with_tools
def generate_research_report_with_tools(prompt: str, model: str = "gpt-4o") -> str:
    """
    Generates a research report using OpenAI's tool-calling with arXiv and Tavily tools.

    Args:
        prompt (str): The user prompt.
        model (str): OpenAI model name.

    Returns:
        str: Final assistant research report text.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a research assistant that can search the web and arXiv to write detailed, "
                "accurate, and properly sourced research reports.\n\n"
                "🔍 Use tools when appropriate (e.g., to find scientific papers or web content).\n"
                "📚 Cite sources whenever relevant. Do NOT omit citations for brevity.\n"
                "🌐 When possible, include full URLs (arXiv links, web sources, etc.).\n"
                "✍️ Use an academic tone, organize output into clearly labeled sections, and include "
                "inline citations or footnotes as needed.\n"
                "🚫 Do not include placeholder text such as '(citation needed)' or '(citations omitted)'."
            )
        },
        {"role": "user", "content": prompt}
    ]

    # List of available tools
    tools = [research_tools.arxiv_tool_def, research_tools.tavily_tool_def]

    # Maximum number of turns
    max_turns = 10
    
    # Iterate for max_turns iterations
    for _ in range(max_turns):

        ### START CODE HERE ###

        # Chat with the LLM via the client and set the correct arguments. Hint: Their names match names of variables already defined.
        # Make sure to let the LLM choose tools automatically. Hint: Look at the docs provided earlier!
        response = CLIENT.chat.completions.create( 
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=1, 
        ) 

        ### END CODE HERE ###

        # Get the response from the LLM and append to messages
        msg = response.choices[0].message 
        messages.append(msg) 

        # Stop when the assistant returns a final answer (no tool calls)
        if not msg.tool_calls:      
            final_text = msg.content
            print("✅ Final answer:")
            print(final_text)
            break

        # Execute tool calls and append results
        for call in msg.tool_calls:
            tool_name = call.function.name
            args = json.loads(call.function.arguments)
            print(f"🛠️ {tool_name}({args})")

            try:
                tool_func = TOOL_MAPPING[tool_name]
                result = tool_func(**args)
            except Exception as e:
                result = {"error": str(e)}

            ### START CODE HERE ###

            # Keep track of tool use in a new message
            new_msg = { 
                # Set role to "tool" (plain string) to signal a tool was used
                "role": "tool",
                # As stated in the markdown when inspecting the ChatCompletionMessage object 
                # every call has an attribute called id
                "tool_call_id": call.id,
                # The name of the tool was already defined above, use that variable
                "name": tool_name,
                # Pass the result of calling the tool to json.dumps
                "content": json.dumps(result)
            }

            ### END CODE HERE ###

            # Append to messages
            messages.append(new_msg)

    return final_text
