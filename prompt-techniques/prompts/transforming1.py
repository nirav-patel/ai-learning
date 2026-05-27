## Transforming

# In this notebook, we will explore how to use Large Language Models for text transformation tasks such as language translation, spelling and grammar checking, tone adjustment, and format conversion.
# ChatGPT is trained with sources in many languages. This gives the model the ability to do translation. Here are some examples of how to use this capability.

prompt1 = """
Translate the following English text to Spanish: \ 
```Hi, I would like to order a blender```
"""

prompt2 = """
Tell me which language this is: 
```Combien coûte le lampadaire?```
"""


prompt3 = """
Translate the following  text to French and Spanish
and English pirate: \
```I want to order a basketball```
"""

prompt = """
Translate the following text to Spanish in both the \
formal and informal forms: 
'Would you like to order a pillow?'
"""
