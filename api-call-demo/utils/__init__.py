"""
utils package for api-call-demo
================================
Re-exports all commonly used symbols so demo files can use short imports:

    from utils import create_client, get_completion_sys_role, moderate_content, ...

Modules:
  bedrock_client      – AWS Bedrock client + LLM call helpers (MODEL_ID, create_client, etc.)
  product_catalog     – Product/category data loading, lookup, and formatting helpers
  content_moderation  – LLM-based content moderation (flag harmful inputs/outputs)
  product_extractor   – Extract product/category mentions from customer queries
"""
from .bedrock_client import (
    MODEL_ID,
    create_client,
    get_completion,
    get_completion_sys_role,
    get_completion_and_token_count,
)
from .product_catalog import (
    products,
    categories,
    get_product_by_name,
    get_products_by_category,
    build_allowed_products_text,
    build_category_names,
    read_string_to_list,
    generate_output_string,
)
from .content_moderation import moderate_content
from .product_extractor import find_category_and_product
