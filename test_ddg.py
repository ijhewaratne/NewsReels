try:
    import duckduckgo_search
    print(f"duckduckgo_search version: {duckduckgo_search.__version__}")
    from duckduckgo_search import DDGS
    print("Successfully imported DDGS from duckduckgo_search")
except ImportError as e:
    print(f"ImportError duckduckgo_search: {e}")

try:
    import ddgs
    print("Successfully imported ddgs")
except ImportError as e:
    print(f"ImportError ddgs: {e}")
