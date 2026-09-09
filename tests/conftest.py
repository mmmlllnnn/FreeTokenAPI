import os

# Keep mocked tests independent of local proxy and login configuration.
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

os.environ["FREETOKENAPI_HUMAN_DELAY_MIN"] = "0"
os.environ["FREETOKENAPI_HUMAN_DELAY_MAX"] = "0"
for _key in (
    "DEEPSEEK_TOKENS",
    "QWEN_TOKENS",
):
    os.environ[_key] = ""
