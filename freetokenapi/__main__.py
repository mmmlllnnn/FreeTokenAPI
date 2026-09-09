import sys

import uvicorn

MIN_PYTHON = (3, 10)


def main() -> None:
    if sys.version_info < MIN_PYTHON:
        print(
            f"FreeTokenAPI requires Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+, got {sys.version.split()[0]}",
            file=sys.stderr,
        )
        sys.exit(1)
    from freetokenapi.config import settings
    from freetokenapi.logging import uvicorn_log_config

    uvicorn.run(
        "freetokenapi.api.openai:app",
        host=settings.host,
        port=settings.port,
        log_config=uvicorn_log_config(),
    )


if __name__ == "__main__":
    main()
