#!/usr/bin/env python3
"""
CLI entry point for HomeLLM benchmark runner
"""

import asyncio
from ..benchmark.runner import main

if __name__ == "__main__":
    asyncio.run(main())