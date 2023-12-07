from typing import Iterable, Optional

from requests.exceptions import HTTPError, ReadTimeout
from urllib3.exceptions import ConnectionError

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.document_loaders.web_base import WebBaseLoader
from langchain.schema import Document
from langchain.tools.base import BaseTool


class YahooFinanceNewsTool(BaseTool):
    """Tool that searches financial news on Yahoo Finance."""

    name: str = "yahoo_finance_news"
    description: str = (
        "Useful for when you need to find financial news "
        "about a public company. "
        "Input should be a company ticker. "
        "For example, AAPL for Apple, MSFT for Microsoft."
    )
    top_k: int = 10
    """The number of results to return."""

    def _run(
        self,
        tool_input: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Yahoo Finance News tool."""
        try:
            import yfinance
        except ImportError:
            raise ImportError(
                "Could not import yfinance python package. "
                "Please install it with `pip install yfinance`."
            )
        company = yfinance.Ticker(tool_input)
        try:
            if company.isin is None:
                return f"Company ticker {tool_input} not found."
        except (HTTPError, ReadTimeout, ConnectionError):
            return f"Company ticker {tool_input} not found."

        links = []
        try:
            links = [n["link"] for n in company.news if n["type"] == "STORY"]
        except (HTTPError, ReadTimeout, ConnectionError):
            if not links:
                return f"No news found for company that searched with {tool_input} ticker."
        if not links:
            return f"No news found for company that searched with {tool_input} ticker."
        loader = WebBaseLoader(web_paths=links)
        docs = loader.load()
        result = self._format_results(docs, tool_input)
        if not result:
            return f"No news found for company that searched with {tool_input} ticker."
        return result

    @staticmethod
    def _format_results(docs: Iterable[Document], tool_input: str) -> str:
        doc_strings = [
            "\n".join([doc.metadata["title"], doc.metadata["description"]])
            for doc in docs
            if tool_input in doc.metadata["description"] or tool_input in doc.metadata["title"]
        ]
        return "\n\n".join(doc_strings)
