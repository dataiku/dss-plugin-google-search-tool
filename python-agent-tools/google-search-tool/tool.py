from dataiku.llm.agent_tools import BaseAgentTool
import requests
from googleapiclient.discovery import build
import logging

class GoogleWebSearchTool(BaseAgentTool):
    def set_config(self, config, plugin_config):
        self.config = config

    def get_descriptor(self, tool):
        return {
            "description": "Searches the web. Returns an array of results. For each result, returns url title, and snippet",            
            "inputSchema" : {
                "$id": "https://dataiku.com/agents/tools/search/input",
                "title": "Input for the search tool",
                "type": "object",
                "properties" : {
                    "q" : {
                        "type": "string",
                        "description": "The query string"
                    }
                },
                "required": ["q"]            
            }
        }

    def invoke(self, input, trace):
        args = input["input"]
        q = args["q"]
        api_key = self.config["google_search_api_connection"]["apiKey"]

        # This logger outputs the key in DEBUG mode ...
        logging.getLogger("googleapiclient.discovery").setLevel("INFO")

        service = build("customsearch", "v1", developerKey=api_key)
        res = service.cse().list(q=q, cx=self.config["cx"]).execute()
        
        source_items = []
        results = []
        for item in res["items"]:
            source_item = {
                "type": "SIMPLE_DOCUMENT",
                "url": item["link"],
                "title": item["title"],
                "htmlSnippet": item["htmlSnippet"]
            }
            if "pagemap" in item and "cse_thumbnail" in item["pagemap"] and "src" in item["pagemap"]["cse_thumbnail"]:
                source_item["thumbnailImageURL"] = item["pagemap"]["cse_thumbnail"]["src"]
                source_item["thumbnailImageW"] = item["pagemap"]["cse_thumbnail"].get("width")
                source_item["thumbnailImageH"] = item["pagemap"]["cse_thumbnail"].get("height")
            
            results.append({
                "url": item["link"],
                "title": item["title"],
                "snippet": item["snippet"]
            })
            source_items.append(source_item)
            
        return { 
            "output" : results,
            "sources":  [{
                "toolCallDescription": "Performed Web Search for: %s" %q,
                "items" : source_items
            }]
        }