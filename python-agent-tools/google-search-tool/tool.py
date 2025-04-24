import base64
import datetime
import logging
import time

import requests
from dataiku.llm.agent_tools import BaseAgentTool


class GongSearchTool(BaseAgentTool):
    def set_config(self, config, plugin_config):
        self.config = config
        # Initialize logger
        self.logger = logging.getLogger(__name__)

    def get_descriptor(self, tool):
        return {
            "description": "Searches Gong for call transcripts and recordings. Returns an array of results with call metadata and transcript content.",
            "inputSchema": {
                "$id": "https://dataiku.com/agents/tools/gong/input",
                "title": "Input for the Gong search tool",
                "type": "object",
                "properties": {
                    "q": {
                        "type": "string",
                        "description": "The search query string for keywords in calls",
                    },
                    "from_date": {
                        "type": "string",
                        "description": "Start date for search (format: YYYY-MM-DD)",
                    },
                    "to_date": {
                        "type": "string",
                        "description": "End date for search (format: YYYY-MM-DD)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 10)",
                    },
                    "workspace_id": {
                        "type": "string",
                        "description": "Optional Gong workspace ID to filter results",
                    },
                },
                "required": ["q"],
            },
        }

    def _get_auth_header(self):
        """
        Create authorization header based on the configured credentials
        Supports both Basic auth and Bearer token auth
        """
        # Check if the config has API key and secret (for Basic auth)
        if "gong_access_key" in self.config and "gong_access_key_secret" in self.config:
            access_key = self.config["gong_access_key"]
            access_key_secret = self.config["gong_access_key_secret"]

            # Create Basic auth token
            credentials = f"{access_key}:{access_key_secret}"
            encoded_credentials = base64.b64encode(credentials.encode()).decode()
            return {"Authorization": f"Basic {encoded_credentials}"}

        # Check if using Bearer token auth
        elif "gong_bearer_token" in self.config:
            return {"Authorization": f"Bearer {self.config['gong_bearer_token']}"}

        # If neither method is configured, raise error
        else:
            raise ValueError(
                "Gong API credentials not properly configured. Please set either 'gong_access_key' and 'gong_access_key_secret' for Basic auth, or 'gong_bearer_token' for OAuth."
            )

    def _handle_rate_limit(self, response):
        """
        Handle rate limiting by respecting Retry-After header
        """
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 1))
            self.logger.warning(
                f"Rate limit exceeded. Waiting for {retry_after} seconds before retrying."
            )
            time.sleep(retry_after)
            return True
        return False

    def _format_datetime(self, date_str):
        """
        Convert YYYY-MM-DD to ISO-8601 format expected by Gong API
        """
        if not date_str:
            return None

        # If already in ISO format, return as is
        if "T" in date_str and (
            "Z" in date_str or "+" in date_str or "-" in date_str.split("T")[1]
        ):
            return date_str

        # Otherwise, convert from YYYY-MM-DD to ISO format
        try:
            date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            # Return in ISO format with UTC timezone
            return date_obj.strftime("%Y-%m-%dT00:00:00Z")
        except ValueError:
            self.logger.error(f"Invalid date format: {date_str}. Expected YYYY-MM-DD.")
            return None

    def invoke(self, input, trace):
        args = input["input"]
        query = args.get("q", "")
        from_date = self._format_datetime(args.get("from_date"))
        to_date = self._format_datetime(args.get("to_date"))
        limit = args.get("limit", 10)
        workspace_id = args.get("workspace_id")

        # Get the Gong API base URL from config
        base_url = self.config.get("gong_base_url", "https://us-13359.api.gong.io")

        # Create headers with authentication
        try:
            headers = self._get_auth_header()
            headers["Content-Type"] = "application/json"
        except ValueError as e:
            return {"output": [], "error": str(e)}

        # Log the search request (without sensitive data)
        self.logger.info(f"Searching Gong for: {query}")

        try:
            # Step 1: Search for calls using the /v2/calls/search endpoint
            search_url = f"{base_url}/v2/calls/search"

            # Build the search payload
            payload = {"filter": {"keywords": query}, "limit": limit}

            # Add date filters if provided
            if from_date or to_date:
                date_filter = {}
                if from_date:
                    date_filter["from"] = from_date
                if to_date:
                    date_filter["to"] = to_date
                payload["filter"]["dateRange"] = date_filter

            # Add workspace filter if provided
            if workspace_id:
                payload["filter"]["workspaceId"] = workspace_id

            # Initial search request
            search_response = requests.post(search_url, json=payload, headers=headers)

            # Handle rate limiting
            retry_count = 0
            max_retries = 3

            while (
                self._handle_rate_limit(search_response) and retry_count < max_retries
            ):
                search_response = requests.post(
                    search_url, json=payload, headers=headers
                )
                retry_count += 1

            search_response.raise_for_status()
            search_data = search_response.json()

            # Get all matching calls
            all_calls = search_data.get("calls", [])

            # Handle pagination with cursor if needed
            cursor = search_data.get("records", {}).get("cursor")
            while cursor and len(all_calls) < limit:
                pagination_payload = payload.copy()
                pagination_payload["cursor"] = cursor

                # Make paginated request
                pagination_response = requests.post(
                    search_url, json=pagination_payload, headers=headers
                )

                if self._handle_rate_limit(pagination_response):
                    continue

                pagination_response.raise_for_status()
                pagination_data = pagination_response.json()

                # Add calls from this page
                all_calls.extend(pagination_data.get("calls", []))

                # Update cursor for next page
                cursor = pagination_data.get("records", {}).get("cursor")

            # Limit to requested number of results
            all_calls = all_calls[:limit]

            # Step 2: For each call, fetch its transcript
            results = []
            source_items = []

            for call in all_calls:
                call_id = call.get("id")

                # Fetch transcript for this call using /v2/calls/{call_id}/transcript
                transcript_url = f"{base_url}/v2/calls/{call_id}/transcript"
                transcript_response = requests.get(transcript_url, headers=headers)

                # Handle rate limiting for transcript request
                retry_count = 0
                while (
                    self._handle_rate_limit(transcript_response)
                    and retry_count < max_retries
                ):
                    transcript_response = requests.get(transcript_url, headers=headers)
                    retry_count += 1

                transcript = "No transcript available"
                transcript_segments = []

                if transcript_response.status_code == 200:
                    transcript_data = transcript_response.json()
                    # Extract transcript segments
                    transcript_segments = transcript_data.get("transcript", [])

                    # Compile full transcript text
                    transcript_texts = []
                    for segment in transcript_segments:
                        speaker = segment.get("speakerName", "Unknown")
                        text = segment.get("text", "")
                        if text:
                            transcript_texts.append(f"{speaker}: {text}")

                    transcript = "\n".join(transcript_texts)

                # Create result object with call metadata and transcript snippet
                result = {
                    "callId": call_id,
                    "title": call.get("title", "Untitled Call"),
                    "date": call.get("startTime", ""),
                    "duration": call.get("duration", 0),
                    "parties": [
                        party.get("name", "Unknown")
                        for party in call.get("parties", [])
                    ],
                    "snippet": (
                        transcript[:200] + "..."
                        if len(transcript) > 200
                        else transcript
                    ),
                    "url": f"{base_url.replace('api.', '')}/call/{call_id}",  # Link to the call in Gong UI
                }

                # Add call content context if available
                if "content" in call:
                    result["context"] = (
                        call["content"].get("contextWorkspace", {}).get("name", "")
                    )

                results.append(result)

                # Create source item for Dataiku Agent UI
                source_item = {
                    "type": "SIMPLE_DOCUMENT",
                    "title": call.get("title", "Untitled Call"),
                    "url": f"{base_url.replace('api.', '')}/call/{call_id}",  # Link to the call in Gong UI
                    "htmlSnippet": f"<b>Date:</b> {call.get('startTime', '')}<br><b>Duration:</b> {call.get('duration', 0)} seconds<br><b>Participants:</b> {', '.join([party.get('name', 'Unknown') for party in call.get('parties', [])])}",
                }
                source_items.append(source_item)

            # Return the final results
            return {
                "output": results,
                "sources": [
                    {
                        "toolCallDescription": f"Searched Gong for: {query}",
                        "items": source_items,
                    }
                ],
            }

        except requests.exceptions.HTTPError as http_err:
            error_message = f"HTTP error occurred: {http_err}"
            self.logger.error(error_message)
            return {"output": [], "error": error_message}
        except Exception as err:
            error_message = f"An error occurred: {err}"
            self.logger.error(error_message)
            return {"output": [], "error": error_message}
