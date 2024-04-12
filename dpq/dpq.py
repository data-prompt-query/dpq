import json
import requests
import os
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from importlib import resources
import dpq.prompts
from pathlib import Path


class Agent:
    """
    A class for dynamically creating and executing functions based on JSON templates
    to interact with an LLM API, with optional parallel execution.

    Attributes:
        url (str): The base URL for the API.
        api_key (dict): API key to authorize the API request.
        model (str): The model identifier for API requests.
        parallel (bool): Whether to execute requests in parallel.
        errors (list): List to store error messages from failed API calls.
    """

    def __init__(
        self,
        url,
        api_key,
        model,
        parallel=True,
        custom_messages_path=None,
    ):
        self.url = url
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        self.model = model
        self.parallel = parallel
        self.errors = []
        self.custom_messages_path = custom_messages_path
        self._load_function_payloads()

    def _load_function_payloads(self):
        """
        Loads message templates from internal packaged prompts and potentially
        from an external directory specified by custom_messages_path.
        """
        templates = {}

        try:
            with resources.path(dpq.prompts, "") as prompts_path:
                for filepath in prompts_path.iterdir():
                    if filepath.suffix == ".json":
                        function_name = filepath.stem
                        with open(filepath, "r") as file:
                            templates[function_name] = json.load(file)
        except Exception as e:
            print(f"Failed to load internal prompts: {e}")

        # Load from custom message path, if specified
        if self.custom_messages_path:
            custom_path = Path(self.custom_messages_path)
            for filepath in custom_path.iterdir():
                if filepath.suffix == ".json":
                    function_name = filepath.stem
                    with open(filepath, "r") as file:
                        templates[function_name] = json.load(file)

        # Set attributes for all loaded templates
        for function_name, template in templates.items():
            setattr(self, function_name, self.generate_function(template))

    def generate_function(self, messages_template):
        def function(data):
            # Initialize an empty list of the same length as data to hold the results
            results = [None] * len(data)

            if self.parallel:
                with ThreadPoolExecutor() as executor:
                    # Create a list to hold futures, pairing each with its corresponding
                    # index
                    futures = {
                        executor.submit(self._process_row, item, messages_template): i
                        for i, item in enumerate(data)
                    }

                    # Iterate over completed futures
                    for future in tqdm(as_completed(futures), total=len(futures)):
                        # Retrieve the original index for this future
                        index = futures[future]
                        try:
                            # Store the result in the correct position based on the
                            # original index
                            results[index] = future.result()
                        except Exception as e:
                            # Error handling, storing None for failures
                            results[index] = None
            else:
                # Sequential execution
                for i, item in enumerate(tqdm(data)):
                    results[i] = self._process_row(item, messages_template)

            return results

        return function

    def _process_row(self, item, messages_template):
        """
        Sends a single API request with the item attached to the message template as the
        last user input.
        """

        # Attach item to message template using deep copy to ensure template is not
        # changed
        messages = copy.deepcopy(messages_template)
        messages.append({"role": "user", "content": str(item)})

        # Prepare the request payload
        payload = {}
        payload["messages"] = messages
        payload["model"] = self.model

        try:
            response = requests.post(self.url, json=payload, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

        except requests.exceptions.HTTPError as e:
            # Log or store the error message from the response
            self.errors.append(str(e.response.text))
            return None

        except Exception as e:
            # Handle other exceptions, e.g., network errors
            self.errors.append(str(e))
            return None
