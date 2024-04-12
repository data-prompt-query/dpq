import pytest
from unittest.mock import patch, MagicMock, mock_open
import dpq
import requests


@pytest.fixture
def agent_instance():
    """
    A pytest fixture that creates an instance of the Agent class.
    This fixture provides a reusable Agent instance with pre-configured settings
    for use in multiple test functions.
    """
    return dpq.Agent(url="http://example.com/api", api_key="dummy_key", model="model1")


def test_load_function_payloads(agent_instance):
    """
    Tests if the Agent class correctly loads function payloads from JSON files.
    It checks if the necessary attributes are set on the Agent instance after loading.
    """
    # Mock 'open' to return a predefined JSON content and 'os.listdir' to simulate
    # existing JSON files
    with patch("builtins.open", mock_open(read_data='{"key": "value"}')), patch(
        "os.listdir", return_value=["message1.json", "message2.json"]
    ):
        agent_instance._load_function_payloads()  # Load the payloads
        assert hasattr(
            agent_instance, "message1"
        )  # Check if the payload has been set as an attribute
        assert hasattr(agent_instance, "message2")  # Check for another payload


def test_process_row_success(agent_instance):
    """
    Tests the successful processing of a row by the Agent's _process_row method.
    Checks if the correct response is returned when the requests.post call succeeds.
    """
    # Mock 'requests.post' to return a successful response
    with patch("requests.post") as mocked_post:
        mocked_response = MagicMock()
        mocked_response.raise_for_status.return_value = (
            None  # No exception for HTTP errors
        )
        mocked_response.json.return_value = {
            "choices": [
                {"message": {"content": "response message"}}
            ]  # Define expected JSON response
        }
        mocked_post.return_value = mocked_response

        # Call the method and verify that the returned result is as expected
        result = agent_instance._process_row(
            "dummy_item", [{"role": "system", "content": "test"}]
        )
        assert result == "response message"  # Validate the response


def test_process_row_failure(agent_instance):
    """
    Tests the handling of HTTP errors by the Agent's _process_row method.
    It checks if the method returns None and logs an error when an HTTPError occurs.
    """
    # Mock 'requests.post' to simulate an HTTP error
    with patch("requests.post") as mocked_post:
        mocked_response = MagicMock()
        mocked_response.text = "Error"  # Mock the text of the error response
        exception = requests.exceptions.HTTPError()  # Create an HTTPError exception
        exception.response = (
            mocked_response  # Attach the mocked response to the exception
        )
        mocked_post.side_effect = (
            exception  # Set the side effect to raise the exception
        )

        # Call the method and check the results
        result = agent_instance._process_row(
            "dummy_item", [{"role": "system", "content": "test"}]
        )
        assert result is None  # Expect None due to error
        assert "Error" in agent_instance.errors  # Error message should be logged
