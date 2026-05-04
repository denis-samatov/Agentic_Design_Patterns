import unittest
from unittest.mock import MagicMock, patch
import json
import sys

# Define mocks
mock_requests = MagicMock()

# Define a mock exception class for requests.exceptions.RequestException
class MockRequestException(Exception):
    pass

mock_requests.exceptions.RequestException = MockRequestException
mock_dotenv = MagicMock()
mock_openai = MagicMock()

class TestClassifyPrompt(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Mock external dependencies before importing the module under test
        cls.original_modules = {
            'requests': sys.modules.get('requests'),
            'dotenv': sys.modules.get('dotenv'),
            'openai': sys.modules.get('openai')
        }
        sys.modules['requests'] = mock_requests
        sys.modules['dotenv'] = mock_dotenv
        sys.modules['openai'] = mock_openai

        # Now it's safe to import the function
        from notebooks.resource_aware_optimization import classify_prompt
        cls.classify_prompt = staticmethod(classify_prompt)

    @classmethod
    def tearDownClass(cls):
        # Restore original modules
        for name, module in cls.original_modules.items():
            if module is None:
                del sys.modules[name]
            else:
                sys.modules[name] = module

    @patch('notebooks.resource_aware_optimization.client.chat.completions.create')
    def test_classify_prompt_simple(self, mock_create):
        # Setup mock response
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"classification": "simple"}'))
        ]
        mock_create.return_value = mock_response

        # Call the function
        prompt = "What is the capital of France?"
        result = self.classify_prompt(prompt)

        # Assertions
        self.assertEqual(result, {"classification": "simple"})
        mock_create.assert_called_once()
        args, kwargs = mock_create.call_args
        self.assertEqual(kwargs['model'], "gpt-4o")
        self.assertEqual(kwargs['messages'][1]['content'], prompt)

    @patch('notebooks.resource_aware_optimization.client.chat.completions.create')
    def test_classify_prompt_reasoning(self, mock_create):
        # Setup mock response
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"classification": "reasoning"}'))
        ]
        mock_create.return_value = mock_response

        # Call the function
        prompt = "Solve for x: 2x + 5 = 15"
        result = self.classify_prompt(prompt)

        # Assertions
        self.assertEqual(result, {"classification": "reasoning"})
        mock_create.assert_called_once()

    @patch('notebooks.resource_aware_optimization.client.chat.completions.create')
    def test_classify_prompt_internet_search(self, mock_create):
        # Setup mock response
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"classification": "internet_search"}'))
        ]
        mock_create.return_value = mock_response

        # Call the function
        prompt = "Who won the Super Bowl in 2024?"
        result = self.classify_prompt(prompt)

        # Assertions
        self.assertEqual(result, {"classification": "internet_search"})
        mock_create.assert_called_once()

    @patch('notebooks.resource_aware_optimization.client.chat.completions.create')
    def test_classify_prompt_invalid_json(self, mock_create):
        # Setup mock response with invalid JSON
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='invalid json'))
        ]
        mock_create.return_value = mock_response

        # Call the function and assert it returns fallback
        prompt = "test prompt"
        result = self.classify_prompt(prompt)
        self.assertEqual(result, {"classification": "simple"})

    @patch('notebooks.resource_aware_optimization.client.chat.completions.create')
    def test_classify_prompt_markdown_json(self, mock_create):
        # Setup mock response with markdown JSON
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='```json\n{"classification": "reasoning"}\n```'))
        ]
        mock_create.return_value = mock_response

        # Call the function
        prompt = "test prompt"
        result = self.classify_prompt(prompt)

        # Assertions
        self.assertEqual(result, {"classification": "reasoning"})

class TestGoogleSearch(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Mock external dependencies before importing the module under test
        cls.original_modules = {
            'requests': sys.modules.get('requests'),
            'dotenv': sys.modules.get('dotenv'),
            'openai': sys.modules.get('openai')
        }
        sys.modules['requests'] = mock_requests
        sys.modules['dotenv'] = mock_dotenv
        sys.modules['openai'] = mock_openai

        # Now it's safe to import the function
        from notebooks.resource_aware_optimization import google_search
        cls.google_search = staticmethod(google_search)

    @classmethod
    def tearDownClass(cls):
        # Restore original modules
        for name, module in cls.original_modules.items():
            if module is None:
                del sys.modules[name]
            else:
                sys.modules[name] = module

    def setUp(self):
        mock_requests.get.reset_mock()
        mock_requests.get.side_effect = None
        mock_requests.get.return_value = MagicMock()

    def test_google_search_success(self):
        # Setup mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "items": [
                {
                    "title": "Test Title",
                    "snippet": "Test Snippet",
                    "link": "https://test.com"
                }
            ]
        }
        mock_requests.get.return_value = mock_response

        # Call the function
        result = self.google_search("test query")

        # Assertions
        expected = [
            {
                "title": "Test Title",
                "snippet": "Test Snippet",
                "link": "https://test.com"
            }
        ]
        self.assertEqual(result, expected)
        mock_requests.get.assert_called_once()

    def test_google_search_no_items(self):
        # Setup mock response with no items
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_requests.get.return_value = mock_response

        # Call the function
        result = self.google_search("test query")

        # Assertions
        self.assertEqual(result, [])
        mock_requests.get.assert_called_once()

    def test_google_search_exception(self):
        # Setup mock to raise RequestException
        mock_requests.get.side_effect = MockRequestException("Connection error")

        # Call the function
        result = self.google_search("test query")

        # Assertions
        self.assertEqual(result, {"error": "Connection error"})
        mock_requests.get.assert_called_once()

class TestHandlePrompt(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Mock external dependencies before importing the module under test
        cls.original_modules = {
            'requests': sys.modules.get('requests'),
            'dotenv': sys.modules.get('dotenv'),
            'openai': sys.modules.get('openai')
        }
        sys.modules['requests'] = mock_requests
        sys.modules['dotenv'] = mock_dotenv
        sys.modules['openai'] = mock_openai

        # Now it's safe to import the function
        from notebooks.resource_aware_optimization import handle_prompt
        cls.handle_prompt = staticmethod(handle_prompt)

    @classmethod
    def tearDownClass(cls):
        # Restore original modules
        for name, module in cls.original_modules.items():
            if module is None:
                del sys.modules[name]
            else:
                sys.modules[name] = module

    @patch('notebooks.resource_aware_optimization.classify_prompt')
    @patch('notebooks.resource_aware_optimization.generate_response')
    def test_handle_prompt_simple(self, mock_generate, mock_classify):
        mock_classify.return_value = {"classification": "simple"}
        mock_generate.return_value = ("Simple answer", "gpt-4o-mini")

        result = self.handle_prompt("Simple question")

        self.assertEqual(result["classification"], "simple")
        self.assertEqual(result["response"], "Simple answer")
        self.assertEqual(result["model"], "gpt-4o-mini")
        mock_generate.assert_called_once_with("Simple question", "simple", None)

    @patch('notebooks.resource_aware_optimization.classify_prompt')
    @patch('notebooks.resource_aware_optimization.generate_response')
    def test_handle_prompt_reasoning(self, mock_generate, mock_classify):
        mock_classify.return_value = {"classification": "reasoning"}
        mock_generate.return_value = ("Reasoning answer", "o1-mini")

        result = self.handle_prompt("Complex question")

        self.assertEqual(result["classification"], "reasoning")
        self.assertEqual(result["response"], "Reasoning answer")
        self.assertEqual(result["model"], "o1-mini")
        mock_generate.assert_called_once_with("Complex question", "reasoning", None)

    @patch('notebooks.resource_aware_optimization.classify_prompt')
    @patch('notebooks.resource_aware_optimization.google_search')
    @patch('notebooks.resource_aware_optimization.generate_response')
    def test_handle_prompt_internet_search_success(self, mock_generate, mock_search, mock_classify):
        mock_classify.return_value = {"classification": "internet_search"}
        search_results = [{"title": "Result", "snippet": "Snippet", "link": "url"}]
        mock_search.return_value = search_results
        mock_generate.return_value = ("Search answer", "gpt-4o")

        result = self.handle_prompt("Current event question")

        self.assertEqual(result["classification"], "internet_search")
        self.assertEqual(result["response"], "Search answer")
        self.assertEqual(result["model"], "gpt-4o")
        mock_search.assert_called_once_with("Current event question")
        mock_generate.assert_called_once_with("Current event question", "internet_search", search_results)

    @patch('notebooks.resource_aware_optimization.classify_prompt')
    @patch('notebooks.resource_aware_optimization.google_search')
    @patch('notebooks.resource_aware_optimization.generate_response')
    def test_handle_prompt_internet_search_error(self, mock_generate, mock_search, mock_classify):
        mock_classify.return_value = {"classification": "internet_search"}
        search_error = {"error": "API Key Invalid"}
        mock_search.return_value = search_error
        mock_generate.return_value = ("Error-based answer", "gpt-4o")

        result = self.handle_prompt("Current event question")

        self.assertEqual(result["classification"], "internet_search")
        self.assertEqual(result["response"], "Error-based answer")
        self.assertEqual(result["model"], "gpt-4o")
        mock_search.assert_called_once_with("Current event question")
        # Here we check if the error was passed to generate_response
        mock_generate.assert_called_once_with("Current event question", "internet_search", search_error)

if __name__ == '__main__':
    unittest.main()
