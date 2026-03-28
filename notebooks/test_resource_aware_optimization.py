import unittest
from unittest.mock import MagicMock, patch
import json
import sys

# Define mocks
mock_requests = MagicMock()
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

        # Call the function and assert it raises json.JSONDecodeError
        prompt = "test prompt"
        with self.assertRaises(json.JSONDecodeError):
            self.classify_prompt(prompt)

if __name__ == '__main__':
    unittest.main()
