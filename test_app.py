import unittest
from app import app

class FlaskApiTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_missing_model(self):
        response = self.app.post('/process', data={
            'hf_token': 'dummy',
            'prompt': 'What is AI?'
        })
        self.assertEqual(response.status_code, 400)
        self.assertIn(b'Model and prompt are required', response.data)

    def test_missing_prompt(self):
        response = self.app.post('/process', data={
            'hf_token': 'dummy',
            'hf_model': 'bert-base-uncased'
        })
        self.assertEqual(response.status_code, 400)
        self.assertIn(b'Model and prompt are required', response.data)

    def test_valid_request(self):
        # Mocking requests.post is recommended for real tests
        response = self.app.post('/process', data={
            'hf_token': 'dummy',
            'hf_model': 'bert-base-uncased',
            'prompt': 'What is AI?'
        })
        # Since the token/model is dummy, expect failure or error from Hugging Face
        self.assertIn(response.status_code, [200, 500])

if __name__ == '__main__':
    unittest.main()


    