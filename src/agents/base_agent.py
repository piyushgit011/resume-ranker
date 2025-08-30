from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import json
import logging

class BaseAgent(ABC):
    """Base class for all AI agents"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o", temperature: float = 0.3):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_key=api_key
        )
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def process(self, input_data: Any) -> Dict[str, Any]:
        """Process input data and return results"""
        pass
    
    def _invoke_llm(self, prompt: str) -> str:
        """Invoke the LLM with error handling"""
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            self.logger.error(f"LLM invocation failed: {str(e)}")
            raise
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response with fallback handling"""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'({.*})', response.replace('\n', ''), re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except:
                    self.logger.warning("Could not parse JSON response")
                    return {"error": "JSON parsing failed", "raw_response": response}
            else:
                self.logger.warning("Could not extract JSON from response")
                return {"error": "JSON extraction failed", "raw_response": response}
    
    def _create_prompt_template(self, template: str) -> PromptTemplate:
        """Create a prompt template"""
        return PromptTemplate.from_template(template)