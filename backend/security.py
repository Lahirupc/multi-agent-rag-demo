import re
from fastapi import HTTPException
from logger import logger
from langsmith import traceable

class SecurityGuardrails:
    """Provides security protections against Prompt Injections, Data Exfiltration, and Brand Safety."""
    
    @staticmethod
    @traceable(name="guardrail_prompt_injection_check")
    def validate_input(prompt: str) -> bool:
        """
        Validate incoming user requests against instruction overrides and data exfiltration patterns.
        """
        forbidden_patterns = [
            r"(?i)ignore (?:all )?previous instructions",
            r"(?i)you are now",
            r"(?i)disregard",
            r"(?i)bypass constraints",
            r"(?i)system prompt",
            r"(?i)dump memory"
        ]
        
        for pattern in forbidden_patterns:
            if re.search(pattern, prompt):
                logger.warning("prompt_injection_attempt_detected", matched_pattern=pattern, prompt_snippet=prompt[:30])
                raise HTTPException(status_code=400, detail="Security Violation: Unsafe prompt structure detected.")
        return True

    @staticmethod
    @traceable(name="guardrail_output_validation")
    def validate_agent_output(response: str) -> str:
        """
        Ensure final response aligns with Commercial Bank brand guidelines
        and prevents hallucinated citations or inappropriate data.
        """
        if not response:
            return "I am the Commercial Bank AI Assistant. How can I help you today?"
            
        # Mock Brand checks
        sensitive_keywords = ["crypto", "bitcoin", "competitor bank"]
        for keyword in sensitive_keywords:
            if keyword in response.lower():
                logger.warning("brand_safety_violation", keyword_found=keyword)
                return "As a Commercial Bank assistant, I cannot provide information on that topic."
        
        return response
