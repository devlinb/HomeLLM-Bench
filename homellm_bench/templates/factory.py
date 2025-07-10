"""
Template factory for creating chat templates dynamically
"""
from typing import Optional
from .base import ChatTemplate
from .phi3 import Phi3ChatTemplate
from .qwen import QwenChatTemplate


class TemplateFactory:
    """Factory for creating chat templates based on template name"""
    
    _templates = {
        "phi3": Phi3ChatTemplate,
        "qwen": QwenChatTemplate,
    }
    
    @classmethod
    def create_template(cls, template_name: str) -> Optional[ChatTemplate]:
        """Create a template instance by name"""
        template_class = cls._templates.get(template_name.lower())
        if template_class:
            return template_class()
        return None
    
    @classmethod
    def get_available_templates(cls) -> list[str]:
        """Get list of available template names"""
        return list(cls._templates.keys())
    
    @classmethod
    def register_template(cls, name: str, template_class: type[ChatTemplate]):
        """Register a new template class"""
        cls._templates[name.lower()] = template_class


# Convenience function
def get_template(template_name: str) -> Optional[ChatTemplate]:
    """Get a template instance by name"""
    return TemplateFactory.create_template(template_name)