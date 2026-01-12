"""
Templates Routes - Prompt template management
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List

from src.api.services.templates import prompt_templates
from src.utils.logger import create_logger

logger = create_logger("TemplatesRoutes")
router = APIRouter(prefix="/api/templates", tags=["templates"])


class CreateTemplateRequest(BaseModel):
    id: str
    name: str
    prefix: str = ""
    suffix: str = ""
    negative: str = ""


class UpdateTemplateRequest(BaseModel):
    name: Optional[str] = None
    prefix: Optional[str] = None
    suffix: Optional[str] = None
    negative: Optional[str] = None


class ApplyTemplateRequest(BaseModel):
    prompt: str
    category: str
    template_id: str
    include_negative: bool = True


class CombineTemplatesRequest(BaseModel):
    prompt: str
    selections: List[dict]  # [{"category": "styles", "id": "anime"}, ...]


@router.get("")
async def get_all_templates():
    """Get all available templates"""
    return {
        "templates": prompt_templates.get_all(),
        "categories": prompt_templates.get_categories()
    }


@router.get("/categories")
async def get_categories():
    """Get all template categories"""
    return {"categories": prompt_templates.get_categories()}


@router.get("/category/{category}")
async def get_by_category(category: str):
    """Get all templates in a category"""
    templates = prompt_templates.get_by_category(category)
    return {
        "category": category,
        "templates": templates
    }


@router.get("/{category}/{template_id}")
async def get_template(category: str, template_id: str):
    """Get a specific template"""
    template = prompt_templates.get_template(category, template_id)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    return {
        "category": category,
        "id": template_id,
        "template": template
    }


@router.post("/custom")
async def create_custom_template(request: CreateTemplateRequest):
    """Create a new custom template"""
    success = prompt_templates.add_custom(
        template_id=request.id,
        name=request.name,
        prefix=request.prefix,
        suffix=request.suffix,
        negative=request.negative
    )
    
    if not success:
        raise HTTPException(status_code=400, detail="Failed to create template")
    
    return {
        "success": True,
        "template_id": request.id,
        "message": f"Created custom template: {request.name}"
    }


@router.put("/custom/{template_id}")
async def update_custom_template(template_id: str, request: UpdateTemplateRequest):
    """Update a custom template"""
    success = prompt_templates.update_custom(
        template_id=template_id,
        name=request.name,
        prefix=request.prefix,
        suffix=request.suffix,
        negative=request.negative
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="Template not found")
    
    return {"success": True, "template_id": template_id}


@router.delete("/custom/{template_id}")
async def delete_custom_template(template_id: str):
    """Delete a custom template"""
    success = prompt_templates.delete_custom(template_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Template not found")
    
    return {"success": True, "template_id": template_id}


@router.post("/apply")
async def apply_template(request: ApplyTemplateRequest):
    """Apply a template to a prompt"""
    result = prompt_templates.apply_template(
        prompt=request.prompt,
        category=request.category,
        template_id=request.template_id,
        include_negative=request.include_negative
    )
    
    return {
        "original_prompt": request.prompt,
        "enhanced_prompt": result["prompt"],
        "negative_prompt": result["negative_prompt"]
    }


@router.post("/combine")
async def combine_templates(request: CombineTemplatesRequest):
    """Combine multiple templates"""
    result = prompt_templates.combine_templates(
        prompt=request.prompt,
        template_selections=request.selections
    )
    
    return {
        "original_prompt": request.prompt,
        "enhanced_prompt": result["prompt"],
        "negative_prompt": result["negative_prompt"],
        "templates_applied": len(request.selections)
    }


# Quick access endpoints for common templates
@router.get("/quick/styles")
async def get_style_templates():
    """Get all style templates"""
    return prompt_templates.get_by_category("styles")


@router.get("/quick/quality")
async def get_quality_templates():
    """Get all quality templates"""
    return prompt_templates.get_by_category("quality")


@router.get("/quick/subjects")
async def get_subject_templates():
    """Get all subject templates"""
    return prompt_templates.get_by_category("subjects")


@router.get("/quick/lighting")
async def get_lighting_templates():
    """Get all lighting templates"""
    return prompt_templates.get_by_category("lighting")
