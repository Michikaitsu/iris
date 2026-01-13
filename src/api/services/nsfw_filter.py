"""
Enhanced NSFW Filter Service - Content Safety
Features: Regex + ML-based detection, obfuscation resistance, configurable strength
"""
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from src.utils.logger import create_logger

logger = create_logger("NSFWFilter")

# Global filter settings
_filter_strength = 2
_filter_enabled = True


def set_filter_strength(strength: int):
    """Set the global filter strength (1=Relaxed, 2=Standard, 3=Strict)"""
    global _filter_strength
    _filter_strength = max(1, min(3, strength))
    logger.info(f"NSFW filter strength set to: {['Relaxed', 'Standard', 'Strict'][_filter_strength - 1]}")


def get_filter_strength() -> int:
    """Get the current filter strength"""
    return _filter_strength


def set_filter_enabled(enabled: bool):
    """Enable or disable the NSFW filter globally"""
    global _filter_enabled
    _filter_enabled = enabled
    logger.info(f"NSFW filter {'enabled' if enabled else 'disabled'}")


def is_filter_enabled() -> bool:
    """Check if the NSFW filter is enabled"""
    return _filter_enabled


@dataclass
class FilterResult:
    """Result of NSFW check"""
    is_unsafe: bool
    category: str = ""
    reason: str = ""
    confidence: float = 0.0
    detected_terms: List[str] = None
    message: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "is_unsafe": self.is_unsafe,
            "category": self.category,
            "reason": self.reason,
            "confidence": self.confidence,
            "detected_terms": self.detected_terms or [],
            "message": self.message
        }


class NSFWFilterService:
    """
    Intelligent NSFW filter with multiple detection methods:
    1. Fast regex-based detection
    2. Obfuscation normalization
    3. Optional ML-based classification (lazy-loaded)
    """
    
    # Allowed non-sexual body & proportion descriptors
    ALLOWED_BODY_DESCRIPTORS = {
        "slim", "slender", "petite", "tall", "short", "athletic", "curvy",
        "wide hips", "narrow waist", "long legs", "soft curves",
        "stylized proportions", "anime proportions", "manga proportions",
        "fashion model body", "elegant posture", "graceful", "beautiful",
        "cute", "kawaii", "pretty", "gorgeous", "stunning"
    }
    
    # Explicit sexual anatomy (HARD BLOCK)
    EXPLICIT_SEXUAL_TERMS = [
        r"\bnipple(s)?\b",
        r"\bareola(s)?\b",
        r"\bgenitals?\b",
        r"\bvagina(l)?\b",
        r"\bpenis\b",
        r"\bclitoris\b",
        r"\btesticle(s)?\b",
        r"\bbreast(s)?\b(?!plate|armor)",  # Allow breastplate/armor
        r"\bboob(s|ies)?\b",
        r"\btit(s|ties)?\b",
        r"\bpussy\b",
        r"\bcock\b",
        r"\bdick\b",
        r"\bass\b(?!ist|ess|et)",  # Allow assist, assess, asset
        r"\bbutt(ocks)?\b",
        r"\banus\b",
    ]
    
    # Sexual context keywords (HARD BLOCK)
    SEXUAL_CONTEXT = [
        r"\bnude\b",
        r"\bnaked\b",
        r"\bsex(y|ual)?\b",
        r"\berotic(a)?\b",
        r"\bporn(o|ographic)?\b",
        r"\bhentai\b",
        r"\borgasm\b",
        r"\bmasturbat",
        r"\bintercourse\b",
        r"\bfuck(ing)?\b",
        r"\bcum(ming|shot)?\b",
        r"\bstrip(ping|tease)?\b",
        r"\blingerie\b",
        r"\bundress(ed|ing)?\b",
        r"\btopless\b",
        r"\bbottomless\b",
        r"\bexposed\b",
        r"\bexplicit\b",
        r"\bnsfw\b",
        r"\blewd\b",
        r"\bsuggestive\b",
        r"\bseductive\b",
        r"\bsensual\b",
        r"\bintimate\b",
        r"\bprovocative\b",
    ]
    
    # Violence/Gore (BLOCK)
    VIOLENCE_TERMS = [
        r"\bgore\b",
        r"\bgory\b",
        r"\bmutilat",
        r"\bdismember",
        r"\bbeheading\b",
        r"\btorture\b",
        r"\bbrutal\b",
        r"\bviolent\b",
    ]
    
    # Character substitution map for obfuscation detection
    CHAR_SUBSTITUTIONS = {
        "0": "o", "1": "i", "3": "e", "4": "a", "5": "s",
        "7": "t", "8": "b", "9": "g", "@": "a", "$": "s",
        "*": "", "_": "", "-": "", ".": "", "!": "i",
        "|": "l", "(": "c", ")": "", "[": "", "]": "",
        "{": "", "}": "", "<": "", ">": "", "~": "",
    }
    
    def __init__(self):
        self._ml_classifier = None
        self._ml_available = None
        
        # Compile regex patterns for performance
        self._explicit_patterns = [re.compile(p, re.IGNORECASE) for p in self.EXPLICIT_SEXUAL_TERMS]
        self._sexual_patterns = [re.compile(p, re.IGNORECASE) for p in self.SEXUAL_CONTEXT]
        self._violence_patterns = [re.compile(p, re.IGNORECASE) for p in self.VIOLENCE_TERMS]
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text to detect obfuscation attempts
        e.g., n1ppl3 -> nipple, n*pple -> npple
        """
        text = text.lower()
        
        # Apply character substitutions
        for char, replacement in self.CHAR_SUBSTITUTIONS.items():
            text = text.replace(char, replacement)
        
        # Remove repeated characters (e.g., "niiipple" -> "nipple")
        text = re.sub(r'(.)\1{2,}', r'\1', text)
        
        # Remove spaces between single characters (e.g., "n i p p l e" -> "nipple")
        text = re.sub(r'\b(\w)\s+(?=\w\b)', r'\1', text)
        
        return text
    
    def _check_patterns(
        self, 
        text: str, 
        patterns: List[re.Pattern]
    ) -> Tuple[bool, List[str]]:
        """Check text against a list of compiled patterns"""
        detected = []
        for pattern in patterns:
            matches = pattern.findall(text)
            if matches:
                detected.extend(matches)
        return bool(detected), detected
    
    def _load_ml_classifier(self) -> bool:
        """Lazy-load ML classifier for advanced detection"""
        if self._ml_available is not None:
            return self._ml_available
        
        try:
            from transformers import pipeline
            self._ml_classifier = pipeline(
                "text-classification",
                model="michellejieli/NSFW_text_classifier",
                device=-1  # CPU to save GPU memory
            )
            self._ml_available = True
            logger.info("ML NSFW classifier loaded")
            return True
        except Exception as e:
            logger.warning(f"ML classifier not available: {e}")
            self._ml_available = False
            return False
    
    def _ml_check(self, text: str) -> Tuple[bool, float]:
        """Run ML-based NSFW classification"""
        if not self._load_ml_classifier():
            return False, 0.0
        
        try:
            result = self._ml_classifier(text[:512])[0]  # Limit input length
            is_nsfw = result["label"] == "NSFW" and result["score"] > 0.85
            return is_nsfw, result["score"]
        except Exception as e:
            logger.warning(f"ML classification failed: {e}")
            return False, 0.0
    
    def check_prompt(
        self, 
        prompt: str, 
        enabled: bool = True,
        use_ml: bool = False,
        strength: int = None
    ) -> FilterResult:
        """
        Check prompt for NSFW content
        
        Args:
            prompt: The text prompt to check
            enabled: Whether filtering is enabled
            use_ml: Whether to use ML-based detection (slower but more accurate)
            strength: Filter strength (1=Relaxed, 2=Standard, 3=Strict). Uses global setting if None.
        
        Returns:
            FilterResult with detection details
        """
        if not enabled:
            return FilterResult(
                is_unsafe=False,
                message="NSFW filter disabled"
            )
        
        if not prompt or not prompt.strip():
            return FilterResult(
                is_unsafe=False,
                message="Empty prompt"
            )
        
        # Use provided strength or global setting
        current_strength = strength if strength is not None else get_filter_strength()
        
        prompt_lower = prompt.lower()
        normalized = self._normalize_text(prompt)
        
        # Check for explicit sexual anatomy (always blocked, even on Relaxed)
        found_explicit, explicit_terms = self._check_patterns(prompt_lower, self._explicit_patterns)
        if not found_explicit:
            found_explicit, explicit_terms = self._check_patterns(normalized, self._explicit_patterns)
        
        if found_explicit:
            logger.warning(f"Blocked explicit content: {explicit_terms[:3]}")
            return FilterResult(
                is_unsafe=True,
                category="explicit_anatomy",
                reason="explicit_content",
                confidence=1.0,
                detected_terms=explicit_terms[:5],
                message=(
                    "❌ Prompt blocked: Explicit anatomical content detected.\n\n"
                    "Please remove explicit body part descriptions.\n"
                    "Artistic body proportions and stylized characters are allowed."
                )
            )
        
        # Check for sexual context (blocked on Standard and Strict)
        if current_strength >= 2:
            found_sexual, sexual_terms = self._check_patterns(prompt_lower, self._sexual_patterns)
            if not found_sexual:
                found_sexual, sexual_terms = self._check_patterns(normalized, self._sexual_patterns)
            
            if found_sexual:
                logger.warning(f"Blocked sexual context: {sexual_terms[:3]}")
                return FilterResult(
                    is_unsafe=True,
                    category="sexual_context",
                    reason="sexual_content",
                    confidence=1.0,
                    detected_terms=sexual_terms[:5],
                    message=(
                        "❌ Prompt blocked: Sexual content detected.\n\n"
                        "Please remove sexual or explicit descriptions.\n"
                        "Artistic and stylized content is allowed."
                    )
                )
        
        # Check for violence/gore (blocked on Standard and Strict)
        if current_strength >= 2:
            found_violence, violence_terms = self._check_patterns(prompt_lower, self._violence_patterns)
            if found_violence:
                logger.warning(f"Blocked violent content: {violence_terms[:3]}")
                return FilterResult(
                    is_unsafe=True,
                    category="violence",
                    reason="violent_content",
                    confidence=1.0,
                    detected_terms=violence_terms[:5],
                    message=(
                        "❌ Prompt blocked: Violent/gore content detected.\n\n"
                        "Please remove violent or graphic descriptions."
                    )
                )
        
        # Optional ML-based check for edge cases (only on Strict mode or when explicitly requested)
        if use_ml or current_strength >= 3:
            is_nsfw_ml, confidence = self._ml_check(prompt)
            if is_nsfw_ml:
                logger.warning(f"ML detected NSFW content (confidence: {confidence:.2f})")
                return FilterResult(
                    is_unsafe=True,
                    category="ml_detected",
                    reason="ml_classification",
                    confidence=confidence,
                    message=(
                        "❌ Prompt blocked: Content flagged by AI safety filter.\n\n"
                        "Please modify your prompt to be more appropriate."
                    )
                )
        
        return FilterResult(
            is_unsafe=False,
            message="✅ Prompt is safe"
        )


# Backward compatibility function
def check_nsfw_prompt(prompt: str, nsfw_filter_enabled: bool = True) -> Dict:
    """
    Legacy function for backward compatibility
    """
    service = NSFWFilterService()
    result = service.check_prompt(prompt, enabled=nsfw_filter_enabled)
    return result.to_dict()


# Global instance for dependency injection
_nsfw_filter_instance: Optional[NSFWFilterService] = None

def get_nsfw_filter() -> NSFWFilterService:
    """Get or create NSFW filter instance"""
    global _nsfw_filter_instance
    if _nsfw_filter_instance is None:
        _nsfw_filter_instance = NSFWFilterService()
    return _nsfw_filter_instance
