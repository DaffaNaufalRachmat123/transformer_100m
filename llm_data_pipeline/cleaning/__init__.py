"""Content cleaning per DATA_PIPELINE_PLAN A.5.

Sub-modules:
  web_artifact_removal — strip HTML/CSS/JS/navigation/footer/ads
  text_repair         — fix encoding, normalize emoji/URL/email/phone
  boilerplate_detector — detect and filter boilerplate-heavy documents
"""

from .web_artifact_removal import WebArtifactRemover
from .text_repair import TextRepairer
from .boilerplate_detector import BoilerplateDetector

__all__ = ["WebArtifactRemover", "TextRepairer", "BoilerplateDetector"]
