import os
import logging
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import AzureError, ClientAuthenticationError, ResourceNotFoundError
import re
from datetime import datetime
import streamlit as st
from typing import Optional, Dict, Any, Union
import io
from db import get_ocr_feedback, upsert_ocr_feedback  # Import your database functions


# Configure logging for Azure operations
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AzureDocumentIntelligenceOCR:
    """
    Azure Document Intelligence OCR service following Azure best practices
    """

    def __init__(self):

        # Load from Streamlit secrets first (Cloud safe)
        self.endpoint = st.secrets.get("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT") or os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
        self.key = st.secrets.get("AZURE_DOCUMENT_INTELLIGENCE_KEY") or os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

        self.client = None

        # ❌ REMOVED st.error() HERE (this caused set_page_config crash)
        if not self.endpoint or not self.key:
            logger.error("Azure credentials not found")
            return

        try:
            self.client = DocumentIntelligenceClient(
                endpoint=self.endpoint,
                credential=AzureKeyCredential(self.key)
            )
            logger.info("Azure client initialized successfully")
        except Exception as e:
            logger.error(f"Azure initialization error: {e}")
            # ❌ REMOVED st.error() HERE


    def extract_expiry_date(self, image_file) -> Optional[Dict[str, Any]]:
        if not self.client:
            st.error("Azure Document Intelligence client not initialized")
            return None

        try:
            if hasattr(image_file, 'seek'):
                image_file.seek(0)

            if hasattr(image_file, 'read'):
                image_bytes = image_file.read()
            else:
                image_bytes = image_file

            if len(image_bytes) > 50 * 1024 * 1024:
                st.error("File size too large. Please use an image smaller than 50MB.")
                return None

            logger.info("Starting document analysis with Azure Document Intelligence")

            poller = self.client.begin_analyze_document(
                model_id="prebuilt-read",
                body=image_bytes,
                content_type="application/octet-stream"
            )

            result = poller.result()
            logger.info("Document analysis completed successfully")

            extracted_text = self._extract_text_from_result(result)

            if not extracted_text.strip():
                st.warning("No text was extracted from the image. Please try with a clearer image.")
                return None

            parsed_info = self._parse_product_information(extracted_text)

            return parsed_info

        except ResourceNotFoundError as e:
            logger.error(f"Azure resource not found: {e}")
            st.error("Azure Document Intelligence resource not found. Please check your endpoint.")
            return None
        except ClientAuthenticationError as e:
            logger.error(f"Authentication error: {e}")
            st.error("Authentication failed. Please check your Azure credentials.")
            return None
        except AzureError as e:
            logger.error(f"Azure service error: {e}")
            st.error(f"Azure service error: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during OCR processing: {e}")
            st.error(f"Error during OCR processing: {str(e)}")
            return None


    def _extract_text_from_result(self, result) -> str:
        extracted_text = ""

        if result.pages:
            for page in result.pages:
                if page.lines:
                    for line in page.lines:
                        extracted_text += line.content + "\n"

        return extracted_text


    def _parse_product_information(self, text: str) -> Dict[str, Any]:
        result = {
            'expiry_date': None,
            'product_name': None,
            'manufacturer': None,
            'batch_number': None,
            'raw_text': text,
            'confidence': None
        }

        expiry_patterns = [
            r'(?:exp|expiry|expires|best\s+before|use\s+by|bb|best\s+by)\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(?:exp|expiry|expires|best\s+before|use\s+by|bb|best\s+by)\s*:?\s*(\d{1,2}\s+\w{3,9}\s+\d{2,4})',
            r'(?:exp|expiry|expires)\s*:?\s*(\d{1,2}\.\d{1,2}\.\d{2,4})',
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(\d{2,4}[/-]\d{1,2}[/-]\d{1,2})',
            r'(\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{2,4})',
            r'(\d{1,2}\.\d{1,2}\.\d{2,4})'
        ]

        for i, pattern in enumerate(expiry_patterns):
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                date_str = match.group(1)
                parsed_date = self._parse_date_string(date_str)
                if parsed_date:
                    result['expiry_date'] = parsed_date
                    result['confidence'] = 'high' if i < 3 else 'medium'
                    break
            if result['expiry_date']:
                break

        result['product_name'] = self._extract_product_name(text)
        result['manufacturer'] = self._extract_manufacturer(text)
        result['batch_number'] = self._extract_batch_number(text)

        return result


    def _extract_product_name(self, text: str) -> Optional[str]:
        lines = text.split('\n')
        potential_names = []

        for i, line in enumerate(lines[:15]):
            line = line.strip()
            if (len(line) > 3 and
                not re.match(r'^\d+$', line) and
                'barcode' not in line.lower() and
                'exp' not in line.lower()[:10] and
                'mfg' not in line.lower()[:10]):
                potential_names.append((line, len(line)))

        if potential_names:
            return max(potential_names, key=lambda x: x[1])[0]
        return None


    def _extract_manufacturer(self, text: str) -> Optional[str]:
        brand_patterns = [
            r'(?:mfg|manufactured\s+by|brand|company)\s*:?\s*([a-zA-Z\s&]+)',
            r'([A-Z][a-zA-Z\s&]{3,25})\s+(?:ltd|inc|corp|pvt|limited)',
            r'(?:by\s+)([A-Z][a-zA-Z\s&]{3,25})'
        ]

        for pattern in brand_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None


    def _extract_batch_number(self, text: str) -> Optional[str]:
        batch_patterns = [
            r'(?:batch|lot|b\.no|lot\s+no|batch\s+no)\s*:?\s*([a-zA-Z0-9]+)',
            r'batch\s*:?\s*([a-zA-Z0-9]+)',
            r'lot\s*:?\s*([a-zA-Z0-9]+)'
        ]

        for pattern in batch_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None


    def _parse_date_string(self, date_str: str) -> Optional[datetime]:
        date_formats = [
            '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d',
            '%d-%m-%Y', '%m-%d-%Y', '%Y-%m-%d',
            '%d.%m.%Y', '%m.%d.%Y', '%Y.%m.%d',
            '%d/%m/%y', '%m/%d/%y', '%y/%m/%d',
            '%d-%m-%y', '%m-%d-%y', '%y-%m-%d',
            '%d.%m.%y', '%m.%d.%y', '%y.%m.%d',
            '%d %b %Y', '%d %B %Y',
            '%b %d %Y', '%B %d %Y',
            '%d %b %y', '%d %B %y',
            '%b %d %y', '%B %d %y'
        ]

        date_str = date_str.strip()

        for fmt in date_formats:
            try:
                parsed_date = datetime.strptime(date_str, fmt)
                if parsed_date.year < 1950:
                    if parsed_date.year < 30:
                        parsed_date = parsed_date.replace(year=parsed_date.year + 2000)
                    else:
                        parsed_date = parsed_date.replace(year=parsed_date.year + 1900)
                return parsed_date
            except ValueError:
                continue

        return None


ocr_service = AzureDocumentIntelligenceOCR()


def extract_expiry_date(image_file):
    return ocr_service.extract_expiry_date(image_file)