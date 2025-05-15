import argparse
import glob
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

import pdfplumber
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class DeepSeekProcessor:
    def __init__(
        self, base_url: str, model_name: str = "qwen1.5:custom", timeout: int = 120
    ):
        """
        Initialize the DeepSeek processor with API configuration.

        Args:
            base_url: Base URL for the DeepSeek API
            model_name: The model to use for FAQ extraction
            timeout: Request timeout in seconds
        """
        self.base_url = base_url
        self.model_name = model_name
        self.timeout = timeout
        self.session = requests.Session()

    def generate_answer(self, text: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        Generate FAQ extraction from document text with retry mechanism.

        Args:
            text: The document text to analyze
            max_retries: Maximum number of retry attempts

        Returns:
            Dictionary containing extracted FAQs
        """
        # Improved prompt with clearer instructions and better JSON formatting
        analysis_prompt = f"""
You are a highly precise document analysis system for healthcare QA applications. Extract every FAQ from the text below with full fidelity.

INSTRUCTIONS:
- Extract **ALL FAQs**, including those meant for healthcare professionals (HCPs) and consumers
- Preserve **exact question phrasing**
- Extract **complete answers**, including:
  - SOP or policy codes (e.g., POL.QA.3656)
  - Manufacturing facility details (e.g., Portage, Indiana)
  - Any corrective or procedural actions (e.g., FDA filings, equipment enhancements)
  - Separate HCP vs. consumer info where applicable
  - Specific document references (e.g., PP-RPT-0020R.01)
  - Technical causes (e.g., buprenorphine HCl particle size)
  - Metadata like document name, reference number (MED-SBF-US-00078), expiry date

FORMAT:
Return a JSON object only:
{{
  "questions": [
    {{
      "question": "Exact question from the text?",
      "answer": "Full answer, including all clinical and regulatory info"
    }},
    ...
  ]
}}

Do NOT paraphrase. Do NOT summarize. Include raw details exactly as written.

Document text:
{text}
"""

        retry_count = 0
        while retry_count < max_retries:
            try:
                logger.info(
                    f"Making request to DeepSeek API (attempt {retry_count + 1}/{max_retries})"
                )
                response = self.session.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": analysis_prompt,
                        "stream": False,
                        "temperature": 0.3,  # Lower temperature for more consistent outputs
                        "max_tokens": 8192,  # Ensure enough tokens for complete responses
                    },
                    timeout=self.timeout,
                )
                response.raise_for_status()
                result = response.json()
                logger.debug(f"API Response: {result}")

                response_text = result.get("response", "")
                return self._parse_json_response(response_text)

            except requests.exceptions.RequestException as e:
                retry_count += 1
                logger.warning(
                    f"Request failed (attempt {retry_count}/{max_retries}): {e}"
                )
                if retry_count >= max_retries:
                    logger.error(
                        f"Max retries reached. Failed to get response from DeepSeek: {e}"
                    )
                    return {"questions": []}
            except Exception as e:
                retry_count += 1
                logger.error(
                    f"Unexpected error (attempt {retry_count}/{max_retries}): {e}"
                )
                if retry_count >= max_retries:
                    return {"questions": []}

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        parsing_strategies = [
            lambda t: (
                json.loads(
                    re.search(r"```(?:json)?\s*(.*?)\s*```", t, re.DOTALL).group(1)
                )
                if re.search(r"```(?:json)?\s*(.*?)\s*```", t, re.DOTALL)
                else None
            ),
            lambda t: json.loads(t),
            lambda t: (
                json.loads(re.search(r"(\{.*\})", t, re.DOTALL).group(1))
                if re.search(r"(\{.*\})", t, re.DOTALL)
                else None
            ),
        ]

        for strategy in parsing_strategies:
            try:
                result = strategy(text)
                if result:
                    return result
            except (json.JSONDecodeError, AttributeError):
                continue

        logger.error("Could not parse JSON from response using any strategy")
        logger.debug(f"Raw response text:\n{text}")  # Add this line for debugging
        return {"questions": []}


class PDFProcessor:
    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        """
        Extract text from a PDF file with improved handling of formatting.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Extracted text from the PDF
        """
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return ""

        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    logger.info(f"Processing page {page_num}/{len(pdf.pages)}")

                    # Extract and clean text
                    page_text = page.extract_text(x_tolerance=3, y_tolerance=3)
                    if page_text:
                        # Remove excessive whitespace but preserve paragraph breaks
                        page_text = re.sub(r"\n\s*\n", "\n\n", page_text)
                        page_text = re.sub(r" +", " ", page_text)
                        text += f"\n\n--- Page {page_num} ---\n\n{page_text}"

            if not text.strip():
                logger.warning(f"No text extracted from {pdf_path}")

            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
            return ""


def extract_and_clean_faqs(
    pdf_paths: List[str], deepseek_processor: DeepSeekProcessor
) -> List[Dict[str, Any]]:
    """
    Extract FAQs from multiple PDFs and deduplicate results.

    Args:
        pdf_paths: List of paths to PDF files
        deepseek_processor: Configured DeepSeek processor instance

    Returns:
        List of unique FAQ pairs
    """
    all_faqs = []
    processed_pdfs = 0

    for pdf_path in pdf_paths:
        try:
            logger.info(f"Processing PDF: {os.path.basename(pdf_path)}")
            text = PDFProcessor.extract_text_from_pdf(pdf_path)

            if not text:
                logger.warning(f"Skipping {pdf_path} due to empty text extraction")
                continue

            # For very large documents, split into chunks to avoid token limits
            if len(text) > 24000:  # Approximate token limit threshold
                logger.info(
                    f"Document is large ({len(text)} chars), processing in chunks"
                )
                chunks = split_text_into_chunks(text, chunk_size=20000, overlap=2000)
                logger.info(f"Split into {len(chunks)} chunks")

                pdf_faqs = []
                for i, chunk in enumerate(chunks):
                    logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                    response = deepseek_processor.generate_answer(chunk)
                    pdf_faqs.extend(response.get("questions", []))
            else:
                logger.info(f"Processing document of size {len(text)} chars")
                response = deepseek_processor.generate_answer(text)
                pdf_faqs = response.get("questions", [])

            # Clean and validate the FAQs
            clean_faqs = []
            for item in pdf_faqs:
                question = item.get("question", "").strip()
                answer = item.get("answer", "").strip()

                # Validate and clean up
                if question and answer:
                    # Ensure question ends with question mark if it's a question
                    if not question.endswith("?") and is_likely_question(question):
                        question = question + "?"

                    clean_faqs.append(
                        {
                            "question": question,
                            "answer": answer,
                            "source": os.path.basename(pdf_path),
                        }
                    )

            logger.info(
                f"Extracted {len(clean_faqs)} FAQs from {os.path.basename(pdf_path)}"
            )
            all_faqs.extend(clean_faqs)
            processed_pdfs += 1

        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")

    # Deduplicate FAQs based on question similarity
    unique_faqs = deduplicate_faqs(all_faqs)
    logger.info(
        f"Processed {processed_pdfs} PDFs, found {len(unique_faqs)} unique FAQs"
    )

    return unique_faqs


def split_text_into_chunks(
    text: str, chunk_size: int = 20000, overlap: int = 2000
) -> List[str]:
    """
    Split large text into overlapping chunks for processing.

    Args:
        text: Text to split
        chunk_size: Maximum size of each chunk
        overlap: Number of characters to overlap between chunks

    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        # Try to find a natural breaking point (paragraph or sentence)
        if end < text_length:
            # Look for paragraph break
            paragraph_break = text.rfind("\n\n", start, end)
            if paragraph_break != -1 and paragraph_break > start + chunk_size // 2:
                end = paragraph_break + 2
            else:
                # Look for sentence break
                sentence_break = text.rfind(". ", start, end)
                if sentence_break != -1 and sentence_break > start + chunk_size // 2:
                    end = sentence_break + 2

        chunks.append(text[start:end])
        start = max(start, end - overlap)  # Create overlap with previous chunk

    return chunks


def is_likely_question(text: str) -> bool:
    """
    Determine if text is likely a question based on content.

    Args:
        text: Text to analyze

    Returns:
        Boolean indicating if text is likely a question
    """
    # Check for question words
    question_starters = [
        "what",
        "how",
        "why",
        "when",
        "where",
        "who",
        "which",
        "can",
        "do",
        "does",
        "is",
        "are",
        "will",
    ]
    lower_text = text.lower()

    # If it starts with a question word, it's likely a question
    for starter in question_starters:
        if lower_text.startswith(starter + " "):
            return True

    # If it contains question-like phrases
    if re.search(r"\b(tell me|explain|describe|provide)\b", lower_text):
        return True

    return False


def deduplicate_faqs(faqs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Remove duplicate FAQs based on question similarity.

    Args:
        faqs: List of FAQ dictionaries

    Returns:
        Deduplicated list of FAQs
    """
    unique_faqs = []
    seen_questions = set()

    for faq in faqs:
        # Normalize question for comparison
        normalized_q = normalize_text(faq["question"])

        # Check if we've seen a similar question
        if normalized_q in seen_questions:
            continue

        seen_questions.add(normalized_q)
        unique_faqs.append(faq)

    return unique_faqs


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison by removing punctuation, extra spaces, and lowercasing.

    Args:
        text: Text to normalize

    Returns:
        Normalized text
    """
    # Remove punctuation, lowercase, and remove extra whitespace
    normalized = re.sub(r"[^\w\s]", "", text.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def main():
    parser = argparse.ArgumentParser(
        description="Extract FAQs from PDF files using DeepSeek API"
    )
    parser.add_argument("input_folder", help="Path to the folder containing PDF files")
    parser.add_argument(
        "output_folder", help="Path to the folder where JSON file will be saved"
    )
    parser.add_argument(
        "--base-url", help="Base URL for DeepSeek API", default="http://localhost:11434"
    )
    parser.add_argument("--model", help="Model name to use", default="qwen2.5:custom")
    parser.add_argument(
        "--timeout", help="API request timeout in seconds", type=int, default=120
    )
    parser.add_argument(
        "--log-level",
        help="Logging level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    args = parser.parse_args()

    # Configure logging level
    logger.setLevel(getattr(logging, args.log_level))

    # Validate input folder
    if not os.path.exists(args.input_folder):
        logger.error(f"Input folder does not exist: {args.input_folder}")
        return 1

    # Create output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    # Initialize processor
    deepseek_processor = DeepSeekProcessor(
        base_url=args.base_url, model_name=args.model, timeout=args.timeout
    )

    # Get all PDF files
    pdf_files = glob.glob(os.path.join(args.input_folder, "*.pdf"))
    if not pdf_files:
        logger.error(f"No PDF files found in {args.input_folder}")
        return 1

    logger.info(f"Found {len(pdf_files)} PDF files to process")

    # Process PDFs and extract FAQs
    all_faqs = extract_and_clean_faqs(pdf_files, deepseek_processor)

    if not all_faqs:
        logger.warning("No FAQs were extracted from any of the PDFs")
        return 1

    # Save results to JSON
    output_json_path = os.path.join(args.output_folder, "all_faqs.json")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(all_faqs, f, indent=2, ensure_ascii=False)

    logger.info(f"Successfully extracted {len(all_faqs)} FAQs")
    logger.info(f"Results saved to {output_json_path}")
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
