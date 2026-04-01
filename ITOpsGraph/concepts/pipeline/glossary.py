"""
Glossary parser for extracting terms and definitions from product glossary pages.

Supports multiple HTML formats:
- IBM docs: <dt class="dlterm"> / <dd> pairs
- HashiCorp docs: <h2> / <p> pairs
- Instana docs: <h3> / <p> pairs
- Generic: Various patterns

Creates GlossaryTerm objects that can be loaded into Neo4j as authoritative
concept definitions.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class GlossaryTerm:
    """A glossary term with its definition."""
    term: str
    definition: str
    product: str
    source_file: str
    aliases: list[str] = field(default_factory=list)
    category: str = ""  # e.g., "A", "B" for alphabetical sections
    related_terms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "term": self.term,
            "definition": self.definition,
            "product": self.product,
            "source_file": self.source_file,
            "aliases": self.aliases,
            "category": self.category,
            "related_terms": self.related_terms,
        }

    @property
    def normalized_term(self) -> str:
        """Normalized term for matching."""
        return self.term.lower().strip()


@dataclass
class GlossaryMetrics:
    """Metrics from glossary parsing."""
    files_processed: int = 0
    terms_extracted: int = 0
    terms_by_product: dict = field(default_factory=dict)
    parse_errors: int = 0

    def to_dict(self) -> dict:
        return {
            "files_processed": self.files_processed,
            "terms_extracted": self.terms_extracted,
            "terms_by_product": self.terms_by_product,
            "parse_errors": self.parse_errors,
        }


class GlossaryParser:
    """
    Parses glossary HTML files to extract terms and definitions.

    Supports multiple formats:
    - IBM Documentation (dt/dd definition lists)
    - HashiCorp Documentation (h2/p pairs)
    - Generic patterns
    """

    # Skip these terms (navigation, UI elements, etc.)
    SKIP_TERMS = {
        'documentation', 'resources', 'overview', 'glossary', 'navigation',
        'rate this content', 'provide more feedback', 'thank you for your submission',
        'error submitting rating', 'feedback', 'home', 'search', 'menu',
        'table of contents', 'previous', 'next', 'back to top',
    }

    def __init__(self):
        self.metrics = GlossaryMetrics()

    def parse_file(self, file_path: Path) -> list[GlossaryTerm]:
        """Parse a glossary HTML file and extract terms."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            logger.error("beautifulsoup4 required: pip install beautifulsoup4")
            raise

        terms = []
        product = self._extract_product_name(file_path)

        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')

        # Try different parsing strategies
        # Strategy 1: IBM dt/dd definition lists
        dt_terms = self._parse_dt_dd(soup, product, str(file_path))
        if dt_terms:
            terms.extend(dt_terms)
            logger.info(f"Parsed {len(dt_terms)} terms from {file_path.name} (dt/dd format)")

        # Strategy 2: H2/P pairs (HashiCorp)
        if not terms:
            h2_terms = self._parse_h2_p(soup, product, str(file_path))
            if h2_terms:
                terms.extend(h2_terms)
                logger.info(f"Parsed {len(h2_terms)} terms from {file_path.name} (h2/p format)")

        # Strategy 3: H3/P pairs (Instana)
        if not terms:
            h3_terms = self._parse_h3_p(soup, product, str(file_path))
            if h3_terms:
                terms.extend(h3_terms)
                logger.info(f"Parsed {len(h3_terms)} terms from {file_path.name} (h3/p format)")

        # Strategy 4: Generic bold/paragraph patterns
        if not terms:
            generic_terms = self._parse_generic(soup, product, str(file_path))
            if generic_terms:
                terms.extend(generic_terms)
                logger.info(f"Parsed {len(generic_terms)} terms from {file_path.name} (generic format)")

        self.metrics.files_processed += 1
        self.metrics.terms_extracted += len(terms)
        self.metrics.terms_by_product[product] = len(terms)

        if not terms:
            logger.warning(f"No terms found in {file_path.name}")

        return terms

    def _extract_product_name(self, file_path: Path) -> str:
        """Extract product name from filename."""
        name = file_path.stem
        # Clean up common suffixes
        for suffix in ['_glossary', '_files', '_observability', '_application_resource_management',
                       '_cost_dimensions_and_metrics', '_utlization_dimensions_and_metrics']:
            name = name.replace(suffix, '')
        # Convert to readable name
        name = name.replace('_', ' ').strip()
        return name

    def _should_skip(self, term: str) -> bool:
        """Check if term should be skipped."""
        term_lower = term.lower().strip()
        if term_lower in self.SKIP_TERMS:
            return True
        if len(term_lower) < 2:
            return True
        if term_lower.startswith('error') or term_lower.startswith('rate '):
            return True
        return False

    def _clean_text(self, text: str) -> str:
        """Clean up extracted text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove common artifacts
        text = text.strip()
        return text

    def _parse_dt_dd(self, soup, product: str, source: str) -> list[GlossaryTerm]:
        """Parse IBM-style definition lists."""
        terms = []
        dts = soup.find_all('dt', class_=lambda x: x and 'dlterm' in str(x))

        for dt in dts:
            term_text = self._clean_text(dt.get_text())
            if self._should_skip(term_text):
                continue

            dd = dt.find_next_sibling('dd')
            definition = self._clean_text(dd.get_text()) if dd else ""

            if term_text and definition:
                # Extract aliases (terms in parentheses)
                aliases = []
                if '(' in term_text:
                    match = re.search(r'\(([^)]+)\)', term_text)
                    if match:
                        alias = match.group(1)
                        aliases.append(alias)
                        # Also keep the full term as-is

                terms.append(GlossaryTerm(
                    term=term_text,
                    definition=definition,
                    product=product,
                    source_file=source,
                    aliases=aliases,
                ))

        return terms

    def _parse_h2_p(self, soup, product: str, source: str) -> list[GlossaryTerm]:
        """Parse HashiCorp-style h2/p pairs."""
        terms = []
        h2s = soup.find_all('h2')

        for h2 in h2s:
            term_text = self._clean_text(h2.get_text())
            if self._should_skip(term_text):
                continue

            # Get next paragraph
            next_p = h2.find_next_sibling('p')
            definition = self._clean_text(next_p.get_text()) if next_p else ""

            # Skip if definition is too short or looks like navigation
            if len(definition) < 20:
                continue

            if term_text and definition:
                terms.append(GlossaryTerm(
                    term=term_text,
                    definition=definition,
                    product=product,
                    source_file=source,
                ))

        return terms

    def _parse_h3_p(self, soup, product: str, source: str) -> list[GlossaryTerm]:
        """Parse Instana-style h3/p pairs."""
        terms = []
        h3s = soup.find_all('h3')

        for h3 in h3s:
            term_text = self._clean_text(h3.get_text())
            if self._should_skip(term_text):
                continue

            # Get next paragraph or div
            next_elem = h3.find_next(['p', 'div'])
            if not next_elem:
                continue

            definition = self._clean_text(next_elem.get_text())

            # Skip if definition is too short
            if len(definition) < 20:
                continue

            if term_text and definition:
                terms.append(GlossaryTerm(
                    term=term_text,
                    definition=definition,
                    product=product,
                    source_file=source,
                ))

        return terms

    def _parse_generic(self, soup, product: str, source: str) -> list[GlossaryTerm]:
        """Parse generic patterns (bold terms, etc.)."""
        terms = []

        # Try finding strong/b tags followed by text
        for strong in soup.find_all(['strong', 'b']):
            term_text = self._clean_text(strong.get_text())
            if self._should_skip(term_text) or len(term_text) > 50:
                continue

            # Get surrounding text as definition
            parent = strong.parent
            if parent:
                full_text = parent.get_text()
                # Remove the term from the definition
                definition = full_text.replace(term_text, '').strip()
                definition = self._clean_text(definition)

                if len(definition) > 30:
                    terms.append(GlossaryTerm(
                        term=term_text,
                        definition=definition,
                        product=product,
                        source_file=source,
                    ))

        return terms

    def parse_directory(self, directory: Path) -> list[GlossaryTerm]:
        """Parse all HTML files in a directory."""
        all_terms = []

        for html_file in directory.glob('*.html'):
            try:
                terms = self.parse_file(html_file)
                all_terms.extend(terms)
            except Exception as e:
                logger.error(f"Error parsing {html_file}: {e}")
                self.metrics.parse_errors += 1

        return all_terms


def parse_glossary_files(
    input_dir: Path,
    output_file: Path,
) -> GlossaryMetrics:
    """
    Parse glossary HTML files and save extracted terms.

    Args:
        input_dir: Directory containing glossary HTML files
        output_file: Output JSON file for extracted terms

    Returns:
        GlossaryMetrics with parsing statistics
    """
    import json

    parser = GlossaryParser()
    terms = parser.parse_directory(input_dir)

    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "created_at": datetime.now().isoformat(),
        "source_dir": str(input_dir),
        "metrics": parser.metrics.to_dict(),
        "terms": [t.to_dict() for t in terms],
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Saved {len(terms)} glossary terms to {output_file}")
    return parser.metrics


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    parser = argparse.ArgumentParser(description="Parse glossary HTML files")
    parser.add_argument("--input", "-i", required=True, help="Input directory with HTML files")
    parser.add_argument("--output", "-o", required=True, help="Output JSON file")

    args = parser.parse_args()

    metrics = parse_glossary_files(
        input_dir=Path(args.input),
        output_file=Path(args.output),
    )

    print("\nGlossary Parsing Summary:")
    print(f"  Files processed: {metrics.files_processed}")
    print(f"  Terms extracted: {metrics.terms_extracted}")
    print(f"  Parse errors: {metrics.parse_errors}")
    print(f"\nTerms by product:")
    for product, count in sorted(metrics.terms_by_product.items()):
        print(f"  {product}: {count}")
