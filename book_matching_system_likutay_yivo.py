from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def tfidf_similarity(text1: str, text2: str) -> float:
    if not text1 or not text2:
        return 0.0

    text1 = str(text1).strip()
    text2 = str(text2).strip()

    if len(text1) < 2 or len(text2) < 2:
        return 0.0

    corpus = [text1, text2]
    try:
        vectorizer = TfidfVectorizer().fit(corpus)
        vectors = vectorizer.transform(corpus)
        sim = cosine_similarity(vectors[0], vectors[1])[0][0]
        return float(sim)
    except ValueError:
        return 0.0



"""
Book Matching System - Likutei Shoshanim to YIVO
=================================================
Comprehensive system for matching book records between Hebrew and Yiddish catalogs
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass
from enum import Enum

# External libraries - install with:
# pip install rapidfuzz pyluach
from rapidfuzz import fuzz, process
import pyluach  # For Hebrew dates


# ========================================
# 1. Data Preprocessing
# ========================================

class HebrewYearConverter:
    """Converts Hebrew years to Gregorian"""

    def __init__(self):
        # Basic Hebrew numerals dictionary
        self.hebrew_numerals = {
            'א': 1, 'ב': 2, 'ג': 3, 'ד': 4, 'ה': 5,
            'ו': 6, 'ז': 7, 'ח': 8, 'ט': 9, 'י': 10,
            'כ': 20, 'ל': 30, 'מ': 40, 'נ': 50, 'ס': 60,
            'ע': 70, 'פ': 80, 'צ': 90, 'ק': 100, 'ר': 200,
            'ש': 300, 'ת': 400
        }

    def hebrew_to_gregorian(self, hebrew_year: str) -> Optional[int]:
        """Convert Hebrew year to Gregorian"""
        if not hebrew_year or pd.isna(hebrew_year):
            return None

        try:
            hebrew_year = str(hebrew_year).strip().replace('"', '').replace("'", "")
            # If it's already a Gregorian year
            if hebrew_year.isdigit():
                return int(hebrew_year)

            # Calculate numeric value
            total = 0
            for char in hebrew_year:
                if char in self.hebrew_numerals:
                    total += self.hebrew_numerals[char]

            # Add 5000 if year is small (assuming regular Hebrew year)
            if total < 1000:
                total += 5000

            # Convert to Gregorian (approximately)
            gregorian_year = total - 3760
            return gregorian_year if 1450 <= gregorian_year <= 2024 else None

        except Exception as e:
            print(f"Error converting Hebrew year {hebrew_year}: {e}")
            return None


class DateParser:
    """Parses dates in various formats"""

    @staticmethod


    def parse_yivo_date(date_str: str) -> Optional[int]:
        if not date_str or pd.isna(date_str):
            return None
        date_str = str(date_str).strip()
        date_str = re.sub(r'[\[\]\?]', '', date_str)
        match = re.search(r'(\d{4})-(\d{4})', date_str)
        if match:
            start, end = int(match.group(1)), int(match.group(2))
            return (start + end) // 2
        match = re.search(r'(\d{4})', date_str)
        if match:
            year = int(match.group(1))
            if 1450 <= year <= 2024:
                return year
        return None

        date_str = str(date_str).strip()

        # Format [1930]
        if '[' in date_str and ']' in date_str:
            match = re.search(r'\[(\d{4})\]', date_str)
            if match:
                return int(match.group(1))

        # Range format 1919-1925
        if '-' in date_str:
            match = re.search(r'(\d{4})-\d{4}', date_str)
            if match:
                return int(match.group(1))

        # Regular year format
        match = re.search(r'(\d{4})', date_str)
        if match:
            year = int(match.group(1))
            if 1450 <= year <= 2024:
                return year

        return None


class PlaceNormalizer:
    """Normalizes publication place names"""

    def __init__(self):
        self.place_mapping = {
            'new york': ['new york', 'ny', 'n.y.', 'niu york', 'nyu york', 'ניו יארק', 'ניו-יארק', 'ניו יורק'],
            'chicago': ['chicago', 'shikage', 'shiḳage', 'שיקאגע', 'שיקאגא', 'שיקגו'],
            'vilna': ['vilna', 'vilnius', 'wilno', 'ווילנא', 'וילנא', 'ווילנה', 'וילנה'],
            'warsaw': ['warsaw', 'varsha', 'warszawa', 'ווארשא', 'וארשא', 'ווארשע', 'ורשה'],
            'lemberg': ['lemberg', 'lwow', 'lviv', 'לעמבערג', 'למברג', 'לבוב'],
            'amsterdam': ['amsterdam', 'אמשטרדם', 'אמסטרדם'],
            'venice': ['venice', 'venezia', 'ויניציאה', 'וינציה', 'ונציה'],
            'constantinople': ['constantinople', 'istanbul', 'קושטאנדינה', 'קושטנדינא', 'קושטא'],
            'odessa': ['odessa', 'אדעסא', 'אודעסא', 'אודסה'],
            'breslau': ['breslau', 'wroclaw', 'ברעסלויא', 'ברסלאו'],
            'pisa': ['pisa', 'פיסא', 'פיזה'],
            'jerusalem': ['jerusalem', 'ירושלים', 'ירושלם'],
            'poznan': ['poznan', 'posen', 'פוזנא', 'פוזנאן'],
        }

        # Create reverse mapping
        self.reverse_mapping = {}
        for standard, variants in self.place_mapping.items():
            for variant in variants:
                self.reverse_mapping[variant.lower()] = standard
            self.reverse_mapping[standard.lower()] = standard

    def normalize(self, place: str) -> str:
        """Normalize place name"""
        if not place or pd.isna(place):
            return ""

        place = str(place).lower().strip()

        # Search in dictionary
        if place in self.reverse_mapping:
            return self.reverse_mapping[place]

        # Partial search
        for key, value in self.reverse_mapping.items():
            if key in place or place in key:
                return value

        return place

    def extract_place_from_publisher(self, publisher_str: str) -> str:
        """Extract place from publisher string"""
        if not publisher_str or pd.isna(publisher_str):
            return ""

        publisher_str = str(publisher_str).lower()

        # Search for known place names
        for place in self.reverse_mapping.keys():
            if place in publisher_str:
                return self.normalize(place)

        # Try to extract using common patterns
        # Pattern: "city : publisher"
        match = re.search(r'^([^:]+):', publisher_str)
        if match:
            potential_place = match.group(1).strip()
            return self.normalize(potential_place)

        return ""


# ========================================
# 2. Text Matching Strategies
# ========================================

class TextNormalizer:
    """Normalizes texts for comparison"""

    @staticmethod
    def normalize_hebrew_yiddish(text: str) -> str:
        """Normalize Hebrew/Yiddish text"""
        if not text or pd.isna(text):
            return ""

        text = str(text).strip()

        # Remove nikud (Hebrew vowel marks)
        text = re.sub(r'[\u0591-\u05C7]', '', text)

        # Convert final letters
        replacements = {
            'ך': 'כ', 'ם': 'מ', 'ן': 'נ', 'ף': 'פ', 'ץ': 'צ',
            # Yiddish letters
            'אָ': 'א', 'וו': 'ו', 'יי': 'י', 'ײ': 'יי', 'ױ': 'וי'
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        # Remove punctuation and extra spaces
        text = re.sub(r'[.,;:!?״"\'״]', ' ', text)
        text = ' '.join(text.split())

        return text.lower()

    @staticmethod
    def extract_core_title(title: str) -> str:
        """Extract core title without additions"""
        if not title:
            return ""

        # Remove common patterns
        remove_patterns = [
            r'\(.*?\)',  # Content in parentheses
            r'ספר\s+',  # "sefer" (book)
            r'על\s+מסכת.*',  # "on tractate..."
            r'חלק\s+[א-ת]',  # "part X"
            r'כרך\s+[א-ת]',  # "volume X"
        ]

        title = TextNormalizer.normalize_hebrew_yiddish(title)

        for pattern in remove_patterns:
            title = re.sub(pattern, '', title)

        return title.strip()


class AuthorNormalizer:
    """Normalizes author names"""

    def __init__(self):
        self.title_prefixes = [
            'הרב', 'רבי', 'ר׳', 'ה״ר', 'הר״ר', 'הגאון', 'האדמו״ר',
            'רב', 'מוהר״ר', 'כש״ת', 'הרה״ג', 'הרה״צ', 'rabbi', 'rav', 'harav'
        ]

    def normalize(self, author: str) -> str:
        """Normalize author name"""
        if not author or pd.isna(author):
            return ""

        author = str(author).strip()

        # Remove titles
        for prefix in self.title_prefixes:
            author = re.sub(f'^{prefix}\\s+', '', author, flags=re.IGNORECASE)
            author = re.sub(f'\\s+{prefix}\\s+', ' ', author, flags=re.IGNORECASE)

        # General normalization
        author = TextNormalizer.normalize_hebrew_yiddish(author)

        # Remove suffixes like "z"l", "ztz"l"
        author = re.sub(r'\s+ז[״״]ל$', '', author)
        author = re.sub(r'\s+זצ[״״]ל$', '', author)

        return author.strip()

    def extract_author_from_text(self, text: str) -> str:
        """Extract author name from text"""
        if not text:
            return ""

        # Search for common patterns
        patterns = [
            r'ה[״״]ר\s+([^\.]+?)[\.,]',
            r'מאת\s+([^\.]+?)[\.,]',
            r'חיבר\s+([^\.]+?)[\.,]',
            r'מחבר[:\s]+([^\.]+?)[\.,]',
            r'by\s+([^\.]+?)[\.,]',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return self.normalize(match.group(1))

        return ""


# ========================================
# 3. Matching System
# ========================================

class BookMatcher:
    """Book matching system using fuzzy matching only"""

    def __init__(self):
        print("Initializing Book Matcher...")

        # Initialize components
        self.hebrew_converter = HebrewYearConverter()
        self.date_parser = DateParser()
        self.place_normalizer = PlaceNormalizer()
        self.text_normalizer = TextNormalizer()
        self.author_normalizer = AuthorNormalizer()

        # Settings and thresholds
        self.weights = {
            'title': 0.40,  # Title similarity
            'author': 0.30,  # Author similarity
            'year': 0.20,  # Year similarity
            'place': 0.10  # Place similarity
        }

        self.thresholds = {
            'certain': 0.85,  # Certain match
            'probable': 0.70,  # Probable match
            'possible': 0.55,  # Possible match
            'minimum': 0.45  # Minimum threshold
        }

    def preprocess_likutei_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess Likutei Shoshanim data"""
        print("Preprocessing Likutei Shoshanim data...")

        # Normalize titles
        df['normalized_title'] = df['Book Title'].apply(self.text_normalizer.extract_core_title)

        # Normalize authors
        df['normalized_author'] = df['Author'].apply(self.author_normalizer.normalize)

        # Normalize places
        df['normalized_place'] = df['Publishing Place'].apply(self.place_normalizer.normalize)

        # Convert Hebrew years
        df['converted_year'] = df.apply(
            lambda row: row['Gregorian Calendar Year'] if pd.notna(row['Gregorian Calendar Year'])
            else self.hebrew_converter.hebrew_to_gregorian(row['Hebrew Calendar Year']),
            axis=1
        )

        return df

    def preprocess_yivo_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess YIVO data"""
        print("Preprocessing YIVO data...")

        # Normalize titles
        df['normalized_title'] = df['vertitle'].apply(self.text_normalizer.extract_core_title)

        # Normalize authors
        df['normalized_author'] = df['creator'].apply(self.author_normalizer.normalize)

        # Parse dates
        df['parsed_year'] = df['creationdate'].apply(self.date_parser.parse_yivo_date)

        # Extract places from publisher
        df['extracted_place'] = df['publisher'].apply(self.place_normalizer.extract_place_from_publisher)
        df['normalized_place'] = df['extracted_place'].apply(self.place_normalizer.normalize)

        return df

    def calculate_fuzzy_similarity(self, text1: str, text2: str) -> float:
        return tfidf_similarity(text1, text2)

    def calculate_year_similarity(self, year1: Optional[int], year2: Optional[int], tolerance: int = 5) -> float:
        """Calculate similarity between years"""
        if year1 is None or year2 is None:
            return 0.0

        diff = abs(year1 - year2)
        if diff == 0:
            return 1.0
        elif diff <= tolerance:
            return 1.0 - (diff / (tolerance * 2))
        else:
            return 0.0

    def calculate_match_score(self, likutei_row: pd.Series, yivo_row: pd.Series) -> Dict:
        """Calculate overall match score"""
        scores = {}

        # Title similarity
        title1 = likutei_row.get('normalized_title', '')
        title2 = yivo_row.get('normalized_title', '')
        scores['title'] = self.calculate_fuzzy_similarity(title1, title2)

        # Author similarity
        author1 = likutei_row.get('normalized_author', '')
        author2 = yivo_row.get('normalized_author', '')
        scores['author'] = self.calculate_fuzzy_similarity(author1, author2)

        # Year similarity
        year1 = likutei_row.get('converted_year')
        year2 = yivo_row.get('parsed_year')
        scores['year'] = self.calculate_year_similarity(year1, year2)

        # Place similarity
        place1 = likutei_row.get('normalized_place', '')
        place2 = yivo_row.get('normalized_place', '')
        scores['place'] = self.calculate_fuzzy_similarity(place1, place2)

        # Calculate weighted score
        total_score = sum(scores[key] * self.weights[key] for key in scores)

        # Determine confidence level
        if total_score >= self.thresholds['certain']:
            confidence = 'certain'
        elif total_score >= self.thresholds['probable']:
            confidence = 'probable'
        elif total_score >= self.thresholds['possible']:
            confidence = 'possible'
        else:
            confidence = 'low'

        return {
            'total_score': total_score,
            'confidence': confidence,
            'detailed_scores': scores
        }

    def find_matches(self, likutei_df: pd.DataFrame, yivo_df: pd.DataFrame) -> pd.DataFrame:
        """Find matches between files"""
        print(f"Finding matches for {len(likutei_df)} Likutei books in {len(yivo_df)} YIVO books...")

        # Preprocess data
        likutei_df = self.preprocess_likutei_data(likutei_df)
        yivo_df = self.preprocess_yivo_data(yivo_df)

        # Create result columns
        yivo_df['likutay_index'] = ''
        yivo_df['match_confidence'] = ''
        yivo_df['match_score'] = 0.0

        # Find matches
        matches_found = 0

        for yivo_idx in range(len(yivo_df)):
            if yivo_idx % 100 == 0:
                print(f"Processing YIVO book {yivo_idx}/{len(yivo_df)}...")

            yivo_row = yivo_df.iloc[yivo_idx]

            # Quick filter: skip if no title
            if not yivo_row.get('normalized_title'):
                continue

            best_match = None
            best_score = 0

            # Check against all Likutei books
            for likutei_idx in range(len(likutei_df)):
                likutei_row = likutei_df.iloc[likutei_idx]

                # Quick pre-filter using title similarity
                title_quick_score = fuzz.ratio(
                    yivo_row.get('normalized_title', ''),
                    likutei_row.get('normalized_title', '')
                ) / 100

                # Skip if title is too different
                if title_quick_score < 0.3:
                    continue

                # Calculate full score
                match_result = self.calculate_match_score(likutei_row, yivo_row)

                if match_result['total_score'] > best_score and match_result['total_score'] >= self.thresholds[
                    'minimum']:
                    best_score = match_result['total_score']
                    best_match = {
                        'index': likutei_row['Index'],
                        'confidence': match_result['confidence'],
                        'score': match_result['total_score'],
                        'details': match_result['detailed_scores']
                    }

            # Update results
            if best_match:
                yivo_df.at[yivo_idx, 'likutay_index'] = best_match['index']
                yivo_df.at[yivo_idx, 'match_confidence'] = best_match['confidence']
                yivo_df.at[yivo_idx, 'match_score'] = best_match['score']
                matches_found += 1

        print(f"\nMatching complete! Found {matches_found} matches")
        return yivo_df


# ========================================
# 4. Quality Control and Reporting
# ========================================

class QualityControl:
    """Quality control and reporting system"""

    @staticmethod
    def generate_match_report(yivo_df: pd.DataFrame) -> Dict:
        """Generate match report"""
        total_books = len(yivo_df)
        matched_books = len(yivo_df[yivo_df['likutay_index'] != ''])

        confidence_counts = yivo_df[yivo_df['likutay_index'] != '']['match_confidence'].value_counts()

        report = {
            'summary': {
                'total_yivo_books': total_books,
                'matched_books': matched_books,
                'unmatched_books': total_books - matched_books,
                'match_rate': f"{(matched_books / total_books) * 100:.2f}%" if total_books > 0 else "0%"
            },
            'confidence_distribution': {
                'certain': int(confidence_counts.get('certain', 0)),
                'probable': int(confidence_counts.get('probable', 0)),
                'possible': int(confidence_counts.get('possible', 0))
            },
            'score_statistics': {}
        }

        # Calculate statistics only if there are matches
        if matched_books > 0:
            matched_scores = yivo_df[yivo_df['match_score'] > 0]['match_score']
            report['score_statistics'] = {
                'mean_score': float(matched_scores.mean()),
                'median_score': float(matched_scores.median()),
                'min_score': float(matched_scores.min()),
                'max_score': float(matched_scores.max())
            }

        return report

    @staticmethod
    def export_for_review(yivo_df: pd.DataFrame, output_path: str = 'matches_for_review.xlsx'):
        """Export for manual review"""
        # Create DataFrame for review
        review_df = yivo_df[yivo_df['likutay_index'] != ''].copy()

        # Sort by confidence and score
        review_df = review_df.sort_values(['match_confidence', 'match_score'],
                                          ascending=[True, False])

        # Select relevant columns
        review_columns = [
            'likutay_index', 'match_confidence', 'match_score',
            'vertitle', 'creator', 'creationdate', 'publisher',
            'normalized_title', 'normalized_author', 'parsed_year', 'normalized_place'
        ]

        # Keep only existing columns
        review_columns = [col for col in review_columns if col in review_df.columns]
        review_df = review_df[review_columns]

        # Export to Excel with different sheets for each confidence level
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                for confidence in ['certain', 'probable', 'possible']:
                    conf_df = review_df[review_df['match_confidence'] == confidence]
                    if not conf_df.empty:
                        conf_df.to_excel(writer, sheet_name=confidence, index=False)
            print(f"Review file exported to {output_path}")
        except Exception as e:
            print(f"Could not export to Excel: {e}")
            # Fall back to CSV
            review_df.to_csv(output_path.replace('.xlsx', '.csv'), index=False)
            print(f"Review file exported to {output_path.replace('.xlsx', '.csv')}")


# ========================================
# Main Function
# ========================================

def main(likutei_path: str, yivo_path: str, output_path: str):
    """Main function to run the system"""

    print("=" * 50)
    print("Book Matching System - Starting")
    print("=" * 50)

    # Load data
    print("\n1. Loading data files...")
    try:
        likutei_df = pd.read_csv(likutei_path, encoding='utf-8')
    except:
        likutei_df = pd.read_csv(likutei_path, encoding='utf-8-sig')

    try:
        yivo_df = pd.read_csv(yivo_path, encoding='utf-8')
    except:
        yivo_df = pd.read_csv(yivo_path, encoding='utf-8-sig')

    print(f"   Loaded {len(likutei_df)} Likutei Shoshanim books")
    print(f"   Loaded {len(yivo_df)} YIVO books")

    # Create matcher
    print("\n2. Initializing matching system...")
    matcher = BookMatcher()

    # Perform matching
    print("\n3. Finding matches...")
    result_df = matcher.find_matches(likutei_df, yivo_df)

    # Quality control
    print("\n4. Quality control...")
    qc = QualityControl()

    # Generate report
    report = qc.generate_match_report(result_df)

    print("\n" + "=" * 50)
    print("MATCHING REPORT")
    print("=" * 50)
    print(f"Total YIVO books: {report['summary']['total_yivo_books']}")
    print(f"Matched books: {report['summary']['matched_books']}")
    print(f"Match rate: {report['summary']['match_rate']}")
    print("\nConfidence Distribution:")
    for level, count in report['confidence_distribution'].items():
        print(f"  {level}: {count}")
    if report['score_statistics']:
        print("\nScore Statistics:")
        for stat, value in report['score_statistics'].items():
            if not pd.isna(value):
                print(f"  {stat}: {value:.3f}")

    # Save results
    print(f"\n5. Saving results to {output_path}...")

    # Keep only necessary columns
    columns_to_keep = list(yivo_df.columns)
    if 'likutay_index' not in columns_to_keep:
        columns_to_keep.append('likutay_index')

    # Save with new column only
    final_df = result_df[columns_to_keep]
    final_df.to_csv(output_path, index=False, encoding='utf-8')

    # Export for review
    review_path = output_path.replace('.csv', '_review.xlsx')
    qc.export_for_review(result_df, review_path)

    # Save report
    report_path = output_path.replace('.csv', '_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Process complete!")
    print(f"   Results saved to: {output_path}")
    print(f"   Review file: {review_path}")
    print(f"   Report: {report_path}")

    return result_df


# ========================================
# Run Script
# ========================================

if __name__ == "__main__":
    # Set file paths
    LIKUTEI_PATH = "/Users/noamkashi/Documents/yivo_likutay/likutei_shoshanim.csv"
    YIVO_PATH = "/Users/noamkashi/Documents/yivo_likutay/yivo_vilna_books.csv"
    OUTPUT_PATH = '/Users/noamkashi/Documents/yivo_likutay/yivo_with_matches_2.csv'

    # Run
    result = main(LIKUTEI_PATH, YIVO_PATH, OUTPUT_PATH)