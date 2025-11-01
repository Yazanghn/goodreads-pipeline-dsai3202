# goodreads-pipeline-dsai3202

1. Features to Keep
After analyzing feature importance and interpretability in Lecture 05, I decided to retain features that capture core textual, semantic, and sentiment information relevant for downstream models.

Kept Features:

Category	Features
Structural	review_length_raw, review_char_count, sentence_count, paragraph_count, punct_count
Lexical / Readability	lexical_diversity, readability_grade
Sentiment	vader_pos, vader_neg, vader_neu, blob_polarity, blob_subjectivity
TF-IDF Stats	tfidf_mean, tfidf_max
Topics	topic1_prob, topic2_prob
Metadata	rating_count, n_votes
Contextual Embeddings	distilbert_embedding

These features balance linguistic structure, readability, and sentiment tone with deeper semantic embeddings.

2. Features to Add

To enrich interpretability and temporal context, I added four derived features:

New Feature	Description
book_age	Difference between current year and publication_year → captures recency
word_count	Total number of tokens per review
avg_word_len	Average word length → style and complexity
has_emoji	Binary flag indicating presence of emojis in review text

These features give quantitative insight into writing style, emotional tone, and publication recency.

3. Features to Remove

I removed redundant, low-signal, or leakage-prone variables that add noise but little predictive power:

Category	Removed Features	Reason
Low-signal counts	quote_count, exclamation_count, question_count, caps_count	Highly correlated with punctuation metrics
Redundant TF-IDF stat	tfidf_min	Minimal variance and overlaps with TF-IDF mean and max
Text metadata	review_text, title, author_names	Raw text handled separately by embeddings
IDs and timestamps	book_id, user_id, review_id, date_added	Not useful for model features / risk of leakage
Zero-shot label	zeroshot_label	Non-numeric categorical text labels — score retained instead


4. Script Update Summary

To reflect these design choices, the final script (goodreads_text_features_final.py) includes:

A derived_features() function computing the four new features.

A PRUNE_COLS list removing the dropped ones before output.

Updated output prefix → features_enriched_full.




